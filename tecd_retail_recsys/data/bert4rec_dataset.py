"""
BERT4Rec Dataset Builder with Rich Features

Создает датасет для BERT4Rec со следующими фичами:
- Item features: brand, category, subcategory
- Price features: price buckets, price tier, price relative to category
- Item embeddings: предобученные эмбеддинги товаров
- Temporal features: day of week, hour of day, time of day
- User features: interaction count, avg price, favorite categories/brands

Usage:
    from tecd_retail_recsys.data.bert4rec_dataset import BERT4RecDatasetBuilder
    
    builder = BERT4RecDatasetBuilder(train_df)
    dataset, item_net_config = builder.build_dataset(
        use_item_embeddings=True,
        use_price_features=True,
        use_temporal_features=False,
        use_user_features=True
    )
    
    # Используем в модели
    model = BERT4RecModel(
        item_net_block_types=item_net_config['item_net_block_types'],
        ...
    )
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict, Optional, List, Any
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models.nn.item_net import IdEmbeddingsItemNet, CatFeaturesItemNet, ItemNetBase


class PretrainedEmbeddingsItemNet(ItemNetBase):
    """ItemNet с предобученными эмбеддингами товаров"""
    
    # Глобальное хранилище для эмбеддингов (чтобы передать через from_dataset)
    _embeddings_storage = {}
    
    def __init__(self, embeddings_matrix: np.ndarray, output_dim: int):
        super().__init__()
        
        # Создаем Embedding layer из предобученных эмбеддингов
        # ВАЖНО: явно указываем device='cpu' для совместимости с MPS
        embeddings_tensor = torch.FloatTensor(embeddings_matrix)
        
        self.embeddings = torch.nn.Embedding.from_pretrained(
            embeddings_tensor,
            freeze=True  # не Разрешаем дообучение
        )
        
        # Проекция на output_dim если размеры не совпадают
        emb_dim = embeddings_matrix.shape[1]
        if emb_dim != output_dim:
            self.projection = torch.nn.Linear(emb_dim, output_dim)
        else:
            self.projection = None
            
    def forward(self, batch):
        # Извлекаем item_ids из batch
        # batch может быть словарем с 'item_id_encoded' или просто тензором
        if isinstance(batch, dict):
            item_ids = batch.get('item_id_encoded', batch.get('item_id'))
        else:
            item_ids = batch
        
        # Убеждаемся что item_ids на том же device что и embeddings
        if item_ids.device != self.embeddings.weight.device:
            item_ids = item_ids.to(self.embeddings.weight.device)
        
        # Получаем эмбеддинги
        emb = self.embeddings(item_ids)
        
        # Проецируем если нужно
        if self.projection is not None:
            emb = self.projection(emb)
            
        return emb
    
    def to(self, device):
        """Переносим все компоненты на нужное устройство"""
        super().to(device)
        if self.embeddings is not None:
            self.embeddings = self.embeddings.to(device)
        if self.projection is not None:
            self.projection = self.projection.to(device)
        return self
    
    @classmethod
    def from_dataset(cls, dataset, n_factors, dropout_rate=0.0, **kwargs):
        """
        Создает экземпляр из глобального хранилища эмбеддингов
        """
        if 'pretrained_embeddings' in cls._embeddings_storage:
            embeddings_matrix = cls._embeddings_storage['pretrained_embeddings']
            return cls(embeddings_matrix=embeddings_matrix, output_dim=n_factors)
        return None
    
    @classmethod
    def set_embeddings(cls, embeddings_matrix: np.ndarray):
        """Сохраняет эмбеддинги в глобальное хранилище"""
        cls._embeddings_storage['pretrained_embeddings'] = embeddings_matrix
    
    @classmethod
    def clear_embeddings(cls):
        """Очищает глобальное хранилище"""
        cls._embeddings_storage.clear()


class BERT4RecDatasetBuilder:
    """
    Построитель датасета для BERT4Rec с богатыми фичами
    
    Args:
        train_df: DataFrame с колонками [user_id, item_id, timestamp, ...]
                  Может содержать: item_brand_id, item_category, item_subcategory,
                                   item_price, item_embedding
    """
    
    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df.copy()
        self.item_features_list = []
        self.user_features_list = []
        self.embeddings_matrix = None
        self.pretrained_net = None
        
    def _prepare_interactions(self) -> pd.DataFrame:
        """Подготовка interactions для RecTools"""
        interactions = self.train_df[['user_id', 'item_id', 'timestamp']].copy()
        interactions.columns = [Columns.User, Columns.Item, Columns.Datetime]
        interactions[Columns.Weight] = 1
        return interactions
    
    def _add_basic_item_features(self) -> None:
        """Добавляем базовые фичи: brand, category, subcategory"""
        print("📦 Добавление базовых item features...")
        
        # Brand
        if 'item_brand_id' in self.train_df.columns:
            brand_feature = self.train_df[['item_id', 'item_brand_id']].drop_duplicates()
            brand_feature.columns = ['id', 'value']
            brand_feature['feature'] = 'brand'
            brand_feature = brand_feature[brand_feature['value'].notna()]
            self.item_features_list.append(brand_feature)
            print(f"  ✅ Brand: {len(brand_feature)} items")
        
        # Category
        if 'item_category' in self.train_df.columns:
            category_feature = self.train_df[['item_id', 'item_category']].drop_duplicates()
            category_feature.columns = ['id', 'value']
            category_feature['feature'] = 'category'
            category_feature = category_feature[category_feature['value'].notna()]
            self.item_features_list.append(category_feature)
            print(f"  ✅ Category: {len(category_feature)} items")
        
        # Subcategory
        if 'item_subcategory' in self.train_df.columns:
            subcategory_feature = self.train_df[['item_id', 'item_subcategory']].drop_duplicates()
            subcategory_feature.columns = ['id', 'value']
            subcategory_feature['feature'] = 'subcategory'
            subcategory_feature = subcategory_feature[subcategory_feature['value'].notna()]
            self.item_features_list.append(subcategory_feature)
            print(f"  ✅ Subcategory: {len(subcategory_feature)} items")
    
    def _add_price_features(self) -> None:
        """Добавляем price features"""
        if 'item_price' not in self.train_df.columns:
            print("  ⚠️  item_price не найден, пропускаем price features")
            return
        
        print("💰 Добавление price features...")
        
        price_data = self.train_df[['item_id', 'item_price', 'item_category']].drop_duplicates()
        price_data = price_data[price_data['item_price'].notna()]
        
        # 1. Price buckets (10 квантилей)
        try:
            price_data['price_bucket'] = pd.qcut(
                price_data['item_price'],
                q=10,
                labels=[f'price_q{i}' for i in range(1, 11)],
                duplicates='drop'
            )
            price_feature = price_data[['item_id', 'price_bucket']].copy()
            price_feature.columns = ['id', 'value']
            price_feature['feature'] = 'price_bucket'
            price_feature = price_feature[price_feature['value'].notna()]
            self.item_features_list.append(price_feature)
            print(f"  ✅ Price buckets: {len(price_feature)} items, {price_feature['value'].nunique()} categories")
        except Exception as e:
            print(f"  ⚠️  Ошибка создания price_bucket: {e}")
        
        # 2. Price tier (3 категории: low, mid, high)
        try:
            price_data['price_tier'] = pd.qcut(
                price_data['item_price'],
                q=3,
                labels=['low', 'mid', 'high'],
                duplicates='drop'
            )
            tier_feature = price_data[['item_id', 'price_tier']].copy()
            tier_feature.columns = ['id', 'value']
            tier_feature['feature'] = 'price_tier'
            tier_feature = tier_feature[tier_feature['value'].notna()]
            self.item_features_list.append(tier_feature)
            print(f"  ✅ Price tier: {len(tier_feature)} items")
        except Exception as e:
            print(f"  ⚠️  Ошибка создания price_tier: {e}")
        
        # 3. Price relative to category
        if 'item_category' in price_data.columns:
            try:
                cat_avg = price_data.groupby('item_category')['item_price'].transform('mean')
                price_ratio = price_data['item_price'] / cat_avg
                
                price_data['price_in_category'] = pd.cut(
                    price_ratio,
                    bins=[0, 0.7, 1.3, float('inf')],
                    labels=['below_avg', 'avg', 'above_avg']
                )
                
                rel_feature = price_data[['item_id', 'price_in_category']].copy()
                rel_feature.columns = ['id', 'value']
                rel_feature['feature'] = 'price_in_category'
                rel_feature = rel_feature[rel_feature['value'].notna()]
                self.item_features_list.append(rel_feature)
                print(f"  ✅ Price in category: {len(rel_feature)} items")
            except Exception as e:
                print(f"  ⚠️  Ошибка создания price_in_category: {e}")
    
    def _add_temporal_features(self) -> None:
        """Добавляем временные фичи"""
        if 'timestamp' not in self.train_df.columns:
            print("  ⚠️  timestamp не найден, пропускаем temporal features")
            return
        
        print("🕐 Добавление temporal features...")
        
        # Преобразуем timestamp в datetime
        self.train_df['datetime'] = pd.to_datetime(self.train_df['timestamp'], unit='s')
        
        # Day of week
        self.train_df['day_of_week'] = self.train_df['datetime'].dt.dayofweek
        dow_feature = self.train_df[['item_id', 'day_of_week']].drop_duplicates()
        dow_feature.columns = ['id', 'value']
        dow_feature['feature'] = 'day_of_week'
        dow_feature['value'] = dow_feature['value'].astype(str)
        self.item_features_list.append(dow_feature)
        print(f"  ✅ Day of week: {len(dow_feature)} items")
        
        # Hour of day buckets
        self.train_df['hour_bucket'] = pd.cut(
            self.train_df['datetime'].dt.hour,
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        hour_feature = self.train_df[['item_id', 'hour_bucket']].drop_duplicates()
        hour_feature.columns = ['id', 'value']
        hour_feature['feature'] = 'hour_bucket'
        hour_feature = hour_feature[hour_feature['value'].notna()]
        self.item_features_list.append(hour_feature)
        print(f"  ✅ Hour buckets: {len(hour_feature)} items")
    
    def _add_user_features(self) -> None:
        """Добавляем user features"""
        print("👤 Добавление user features...")
        
        # 1. Interaction count (количество взаимодействий пользователя)
        user_interaction_count = self.train_df.groupby('user_id').size().reset_index(name='interaction_count')
        user_interaction_count['interaction_bucket'] = pd.qcut(
            user_interaction_count['interaction_count'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )
        interaction_feature = user_interaction_count[['user_id', 'interaction_bucket']].copy()
        interaction_feature.columns = ['id', 'value']
        interaction_feature['feature'] = 'interaction_level'
        self.user_features_list.append(interaction_feature)
        print(f"  ✅ Interaction level: {len(interaction_feature)} users")
        
        # 2. Average price (средняя цена покупок пользователя)
        if 'item_price' in self.train_df.columns:
            user_avg_price = self.train_df.groupby('user_id')['item_price'].mean().reset_index(name='avg_price')
            user_avg_price = user_avg_price[user_avg_price['avg_price'].notna()]
            
            try:
                user_avg_price['price_segment'] = pd.qcut(
                    user_avg_price['avg_price'],
                    q=3,
                    labels=['budget', 'standard', 'premium'],
                    duplicates='drop'
                )
                price_seg_feature = user_avg_price[['user_id', 'price_segment']].copy()
                price_seg_feature.columns = ['id', 'value']
                price_seg_feature['feature'] = 'user_price_segment'
                price_seg_feature = price_seg_feature[price_seg_feature['value'].notna()]
                self.user_features_list.append(price_seg_feature)
                print(f"  ✅ User price segment: {len(price_seg_feature)} users")
            except Exception as e:
                print(f"  ⚠️  Ошибка создания user_price_segment: {e}")
        
        # 3. Favorite brand (самый популярный бренд пользователя)
        if 'item_brand_id' in self.train_df.columns:
            user_brand = self.train_df[self.train_df['item_brand_id'].notna()].copy()
            user_fav_brand = user_brand.groupby(['user_id', 'item_brand_id']).size().reset_index(name='count')
            user_fav_brand = user_fav_brand.sort_values(['user_id', 'count'], ascending=[True, False])
            user_fav_brand = user_fav_brand.groupby('user_id').first().reset_index()
            
            brand_feature = user_fav_brand[['user_id', 'item_brand_id']].copy()
            brand_feature.columns = ['id', 'value']
            brand_feature['feature'] = 'favorite_brand'
            brand_feature['value'] = 'brand_' + brand_feature['value'].astype(str)
            self.user_features_list.append(brand_feature)
            print(f"  ✅ Favorite brand: {len(brand_feature)} users")
        
        # 4. Favorite category (самая популярная категория пользователя)
        if 'item_category' in self.train_df.columns:
            user_category = self.train_df[self.train_df['item_category'].notna()].copy()
            user_fav_category = user_category.groupby(['user_id', 'item_category']).size().reset_index(name='count')
            user_fav_category = user_fav_category.sort_values(['user_id', 'count'], ascending=[True, False])
            user_fav_category = user_fav_category.groupby('user_id').first().reset_index()
            
            category_feature = user_fav_category[['user_id', 'item_category']].copy()
            category_feature.columns = ['id', 'value']
            category_feature['feature'] = 'favorite_category'
            category_feature['value'] = 'category_' + category_feature['value'].astype(str)
            self.user_features_list.append(category_feature)
            print(f"  ✅ Favorite category: {len(category_feature)} users")
        
        # 5. User diversity (сколько уникальных категорий покупал)
        if 'item_category' in self.train_df.columns:
            user_diversity = self.train_df.groupby('user_id')['item_category'].nunique().reset_index(name='category_diversity')
            user_diversity['diversity_level'] = pd.cut(
                user_diversity['category_diversity'],
                bins=[0, 2, 5, 10, float('inf')],
                labels=['focused', 'moderate', 'diverse', 'very_diverse']
            )
            
            diversity_feature = user_diversity[['user_id', 'diversity_level']].copy()
            diversity_feature.columns = ['id', 'value']
            diversity_feature['feature'] = 'user_diversity'
            diversity_feature = diversity_feature[diversity_feature['value'].notna()]
            self.user_features_list.append(diversity_feature)
            print(f"  ✅ User diversity: {len(diversity_feature)} users")
        
        # 6. Activity recency (как давно был последний интеракшн)
        if 'timestamp' in self.train_df.columns:
            user_last_interaction = self.train_df.groupby('user_id')['timestamp'].max().reset_index(name='last_timestamp')
            max_timestamp = self.train_df['timestamp'].max()
            user_last_interaction['days_since_last'] = (max_timestamp - user_last_interaction['last_timestamp']) / (24 * 3600)
            
            user_last_interaction['recency'] = pd.cut(
                user_last_interaction['days_since_last'],
                bins=[-1, 7, 30, 90, float('inf')],
                labels=['recent', 'active', 'occasional', 'dormant']
            )
            
            recency_feature = user_last_interaction[['user_id', 'recency']].copy()
            recency_feature.columns = ['id', 'value']
            recency_feature['feature'] = 'user_recency'
            recency_feature = recency_feature[recency_feature['value'].notna()]
            self.user_features_list.append(recency_feature)
            print(f"  ✅ User recency: {len(recency_feature)} users")
    
    def _prepare_embeddings(self, dataset: Dataset, n_factors: int) -> Optional[PretrainedEmbeddingsItemNet]:
        """Подготовка предобученных эмбеддингов"""
        if 'item_embedding' not in self.train_df.columns:
            print("  ⚠️  item_embedding не найден, пропускаем embeddings")
            return None
        
        print("🎨 Подготовка предобученных эмбеддингов...")
        
        # Группируем по item_id
        item_emb_df = self.train_df.groupby('item_id').agg({
            'item_embedding': 'first'
        }).reset_index()
        
        item_emb_df = item_emb_df[item_emb_df['item_embedding'].notna()]
        
        # Проверяем размерность
        sample_emb = item_emb_df['item_embedding'].iloc[0]
        if isinstance(sample_emb, list):
            emb_dim = len(sample_emb)
        elif isinstance(sample_emb, np.ndarray):
            emb_dim = sample_emb.shape[0]
        else:
            emb_dim = len(sample_emb)
        
        num_items = dataset.item_id_map.size
        
        print(f"  📊 Размерность эмбеддингов: {emb_dim}")
        print(f"  📊 Количество товаров: {num_items}")
        
        # Инициализируем матрицу
        self.embeddings_matrix = np.random.randn(num_items, emb_dim).astype(np.float32) * 0.01
        
        # Заполняем предобученными эмбеддингами
        items_found = 0
        for _, row in item_emb_df.iterrows():
            item_id = row['item_id']
            if item_id in dataset.item_id_map.external_ids:
                internal_id = dataset.item_id_map.to_internal[item_id]
                emb = row['item_embedding']
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                elif not isinstance(emb, np.ndarray):
                    emb = np.array(emb, dtype=np.float32)
                self.embeddings_matrix[internal_id] = emb
                items_found += 1
        
        print(f"  ✅ Заполнено эмбеддингов: {items_found}/{num_items} ({items_found/num_items*100:.1f}%)")
        
        # Создаем PretrainedEmbeddingsItemNet
        self.pretrained_net = PretrainedEmbeddingsItemNet(
            embeddings_matrix=self.embeddings_matrix,
            output_dim=n_factors
        )
        
        return self.pretrained_net
    
    def build_dataset(
        self,
        use_price_features: bool = True,
        use_temporal_features: bool = False,
        use_item_embeddings: bool = True,
        use_user_features: bool = False,
        n_factors: int = 256
    ) -> Tuple[Dataset, Dict[str, Any]]:
        """
        Построить датасет для BERT4Rec со всеми фичами
        
        Args:
            use_price_features: использовать price features
            use_temporal_features: использовать temporal features
            use_item_embeddings: использовать предобученные эмбеддинги
            use_user_features: использовать user features
            n_factors: размерность латентных факторов (для проекции эмбеддингов)
        
        Returns:
            dataset: RecTools Dataset
            config: словарь с конфигурацией для BERT4RecModel
                    содержит 'item_net_block_types', 'cat_item_features', 'cat_user_features'
        """
        print("\n" + "="*70)
        print("🏗️  BERT4Rec Dataset Builder")
        print("="*70)
        
        # 1. Подготовка interactions
        interactions = self._prepare_interactions()
        print(f"✅ Interactions: {len(interactions)} строк")
        
        # 2. Добавляем базовые фичи
        self._add_basic_item_features()
        
        # 3. Добавляем price features
        if use_price_features:
            self._add_price_features()
        
        # 4. Добавляем temporal features
        if use_temporal_features:
            self._add_temporal_features()
        
        # 5. Добавляем user features
        if use_user_features:
            self._add_user_features()
        
        # 6. Объединяем все item features
        if self.item_features_list:
            item_features = pd.concat(self.item_features_list, ignore_index=True)
            print(f"\n📦 Итого item features: {item_features.shape[0]} строк")
            print(f"   Фичи: {list(item_features['feature'].unique())}")
            print(f"   Уникальных товаров: {item_features['id'].nunique()}")
            
            # Список категориальных фичей
            cat_item_features = list(item_features['feature'].unique())
        else:
            item_features = None
            cat_item_features = []
        
        # 7. Объединяем все user features
        if self.user_features_list:
            user_features = pd.concat(self.user_features_list, ignore_index=True)
            print(f"\n👤 Итого user features: {user_features.shape[0]} строк")
            print(f"   Фичи: {list(user_features['feature'].unique())}")
            print(f"   Уникальных пользователей: {user_features['id'].nunique()}")
            
            # Список категориальных user фичей
            cat_user_features = list(user_features['feature'].unique())
        else:
            user_features = None
            cat_user_features = []
        
        # 8. Создаем базовый датасет
        print("\n🔨 Создание RecTools Dataset...")
        dataset_kwargs = {'interactions_df': interactions}
        
        if item_features is not None:
            dataset_kwargs['item_features_df'] = item_features
            dataset_kwargs['cat_item_features'] = cat_item_features
        
        if user_features is not None:
            dataset_kwargs['user_features_df'] = user_features
            dataset_kwargs['cat_user_features'] = cat_user_features
        
        dataset = Dataset.construct(**dataset_kwargs)
        
        print(f"✅ Dataset: {dataset.user_id_map.size} users, {dataset.item_id_map.size} items")
        
        # 9. Подготовка эмбеддингов (если нужно)
        use_pretrained_emb = False
        if use_item_embeddings:
            pretrained_net = self._prepare_embeddings(dataset, n_factors)
            if pretrained_net is not None and self.embeddings_matrix is not None:
                # Сохраняем эмбеддинги в глобальное хранилище класса
                PretrainedEmbeddingsItemNet.set_embeddings(self.embeddings_matrix)
                use_pretrained_emb = True
        
        # 10. Формируем конфигурацию для модели
        if use_pretrained_emb:
            # ID + Categorical + Pretrained Embeddings
            item_net_block_types = (
                IdEmbeddingsItemNet,
                CatFeaturesItemNet,
                PretrainedEmbeddingsItemNet  # Передаем класс, а не экземпляр
            )
            print(f"\n✅ ItemNet: ID + Categorical + Pretrained Embeddings")
        elif item_features is not None:
            # ID + Categorical
            item_net_block_types = (IdEmbeddingsItemNet, CatFeaturesItemNet)
            print(f"\n✅ ItemNet: ID + Categorical")
        else:
            # Только ID
            item_net_block_types = (IdEmbeddingsItemNet,)
            print(f"\n✅ ItemNet: ID only")
        
        config = {
            'item_net_block_types': item_net_block_types,
            'cat_item_features': cat_item_features,
            'cat_user_features': cat_user_features,
            'embeddings_matrix': self.embeddings_matrix,
            'use_pretrained_emb': use_pretrained_emb
        }
        
        print("="*70)
        print("✅ Dataset готов к использованию!")
        print("="*70 + "\n")
        
        return dataset, config


def create_bert4rec_dataset(
    train_df: pd.DataFrame,
    use_price_features: bool = True,
    use_temporal_features: bool = False,
    use_item_embeddings: bool = True,
    use_user_features: bool = False,
    n_factors: int = 256
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Удобная функция для создания BERT4Rec датасета
    
    Args:
        train_df: DataFrame с данными
        use_price_features: использовать price features
        use_temporal_features: использовать temporal features
        use_item_embeddings: использовать предобученные эмбеддинги
        use_user_features: использовать user features
        n_factors: размерность латентных факторов
    
    Returns:
        dataset, config
    
    Example:
        >>> dataset, config = create_bert4rec_dataset(
        ...     train_df, 
        ...     use_user_features=True,
        ...     n_factors=256
        ... )
        >>> model = BERT4RecModel(
        ...     item_net_block_types=config['item_net_block_types'],
        ...     n_factors=256,
        ...     ...
        ... )
    """
    builder = BERT4RecDatasetBuilder(train_df)
    return builder.build_dataset(
        use_price_features=use_price_features,
        use_temporal_features=use_temporal_features,
        use_item_embeddings=use_item_embeddings,
        use_user_features=use_user_features,
        n_factors=n_factors
    )
