import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple, List, Dict
import numpy as np
from tqdm import tqdm


class BasketDataset(Dataset):
    """
    Dataset for basket-based sequential recommendation.
    Groups items by user and 2-hour time windows to form baskets.
    """
    def __init__(
        self,
        df,
        max_basket_size: int = 50,
        max_history_baskets: int = 10,
        num_negatives: int = 5,
        num_items: int = None,
        basket_time_window: int = 7200  # 2 hours in seconds
    ):
        self.df = df.copy()
        self.max_basket_size = max_basket_size
        self.max_history_baskets = max_history_baskets
        self.num_negatives = num_negatives
        self.num_items = num_items
        self.basket_time_window = basket_time_window
        
        self.df = self.df.sort_values(['user_id', 'timestamp'])
        
        self.user_baskets = self._create_baskets()
        self.samples = self._create_training_samples()
        
        self.item_freq = self.df['item_id'].value_counts().to_dict()
        self.all_items = set(range(num_items)) if num_items else set(self.df['item_id'].unique())
    
    def _create_baskets(self) -> Dict[int, List[List[int]]]:
        """
        Group items by user and 2-hour time windows to create baskets.
        Each basket contains items added within a 2-hour window.
        """
        user_baskets = {}
        
        for user_id, user_df in self.df.groupby('user_id'):
            user_df = user_df.sort_values('timestamp')
            baskets = []
            current_basket = []
            basket_start_time = None
            
            for _, row in user_df.iterrows():
                timestamp = row['timestamp']
                item_id = row['item_id']
                
                if basket_start_time is None:
                    basket_start_time = timestamp
                    current_basket = [item_id]
                elif timestamp - basket_start_time <= self.basket_time_window:
                    current_basket.append(item_id)
                else:
                    if len(current_basket) > 0:
                        baskets.append(current_basket)
                    basket_start_time = timestamp
                    current_basket = [item_id]
            
            if len(current_basket) > 0:
                baskets.append(current_basket)
            
            user_baskets[user_id] = baskets
        
        return user_baskets
    
    def _create_training_samples(self) -> List[Tuple]:
        """
        Create training samples with autoregressive structure.
        For each basket, predict each next item given previous items.
        """
        samples = []
        
        for user_id, baskets in self.user_baskets.items():
            if len(baskets) == 0:
                continue
            
            for basket_idx, basket in enumerate(baskets):
                history_baskets = baskets[max(0, basket_idx - self.max_history_baskets):basket_idx]
                
                for pos in range(len(basket)):
                    current_basket_prefix = basket[:pos]
                    target_item = basket[pos]
                    
                    samples.append({
                        'user_id': user_id,
                        'history_baskets': history_baskets,
                        'current_basket_prefix': current_basket_prefix,
                        'target_item': target_item
                    })
        
        return samples
    
    def _sample_negatives(self, positive_items: set) -> List[int]:
        """Sample negative items using popularity-based sampling."""
        negatives = []
        available_items = list(self.all_items - positive_items)
        
        if len(available_items) < self.num_negatives:
            negatives = available_items + [0] * (self.num_negatives - len(available_items))
        else:
            negatives = np.random.choice(available_items, size=self.num_negatives, replace=False).tolist()
        
        return negatives
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        history_flat = []
        for basket in sample['history_baskets']:
            history_flat.extend(basket[:self.max_basket_size])
        
        current = sample['current_basket_prefix'][:self.max_basket_size]
        
        sequence = history_flat + current
        
        positive_items = set(history_flat + current + [sample['target_item']])
        negatives = self._sample_negatives(positive_items)
        
        return {
            'sequence': sequence,
            'target': sample['target_item'],
            'negatives': negatives,
            'user_id': sample['user_id']
        }


def collate_fn(batch):
    """Collate function for batching sequences of variable length."""
    max_seq_len = max(len(item['sequence']) for item in batch)
    max_seq_len = max(max_seq_len, 1)
    
    sequences = []
    masks = []
    targets = []
    negatives = []
    user_ids = []
    
    for item in batch:
        seq = item['sequence']
        seq_len = len(seq)
        
        padded_seq = seq + [0] * (max_seq_len - seq_len)
        mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
        
        sequences.append(padded_seq)
        masks.append(mask)
        targets.append(item['target'])
        negatives.append(item['negatives'])
        user_ids.append(item['user_id'])
    
    return {
        'sequences': torch.LongTensor(sequences),
        'masks': torch.BoolTensor(masks),
        'targets': torch.LongTensor(targets),
        'negatives': torch.LongTensor(negatives),
        'user_ids': torch.LongTensor(user_ids)
    }


class TransformerBasketEncoder(nn.Module):
    """
    Transformer-based encoder for basket sequences.
    """
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_length: int = 200
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_length, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: (batch_size, seq_len) - item indices
            masks: (batch_size, seq_len) - 1 for valid positions, 0 for padding
        
        Returns:
            basket_embedding: (batch_size, embedding_dim) - basket representation
        """
        batch_size, seq_len = sequences.shape
        
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
        
        item_emb = self.item_embeddings(sequences)
        pos_emb = self.position_embeddings(positions)
        
        x = self.dropout(item_emb + pos_emb)
        x = self.layer_norm(x)
        
        padding_mask = ~masks
        
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        masked_out = transformer_out * masks.unsqueeze(-1).float()
        sum_out = masked_out.sum(dim=1)
        count = masks.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        basket_embedding = sum_out / count
        
        return basket_embedding


class EmbeddingRetrievalModel(pl.LightningModule):
    """
    Embedding-based Retrieval model for next-item prediction.
    Uses Transformer encoder to get basket embeddings and predicts next item
    via similarity with item embeddings.
    """
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        temperature: float = 0.07,
        weight_decay: float = 1e-4,
        max_seq_length: int = 200
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        self.encoder = TransformerBasketEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        self.item_embeddings = self.encoder.item_embeddings
    
    def forward(self, sequences: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Get basket embedding."""
        return self.encoder(sequences, masks)
    
    def compute_loss(
        self,
        basket_emb: torch.Tensor,
        target_items: torch.Tensor,
        negative_items: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE loss with negative sampling.
        
        Args:
            basket_emb: (batch_size, embedding_dim)
            target_items: (batch_size,)
            negative_items: (batch_size, num_negatives)
        """
        target_emb = self.item_embeddings(target_items)
        negative_emb = self.item_embeddings(negative_items)
        
        pos_logits = (basket_emb * target_emb).sum(dim=-1) / self.temperature
        
        neg_logits = torch.bmm(
            negative_emb,
            basket_emb.unsqueeze(-1)
        ).squeeze(-1) / self.temperature
        
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        sequences = batch['sequences']
        masks = batch['masks']
        targets = batch['targets']
        negatives = batch['negatives']
        
        basket_emb = self(sequences, masks)
        loss = self.compute_loss(basket_emb, targets, negatives)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences = batch['sequences']
        masks = batch['masks']
        targets = batch['targets']
        negatives = batch['negatives']
        
        basket_emb = self(sequences, masks)
        loss = self.compute_loss(basket_emb, targets, negatives)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    @torch.no_grad()
    def predict_top_k(
        self,
        sequences: torch.Tensor,
        masks: torch.Tensor,
        k: int = 100,
        exclude_items: Optional[List[set]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-K items for given basket sequences.
        
        Args:
            sequences: (batch_size, seq_len)
            masks: (batch_size, seq_len)
            k: number of items to recommend
            exclude_items: list of sets with items to exclude for each batch element
        
        Returns:
            top_k_items: (batch_size, k) - item indices
            top_k_scores: (batch_size, k) - similarity scores
        """
        self.eval()
        
        basket_emb = self(sequences, masks)
        basket_emb = F.normalize(basket_emb, p=2, dim=-1)
        
        all_item_emb = self.item_embeddings.weight[1:]
        all_item_emb = F.normalize(all_item_emb, p=2, dim=-1)
        
        scores = torch.matmul(basket_emb, all_item_emb.t())
        
        if exclude_items is not None:
            for i, items_to_exclude in enumerate(exclude_items):
                if items_to_exclude:
                    exclude_indices = list(items_to_exclude)
                    exclude_indices = [idx for idx in exclude_indices if idx > 0]
                    if exclude_indices:
                        scores[i, exclude_indices] = -float('inf')
        
        top_k_scores, top_k_indices = torch.topk(scores, k=min(k, scores.size(1)), dim=-1)
        
        top_k_items = top_k_indices + 1
        
        return top_k_items, top_k_scores


class EmbeddingRetrievalRecommender:
    """
    Wrapper class for training and inference with EmbeddingRetrievalModel.
    """
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        temperature: float = 0.07,
        weight_decay: float = 1e-4,
        max_seq_length: int = 200,
        max_basket_size: int = 50,
        max_history_baskets: int = 10,
        num_negatives: int = 5,
        basket_time_window: int = 7200  # 2 hours in seconds
    ):
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length
        self.max_basket_size = max_basket_size
        self.max_history_baskets = max_history_baskets
        self.num_negatives = num_negatives
        self.basket_time_window = basket_time_window
        
        self.model = None
        self.trainer = None
    
    def fit(
        self,
        train_df,
        val_df=None,
        batch_size: int = 256,
        num_epochs: int = 10,
        num_workers: int = 4,
        accelerator: str = 'auto',
        devices: int = 1
    ):
        """Train the model."""
        train_dataset = BasketDataset(
            train_df,
            max_basket_size=self.max_basket_size,
            max_history_baskets=self.max_history_baskets,
            num_negatives=self.num_negatives,
            num_items=self.num_items,
            basket_time_window=self.basket_time_window
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = None
        if val_df is not None:
            val_dataset = BasketDataset(
                val_df,
                max_basket_size=self.max_basket_size,
                max_history_baskets=self.max_history_baskets,
                num_negatives=self.num_negatives,
                num_items=self.num_items,
                basket_time_window=self.basket_time_window
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                persistent_workers=True if num_workers > 0 else False
            )
        
        self.model = EmbeddingRetrievalModel(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            weight_decay=self.weight_decay,
            max_seq_length=self.max_seq_length
        )
        
        self.trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=accelerator,
            devices=devices,
            log_every_n_steps=50,
            enable_checkpointing=True,
            enable_progress_bar=True,
            gradient_clip_val=1.0
        )
        
        self.trainer.fit(self.model, train_loader, val_loader)
    
    def recommend(
        self,
        df,
        k: int = 100,
        batch_size: int = 256,
        exclude_seen: bool = True
    ) -> Dict[int, List[int]]:
        """
        Generate recommendations for users in the dataframe.
        
        Args:
            df: DataFrame with user interactions
            k: number of items to recommend
            batch_size: batch size for inference
            exclude_seen: whether to exclude already seen items
        
        Returns:
            Dictionary mapping user_id to list of recommended item_ids
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        df_sorted = df.sort_values(['user_id', 'timestamp'])
        user_histories = {}
        
        for user_id, user_df in df_sorted.groupby('user_id'):
            user_df = user_df.sort_values('timestamp')
            baskets = []
            seen_items = set()
            current_basket = []
            basket_start_time = None
            
            for _, row in user_df.iterrows():
                timestamp = row['timestamp']
                item_id = row['item_id']
                
                if basket_start_time is None:
                    basket_start_time = timestamp
                    current_basket = [item_id]
                elif timestamp - basket_start_time <= self.basket_time_window:
                    current_basket.append(item_id)
                else:
                    if len(current_basket) > 0:
                        baskets.append(current_basket)
                    basket_start_time = timestamp
                    current_basket = [item_id]
                
                seen_items.add(item_id)
            
            if len(current_basket) > 0:
                baskets.append(current_basket)
            
            user_histories[user_id] = {
                'baskets': baskets,
                'seen_items': seen_items
            }
        
        recommendations = {}
        user_ids = list(user_histories.keys())
        
        for i in tqdm(range(0, len(user_ids), batch_size), desc="Generating recommendations"):
            batch_users = user_ids[i:i + batch_size]
            
            sequences = []
            masks = []
            exclude_items_batch = []
            
            for user_id in batch_users:
                history = user_histories[user_id]
                baskets = history['baskets'][-self.max_history_baskets:]
                
                sequence = []
                for basket in baskets:
                    sequence.extend(basket[:self.max_basket_size])
                
                # Обрезаем последовательность до max_seq_length
                if len(sequence) > self.max_seq_length:
                    sequence = sequence[-self.max_seq_length:]
                
                if len(sequence) == 0:
                    sequence = [0]
                
                sequences.append(sequence)
                
                if exclude_seen:
                    exclude_items_batch.append(history['seen_items'])
                else:
                    exclude_items_batch.append(set())
            
            # Ограничиваем max_len значением max_seq_length
            max_len = min(max(len(seq) for seq in sequences), self.max_seq_length)
            padded_sequences = []
            padded_masks = []
            
            for seq in sequences:
                seq_len = len(seq)
                padded_seq = seq + [0] * (max_len - seq_len)
                mask = [1] * seq_len + [0] * (max_len - seq_len)
                padded_sequences.append(padded_seq)
                padded_masks.append(mask)
            
            sequences_tensor = torch.LongTensor(padded_sequences).to(device)
            masks_tensor = torch.BoolTensor(padded_masks).to(device)
            
            top_k_items, _ = self.model.predict_top_k(
                sequences_tensor,
                masks_tensor,
                k=k,
                exclude_items=exclude_items_batch
            )
            
            for j, user_id in enumerate(batch_users):
                recommendations[user_id] = top_k_items[j].cpu().tolist()
        
        return recommendations
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        self.trainer.save_checkpoint(path)
    
    def load_model(self, path: str):
        """Load model from checkpoint."""
        self.model = EmbeddingRetrievalModel.load_from_checkpoint(path)
