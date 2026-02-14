import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional, Dict

from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    catalog_coverage,
    novelty,
    diversity,
    serendipity
)


def _extract_items(predicted: Union[List, List[Tuple]]) -> List:
    if not predicted or len(predicted) == 0:
        return []

    if isinstance(predicted[0], tuple) and len(predicted[0]) == 2:
        return [item for item, score in predicted]
    else:
        return predicted


def _extract_scores(predicted: Union[List, List[Tuple]]) -> List[float]:
    if not predicted or len(predicted) == 0:
        return []

    if isinstance(predicted[0], tuple) and len(predicted[0]) == 2:
        return [score for item, score in predicted]
    else:
        return [1.0 / (i + 1) for i in range(len(predicted))]


def calculate_inter_user_diversity(df: pd.DataFrame, model_preds: str) -> float:
    """
    Calculate inter-user diversity (насколько разные рекомендации между всеми пользователями).
    
    Метрика измеряет персонализацию:
    - Значение ~0: все пользователи получают одинаковые рекомендации (как TopPopular)
    - Значение ~1: каждый пользователь получает уникальные рекомендации
    """
    item_user_count = {}
    total_users = len(df)
    
    for idx, row in df.iterrows():
        predictions = row[model_preds] if row[model_preds] else []
        items = _extract_items(predictions)
        
        for item in items:
            if item not in item_user_count:
                item_user_count[item] = 0
            item_user_count[item] += 1
    
    if not item_user_count or total_users == 0:
        return 0.0

    avg_user_fraction = np.mean([count / total_users for count in item_user_count.values()])
    diversity = 1.0 - avg_user_fraction
    
    return diversity


def prepare_for_evaluation(df: pd.DataFrame, 
                           model_preds: str, 
                           gt_col: str,
                           user_col: str = 'user_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
    true_rows = []
    for idx, row in df.iterrows():
        user_id = row.get(user_col, idx)
        gt_items = [item[0] for item in row[gt_col]] if row[gt_col] else []
        
        for item_id in gt_items:
            true_rows.append({
                'user_id': user_id, 
                'item_id': item_id, 
                'rating': 1.0
            })
    
    ratings_true = pd.DataFrame(true_rows)

    pred_rows = []
    for idx, row in df.iterrows():
        user_id = row.get(user_col, idx)
        predictions = row[model_preds] if row[model_preds] else []
        
        if not predictions:
            continue
        items = _extract_items(predictions)
        scores = _extract_scores(predictions)
        
        for item_id, score in zip(items, scores):
            pred_rows.append({
                'user_id': user_id,
                'item_id': item_id,
                'prediction': score
            })
    
    ratings_pred = pd.DataFrame(pred_rows)
    
    return ratings_true, ratings_pred


def calculate_mrr(df: pd.DataFrame, 
                  model_preds: str, 
                  gt_col: str) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []
    
    for idx, row in df.iterrows():
        gt_items = set([item[0] for item in row[gt_col]]) if row[gt_col] else set()
        predictions = row[model_preds] if row[model_preds] else []
        
        if not predictions or not gt_items:
            reciprocal_ranks.append(0.0)
            continue

        pred_items = _extract_items(predictions)

        for rank, item in enumerate(pred_items, start=1):
            if item in gt_items:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def _calculate_metrics_at_k(df: pd.DataFrame,
                          model_preds: str,
                          gt_col: str,
                          train_col: Optional[str] = None,
                          k_values: List[int] = [10, 100],
                          verbose: bool = False) -> Dict[str, float]:
    metrics = {}
    ratings_true, ratings_pred = prepare_for_evaluation(df, model_preds, gt_col)
    
    if ratings_true.empty or ratings_pred.empty:
        print("Warning: Empty ratings data, returning zero metrics")
        return {f"{metric}@{k}": 0.0 
                for metric in ['MAP', 'NDCG', 'Precision', 'Recall'] 
                for k in k_values}
    
    for k in k_values:
        # MAP@k
        metrics[f'MAP@{k}'] = map_at_k(
            ratings_true, ratings_pred,
            col_user='user_id', col_item='item_id',
            col_rating='rating', col_prediction='prediction',
            k=k
        )
        
        # NDCG@k
        metrics[f'NDCG@{k}'] = ndcg_at_k(
            ratings_true, ratings_pred,
            col_user='user_id', col_item='item_id',
            col_rating='rating', col_prediction='prediction',
            k=k
        )
        
        # Precision@k
        metrics[f'Precision@{k}'] = precision_at_k(
            ratings_true, ratings_pred,
            col_user='user_id', col_item='item_id',
            col_rating='rating', col_prediction='prediction',
            k=k
        )
        
        # Recall@k
        metrics[f'Recall@{k}'] = recall_at_k(
            ratings_true, ratings_pred,
            col_user='user_id', col_item='item_id',
            col_rating='rating', col_prediction='prediction',
            k=k
        )
    
    # MRR (Mean Reciprocal Rank)
    try:
        metrics['MRR'] = calculate_mrr(df, model_preds, gt_col)
    except Exception as e:
        print(f"Warning: Could not calculate MRR: {e}")
        metrics['MRR'] = 0.0

    train_true = None
    if train_col is not None:
        try:
            # Prepare train data from train_col (not using model_preds)
            train_rows = []
            for idx, row in df.iterrows():
                user_id = row.get('user_id', idx)
                train_items = [item[0] for item in row[train_col]] if row[train_col] else []
                
                for item_id in train_items:
                    train_rows.append({
                        'user_id': user_id, 
                        'item_id': item_id, 
                        'rating': 1.0
                    })
            
            train_true = pd.DataFrame(train_rows)
            
            # Filter out train items from recommendations for novelty/serendipity
            if not train_true.empty:
                train_pairs = set(zip(train_true['user_id'], train_true['item_id']))
                ratings_pred_filtered = ratings_pred[
                    ~ratings_pred.apply(lambda x: (x['user_id'], x['item_id']) in train_pairs, axis=1)
                ].copy()
            else:
                ratings_pred_filtered = ratings_pred.copy()
        except Exception as e:
            print(f"Warning: Could not prepare train data: {e}")
            ratings_pred_filtered = ratings_pred.copy()
    
    
    # Catalog Coverage
    try:
        if train_true is not None and not train_true.empty:
            catalog_items = set(train_true['item_id'].unique())
            recommended_items = set(ratings_pred['item_id'].unique())
            metrics['Catalog_Coverage'] = len(recommended_items) / len(catalog_items) if len(catalog_items) > 0 else 0.0
        else:
            metrics['Catalog_Coverage'] = 0.0
    except Exception as e:
        print(f"Warning: Could not calculate Catalog Coverage: {e}")
        metrics['Catalog_Coverage'] = 0.0
    
    
    # Inter-User Diversity
    try:
        metrics['Diversity'] = calculate_inter_user_diversity(df, model_preds)
    except Exception as e:
        print(f"Warning: Could not calculate Diversity: {e}")
        metrics['Diversity'] = 0.0
    
    
    # Novelty and Serendipity
    if train_true is not None and not train_true.empty:
        try:
            novelty_raw = novelty(
                train_df=train_true,
                reco_df=ratings_pred_filtered,
                col_user='user_id',
                col_item='item_id'
            )
            
            # Normalize novelty to [0, 1] range
            n_items = len(train_true['item_id'].unique())
            max_novelty = np.log2(n_items) if n_items > 1 else 1.0
            metrics['Novelty'] = min(1.0, novelty_raw / max_novelty) if max_novelty > 0 else 0.0
            
        except Exception as e:
            print(f"Warning: Could not calculate Novelty: {e}")
            metrics['Novelty'] = 0.0
        
        # Serendipity
        try:
            metrics['Serendipity'] = serendipity(
                train_df=train_true,
                reco_df=ratings_pred_filtered,
                col_user='user_id',
                col_item='item_id',
                col_relevance='prediction'
            )
        except Exception as e:
            print(f"Warning: Could not calculate Serendipity: {e}")
            metrics['Serendipity'] = 0.0
    
    return metrics


def calculate_metrics(df: pd.DataFrame,
                     model_preds: str,
                     gt_col: str,
                     train_col: Optional[str] = None,
                     verbose: bool = False) -> Dict[str, float]:
    all_metrics = _calculate_metrics_at_k(
        df, model_preds, gt_col, 
        train_col=train_col,
        k_values=[10, 100], 
        verbose=False
    )
    
    if verbose:
        for k in [10, 100]:
            print(f"\nAt k={k}:")
            print(f"  MAP@{k}       = {all_metrics.get(f'MAP@{k}', 0.0):.4f}")
            print(f"  NDCG@{k}      = {all_metrics.get(f'NDCG@{k}', 0.0):.4f}")
            print(f"  Precision@{k} = {all_metrics.get(f'Precision@{k}', 0.0):.4f}")
            print(f"  Recall@{k}    = {all_metrics.get(f'Recall@{k}', 0.0):.4f}")
        
        print(f"\nOther Metrics:")
        print(f"  MRR                 = {all_metrics.get('MRR', 0.0):.4f}")
        print(f"  Catalog Coverage    = {all_metrics.get('Catalog_Coverage', 0.0):.4f}")
        print(f"  Diversity     = {all_metrics.get('Diversity', 0.0):.4f}  [0=same recs for all, 1=unique recs]")
        
        if 'Novelty' in all_metrics:
            print(f"  Novelty             = {all_metrics.get('Novelty', 0.0):.4f}")
        if 'Serendipity' in all_metrics:
            print(f"  Serendipity         = {all_metrics.get('Serendipity', 0.0):.4f}")
    
    return all_metrics
