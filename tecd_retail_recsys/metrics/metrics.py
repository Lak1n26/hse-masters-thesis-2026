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


def _safe_nonempty(x) -> bool:
    """Check if value is non-empty without using truth value (works for list, np.array)."""
    if x is None:
        return False
    try:
        return len(x) > 0
    except (TypeError, AttributeError):
        return False


def _get_el_at(x, index: int) -> Optional[int]:
    """Get element at index from interaction (tuple/list/dict/array); cast to int."""
    if x is None:
        return None
    try:
        if isinstance(x, (list, tuple)) and len(x) > index:
            v = x[index]
            return int(v) if v is not None else None
        if hasattr(x, "get") and callable(x.get):
            v = x.get("item_id") if index == 0 else (x.get(index) or x.get("f%d" % index))
            if v is not None:
                return int(v)
        if hasattr(x, "shape") and getattr(x, "size", 0) > index:
            v = x[index]
            return int(v) if v is not None else None
        return None
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _first_el(x) -> Optional[int]:
    """Get item_id from interaction: tuple/list (item_id, ts, ...), dict-like, or numpy row."""
    return _get_el_at(x, 0)


def _detect_item_id_index(df: pd.DataFrame, gt_col: str, model_preds: str, user_col: str = "user_id") -> int:
    """Detect which index in interaction tuple is item_id (0 or 1) by checking overlap with predictions."""
    for _, row in df.head(20).iterrows():
        raw_gt = row.get(gt_col)
        if not _safe_nonempty(raw_gt):
            continue
        pred_list = _extract_items(row.get(model_preds) or [])
        pred_set = set()
        for x in pred_list:
            try:
                pred_set.add(int(x))
            except (TypeError, ValueError):
                pass
        if not pred_set:
            continue
        for idx in (0, 1):
            gt_set = set()
            for x in raw_gt:
                v = _get_el_at(x, idx)
                if v is not None:
                    gt_set.add(v)
            if gt_set and len(gt_set & pred_set) > 0:
                return idx
    return 0


def _extract_items(predicted: Union[List, List[Tuple]]) -> List:
    if not _safe_nonempty(predicted):
        return []

    if isinstance(predicted[0], tuple) and len(predicted[0]) == 2:
        return [item for item, score in predicted]
    else:
        return predicted


def _extract_scores(predicted: Union[List, List[Tuple]]) -> List[float]:
    if not _safe_nonempty(predicted):
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
        predictions = row.get(model_preds) if _safe_nonempty(row.get(model_preds)) else []
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


def _resolve_gt_col(df: pd.DataFrame, gt_col: str) -> str:
    """Use gt_col if it exists and has data; else try val_interactions_y, val_interactions_x (pandas merge suffixes)."""
    if gt_col in df.columns:
        for _, row in df.head(5).iterrows():
            if _safe_nonempty(row.get(gt_col)):
                return gt_col
    for candidate in [gt_col + "_y", gt_col + "_x", "val_interactions_y", "val_interactions_x"]:
        if candidate in df.columns:
            for _, row in df.head(5).iterrows():
                if _safe_nonempty(row.get(candidate)):
                    return candidate
    return gt_col


def prepare_for_evaluation(df: pd.DataFrame, 
                           model_preds: str, 
                           gt_col: str,
                           user_col: str = 'user_id',
                           item_id_index: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if item_id_index is None:
        item_id_index = _detect_item_id_index(df, gt_col, model_preds, user_col)
    true_rows = []
    for idx, row in df.iterrows():
        user_id = row.get(user_col, idx)
        try:
            uid = int(user_id) if user_id is not None and not (isinstance(user_id, float) and np.isnan(user_id)) else idx
        except (TypeError, ValueError):
            uid = idx
        raw_gt = row.get(gt_col)
        gt_items = [_get_el_at(item, item_id_index) for item in (raw_gt if _safe_nonempty(raw_gt) else [])]
        gt_items = [i for i in gt_items if i is not None]
        for item_id in gt_items:
            try:
                iid = int(item_id)
            except (TypeError, ValueError):
                continue
            true_rows.append({
                'user_id': uid,
                'item_id': iid,
                'rating': 1.0
            })

    ratings_true = pd.DataFrame(true_rows)

    pred_rows = []
    for idx, row in df.iterrows():
        user_id = row.get(user_col, idx)
        try:
            uid = int(user_id) if user_id is not None and not (isinstance(user_id, float) and np.isnan(user_id)) else idx
        except (TypeError, ValueError):
            uid = idx
        predictions = row.get(model_preds) if _safe_nonempty(row.get(model_preds)) else []
        if not _safe_nonempty(predictions):
            continue
        items = _extract_items(predictions)
        scores = _extract_scores(predictions)
        for it, score in zip(items, scores):
            try:
                iid = int(it)
            except (TypeError, ValueError):
                continue
            pred_rows.append({
                'user_id': uid,
                'item_id': iid,
                'prediction': score
            })
    
    ratings_pred = pd.DataFrame(pred_rows)

    # Приводим user_id и item_id к одному типу (recommenders требует одинаковый base dtype)
    for col in ("user_id", "item_id"):
        if col in ratings_true.columns and not ratings_true.empty:
            ratings_true[col] = ratings_true[col].astype(np.int64)
        if col in ratings_pred.columns and not ratings_pred.empty:
            ratings_pred[col] = ratings_pred[col].astype(np.int64)

    return ratings_true, ratings_pred


def calculate_mrr(df: pd.DataFrame, 
                  model_preds: str, 
                  gt_col: str,
                  item_id_index: int = 0) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []
    
    for idx, row in df.iterrows():
        raw_gt = row.get(gt_col)
        raw_gt = raw_gt if _safe_nonempty(raw_gt) else []
        gt_items = set()
        for x in raw_gt:
            iid = _get_el_at(x, item_id_index) if not isinstance(x, (int, float)) else (int(x) if x is not None else None)
            if iid is not None:
                gt_items.add(iid)
            else:
                try:
                    gt_items.add(int(x))
                except (TypeError, ValueError):
                    pass
        predictions = row.get(model_preds) if _safe_nonempty(row.get(model_preds)) else []
        if not predictions or not gt_items:
            reciprocal_ranks.append(0.0)
            continue

        pred_items = _extract_items(predictions)
        pred_items_int = []
        for x in pred_items:
            try:
                pred_items_int.append(int(x))
            except (TypeError, ValueError):
                pass

        for rank, item in enumerate(pred_items_int, start=1):
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
    gt_col = _resolve_gt_col(df, gt_col)
    item_id_index = _detect_item_id_index(df, gt_col, model_preds)
    if verbose:
        print("[Metrics debug] resolved gt_col=%r item_id_index=%s" % (gt_col, item_id_index))
    metrics = {}
    ratings_true, ratings_pred = prepare_for_evaluation(df, model_preds, gt_col, item_id_index=item_id_index)

    if verbose:
        print("[Metrics debug] ratings_true shape:", ratings_true.shape, "ratings_pred shape:", ratings_pred.shape)
        if not ratings_true.empty:
            print("  ratings_true dtypes:", ratings_true[['user_id', 'item_id']].dtypes.to_dict())
        if not ratings_pred.empty:
            print("  ratings_pred dtypes:", ratings_pred[['user_id', 'item_id']].dtypes.to_dict())
        # First user overlap
        for idx, row in df.head(3).iterrows():
            uid = row.get('user_id', idx)
            try:
                uid = int(uid)
            except (TypeError, ValueError):
                uid = idx
            gt_raw = row.get(gt_col)
            gt_set = set()
            for x in (gt_raw if _safe_nonempty(gt_raw) else []):
                iid = _get_el_at(x, item_id_index) if not isinstance(x, (int, float)) else (int(x) if x is not None else None)
                if iid is not None:
                    gt_set.add(iid)
            pred_list = _extract_items(row.get(model_preds) or [])
            pred_set = set()
            for x in pred_list:
                try:
                    pred_set.add(int(x))
                except (TypeError, ValueError):
                    pass
            overlap = len(gt_set & pred_set)
            print(f"  user_id={uid} gt_count={len(gt_set)} pred_count={len(pred_set)} overlap={overlap}")
            if overlap == 0 and gt_set and pred_set:
                gt_sample = sorted(gt_set)[:5]
                rec_sample = sorted(pred_set)[:5]
                print(f"    [ID spaces] gt sample={gt_sample} range=[{min(gt_set)}, {max(gt_set)}] | rec sample={rec_sample} range=[{min(pred_set)}, {max(pred_set)}]")
        if ratings_true.empty and not df.empty:
            cols = [c for c in df.columns if "val" in c.lower() or "interaction" in c.lower() or c == gt_col]
            print("  [GT column inspect] columns related to val/gt:", cols)
            row0 = df.iloc[0]
            raw = row0.get(gt_col)
            print("  gt_col=%r type(raw)=%s len=%s" % (
                gt_col, type(raw).__name__, len(raw) if _safe_nonempty(raw) else 0
            ))
            if _safe_nonempty(raw):
                first = raw[0]
                print("    first element: type=%s repr=%r _first_el(first)=%s" % (
                    type(first).__name__, first, _first_el(first)
                ))
            else:
                for c in cols:
                    val = row0.get(c)
                    if _safe_nonempty(val):
                        print("    %r has len=%s first=%r" % (c, len(val), val[0] if len(val) else None))
    
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
        metrics['MRR'] = calculate_mrr(df, model_preds, gt_col, item_id_index=item_id_index)
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
                raw_train_raw = row.get(train_col)
                raw_train = [_get_el_at(item, item_id_index) for item in (raw_train_raw if _safe_nonempty(raw_train_raw) else [])]
                raw_train = [i for i in raw_train if i is not None]
                try:
                    uid = int(user_id) if user_id is not None else int(idx)
                except (TypeError, ValueError):
                    uid = int(idx)
                for item_id in raw_train:
                    try:
                        train_rows.append({
                            'user_id': uid,
                            'item_id': int(item_id),
                            'rating': 1.0
                        })
                    except (TypeError, ValueError):
                        pass
            
            train_true = pd.DataFrame(train_rows)
            if not train_true.empty:
                for col in ("user_id", "item_id"):
                    if col in train_true.columns:
                        train_true[col] = train_true[col].astype(np.int64)

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
        verbose=verbose
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
