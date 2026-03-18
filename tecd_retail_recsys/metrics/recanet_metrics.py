"""Utility metrics for ReCANet evaluation"""

import numpy as np


def recall_k(gt_items, pred_items, k):
    """
    Calculate Recall@k
    
    Args:
        gt_items: Ground truth items (list)
        pred_items: Predicted items (list) 
        k: Top-k to consider
    
    Returns:
        recall: Recall@k score
    """
    if not gt_items:
        return 0.0
    
    pred_k = pred_items[:k]
    hits = len(set(gt_items) & set(pred_k))
    return hits / len(gt_items)


def ndcg_k(gt_items, pred_items, k):
    """
    Calculate NDCG@k
    
    Args:
        gt_items: Ground truth items (list)
        pred_items: Predicted items (list)
        k: Top-k to consider
    
    Returns:
        ndcg: NDCG@k score
    """
    if not gt_items:
        return 0.0
    
    pred_k = pred_items[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(pred_k):
        if item in gt_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed
    
    # Calculate IDCG
    idcg = 0.0
    for i in range(min(len(gt_items), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg
