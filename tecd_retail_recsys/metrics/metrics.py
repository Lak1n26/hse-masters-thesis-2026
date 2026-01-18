import numpy as np

def ndcg_metric(gt_items, predicted):
    at = len(predicted)
    relevance = np.array([1 if x in predicted else 0 for x in gt_items])
    rank_dcg = dcg(relevance)
    if rank_dcg == 0.0:
        return 0.0
    ideal_dcg = dcg(np.sort(relevance)[::-1][:at])
    if ideal_dcg == 0.0:
        return 0.0
    ndcg_ = rank_dcg / ideal_dcg
    return ndcg_


def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float64) + 2)),
                  dtype=np.float64)


def recall_metric(gt_items, predicted):
    n_gt = len(gt_items)
    intersection = len(set(gt_items).intersection(set(predicted)))
    return intersection / n_gt

def evaluate_recommender(df, model_preds, gt_col='test_interactions', topn=10):
    metric_values = []
    for idx, row in df.iterrows():
        gt_items = [x[0] for x in row[gt_col]]
        metric_values.append((ndcg_metric(gt_items, row[model_preds][:topn]),
                              recall_metric(gt_items, row[model_preds][:topn])))
        
    return {'ndcg':np.mean([x[0] for x in metric_values]),
            'recall':np.mean([x[1] for x in metric_values])}


def calculate_metrics(df, model_preds, gt_col, verbose=False):
    at_10 = evaluate_recommender(df, model_preds=model_preds, gt_col=gt_col, topn=10)
    at_100 = evaluate_recommender(df, model_preds=model_preds, gt_col=gt_col, topn=100)
    if verbose:
        print(f'NDCG@10 = {at_10["ndcg"]:.4f}')
        print(f'Recall@10 = {at_10["recall"]:.4f}')
        print(f'NDCG@100 = {at_100["ndcg"]:.4f}')
        print(f'Recall@100 = {at_100["recall"]:.4f}')
    return {
        'NDCG@10': at_10["ndcg"],
        'Recall@10': at_10['recall'],
        'NDCG@100': at_100['ndcg'],
        'Recall@100': at_100['recall']
        }
