import logging
import sys
import pathlib
import random
import os

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tecd_retail_recsys.models.sasrec.model import SASRecEncoder
from tecd_retail_recsys.models.sasrec.data import Data, EvalDataset, collate_fn, preprocess_for_sasrec


logging.basicConfig(
    level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def infer_users(eval_dataloader: DataLoader, model: torch.nn.Module, device: str):
    """Generate user embeddings from their interaction sequences."""
    user_ids = []
    user_embeddings = []

    model.eval()
    for batch in eval_dataloader:
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        user_ids.append(batch['user.ids'])  # (batch_size)
        user_embeddings.append(model(batch))  # (batch_size, embedding_dim)

    return torch.cat(user_ids, dim=0), torch.cat(user_embeddings, dim=0)


def infer_items(model: SASRecEncoder):
    """Get item embeddings from the model."""
    return model.item_embeddings.weight.data


def generate_recommendations(user_embeddings: torch.Tensor, 
                            item_embeddings: torch.Tensor,
                            user_ids: torch.Tensor,
                            topk: int = 100) -> pd.DataFrame:
    """
    Generate top-K recommendations for each user.
    
    Args:
        user_embeddings: Tensor of shape (num_users, embedding_dim)
        item_embeddings: Tensor of shape (num_items + 1, embedding_dim) (0 is padding)
        user_ids: Tensor of user IDs
        topk: Number of items to recommend per user
        
    Returns:
        DataFrame with columns: user_id, recommendations (list of item_ids)
    """
    # Remove padding embedding (index 0)
    item_embeddings = item_embeddings[1:]  # (num_items, embedding_dim)
    
    # Compute scores: (num_users, num_items)
    scores = user_embeddings @ item_embeddings.T
    
    # Get top-K items for each user
    topk_scores, topk_indices = torch.topk(scores, k=topk, dim=1)
    
    # Convert to lists
    recommendations = []
    for i in range(len(user_ids)):
        user_id = user_ids[i].item()
        rec_items = topk_indices[i].cpu().tolist()
        recommendations.append({
            'user_id': user_id,
            'recommendations': rec_items
        })
    
    return pd.DataFrame(recommendations)


@click.command()
@click.option('--exp_name', required=True, type=str, help='Experiment name (same as training)')
@click.option('--processed_data_dir', required=True, type=str, default='processed_data/', show_default=True, help='Directory with processed parquet files')
@click.option('--checkpoint_dir', required=True, type=str, default='./checkpoints/', show_default=True, help='Directory with saved model')
@click.option('--output_path', required=True, type=str, help='Path to save recommendations parquet file')
@click.option('--batch_size', required=True, type=int, default=256, show_default=True)
@click.option('--max_seq_len', required=False, type=int, default=50, show_default=True)
@click.option('--topk', required=False, type=int, default=100, show_default=True, help='Number of recommendations per user')
@click.option('--seed', required=False, type=int, default=42, show_default=True)
@click.option('--device', required=True, type=str, default='cpu', show_default=True)
@click.option('--split', required=True, type=click.Choice(['val', 'test']), default='val', show_default=True, help='Which split to evaluate')
def main(
    exp_name: str,
    processed_data_dir: str,
    checkpoint_dir: str,
    output_path: str,
    batch_size: int,
    max_seq_len: int,
    topk: int,
    seed: int,
    device: str,
    split: str,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision('high')

    # Load preprocessed data
    logger.debug('Loading preprocessed data...')
    train_grouped = pd.read_parquet(os.path.join(processed_data_dir, 'train_grouped.parquet'))
    val_grouped = pd.read_parquet(os.path.join(processed_data_dir, 'val_grouped.parquet'))
    test_grouped = pd.read_parquet(os.path.join(processed_data_dir, 'test_grouped.parquet'))
    
    # Load item count
    with open(os.path.join(processed_data_dir, 'num_items.txt'), 'r') as f:
        num_items = int(f.read().strip())

    logger.debug('Preprocessing data for evaluation...')
    data: Data = preprocess_for_sasrec(train_grouped, val_grouped, test_grouped, num_items, max_seq_len)
    
    # Choose validation or test split
    if split == 'val':
        eval_df = data.validation
        logger.debug(f'Evaluating on validation set with {len(eval_df)} users')
    else:
        eval_df = data.test
        logger.debug(f'Evaluating on test set with {len(eval_df)} users')
    
    # Merge with train data to get full sequences
    eval_data = data.train.merge(eval_df, on='user_id', how='inner')
    
    eval_dataset = EvalDataset(dataset=eval_data, max_seq_len=max_seq_len)

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    # Load model
    checkpoint_path = pathlib.Path(checkpoint_dir) / f'{exp_name}_best_state.pth'
    logger.debug(f'Loading model from {checkpoint_path}')
    model = torch.load(checkpoint_path, weights_only=False, map_location=device).to(device)
    model.eval()
    
    with torch.inference_mode():
        logger.debug('Generating user embeddings...')
        user_ids, user_embeddings = infer_users(eval_dataloader=eval_dataloader, model=model, device=device)

        logger.debug('Getting item embeddings...')
        item_embeddings = infer_items(model=model)

        logger.debug('Generating recommendations...')
        recommendations_df = generate_recommendations(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            user_ids=user_ids,
            topk=topk
        )

    # Save recommendations
    logger.debug(f'Saving recommendations to {output_path}')
    recommendations_df.to_parquet(output_path, index=False)
    logger.debug(f'Saved {len(recommendations_df)} user recommendations')


if __name__ == '__main__':
    main()
