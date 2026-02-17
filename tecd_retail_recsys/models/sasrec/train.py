import logging
import os
import sys
import pathlib
import random

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
from tecd_retail_recsys.models.sasrec.data import Data, TrainDataset, collate_fn, preprocess_for_sasrec


logging.basicConfig(
    level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def train(
    train_dataloader: DataLoader,
    model: SASRecEncoder,
    optimizer: torch.optim.Optimizer,
    device: str = 'cpu',
    num_epochs: int = 100,
):
    logger.debug('Start training...')

    model.train()

    for epoch_num in range(num_epochs):
        logger.debug(f'Start epoch {epoch_num + 1}')
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_dataloader:
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        logger.debug(f'Epoch {epoch_num + 1} completed. Average loss: {avg_loss:.4f}')

    logger.debug('Training procedure has been finished!')
    return model.state_dict()


@click.command()
@click.option('--exp_name', required=True, type=str, help='Experiment name for saving model')
@click.option('--processed_data_dir', required=True, type=str, default='processed_data/', show_default=True, help='Directory with processed parquet files')
@click.option('--checkpoint_dir', required=True, type=str, default='./checkpoints/', show_default=True, help='Directory to save model checkpoints')
@click.option('--batch_size', required=True, type=int, default=256, show_default=True)
@click.option('--max_seq_len', required=False, type=int, default=50, show_default=True, help='Maximum sequence length')
@click.option('--embedding_dim', required=False, type=int, default=64, show_default=True)
@click.option('--num_heads', required=False, type=int, default=2, show_default=True)
@click.option('--num_layers', required=False, type=int, default=2, show_default=True)
@click.option('--learning_rate', required=False, type=float, default=1e-3, show_default=True)
@click.option('--dropout', required=False, type=float, default=0.2, show_default=True)
@click.option('--seed', required=False, type=int, default=42, show_default=True)
@click.option('--device', required=True, type=str, default='cpu', show_default=True, help='Device: cpu, cuda, or mps')
@click.option('--num_epochs', required=True, type=int, default=10, show_default=True)
def main(
    exp_name: str,
    processed_data_dir: str,
    checkpoint_dir: str,
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    learning_rate: float,
    dropout: float,
    seed: int,
    device: str,
    num_epochs: int,
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
    
    logger.debug(f'Loaded data: {len(train_grouped)} users, {num_items} items')

    checkpoint_path = pathlib.Path(checkpoint_dir) / f'{exp_name}_best_state.pth'
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.debug('Preprocessing data for SASRec...')
    data: Data = preprocess_for_sasrec(train_grouped, val_grouped, test_grouped, num_items, max_seq_len)
    logger.debug('Preprocessing data has finished!')

    train_dataset = TrainDataset(dataset=data.train, num_items=num_items, max_seq_len=max_seq_len)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility, increase if using CUDA
    )

    model = SASRecEncoder(
        num_items=num_items,
        max_sequence_length=max_seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_checkpoint = train(
        train_dataloader=train_dataloader, model=model, optimizer=optimizer, device=device, num_epochs=num_epochs
    )

    logger.debug('Saving model...')

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.load_state_dict(best_checkpoint)
    torch.save(model, checkpoint_path)
    logger.debug(f'Saved model as {checkpoint_path}')


if __name__ == '__main__':
    main()
