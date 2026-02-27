import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


logger = logging.getLogger(__name__)


@dataclass
class Data:
    train: pd.DataFrame
    validation: pd.DataFrame | None
    test: pd.DataFrame
    num_items: int
    trainval: pd.DataFrame | None = None

    def __post_init__(self):
        # Extract unique item IDs from train set
        all_items = set()
        for interactions in self.train['train_interactions']:
            for item_id, _, _ in interactions:
                all_items.add(item_id)
        self.item_ids = sorted(list(all_items))


def preprocess_for_sasrec(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame, num_items: int, 
                          max_seq_len: int = 200) -> Data:
    """
    Preprocesses grouped retail data for SASRec training.

    Args:
        train_df (pd.DataFrame): Grouped training data with 'train_interactions' column
        val_df (pd.DataFrame): Grouped validation data with 'val_interactions' column  
        test_df (pd.DataFrame): Grouped test data with 'test_interactions' column
        num_items (int): Total number of unique items in the dataset
        max_seq_len (int): Maximum sequence length to use

    Returns:
        Data: Dataclass containing train, validation, test DataFrames and item info
    """
    
    # Merge train, val, test by user_id
    merged = train_df.merge(val_df, on='user_id', how='inner')
    merged = merged.merge(test_df, on='user_id', how='inner')
    
    # Truncate train sequences to max_seq_len
    def truncate_interactions(interactions, max_len):
        if len(interactions) > max_len:
            return interactions[-max_len:]
        return interactions
    
    merged['train_interactions'] = merged['train_interactions'].apply(
        lambda x: truncate_interactions(x, max_seq_len)
    )
    
    # Concatenate train and val interactions for training on test
    merged['train_val_interactions'] = merged.apply(
        lambda row: list(row['train_interactions']) + list(row['val_interactions']), axis=1
    )

    return Data(
        train=merged[['user_id', 'train_interactions']],
        validation=merged[['user_id', 'val_interactions']],
        test=merged[['user_id', 'test_interactions']],
        trainval=merged[['user_id', 'train_val_interactions']],
        num_items=num_items
    )


class TrainDataset:
    def __init__(self, dataset: pd.DataFrame, num_items: int, max_seq_len: int, interactions_col: str = 'train_interactions'):
        self._dataset = dataset
        self._num_items = num_items
        self._max_seq_len = max_seq_len
        self._interactions_col = interactions_col

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:
        row = self._dataset.iloc[index]
        
        # Extract item IDs from interactions column (item_id, timestamp, price)
        interactions = row[self._interactions_col]
        # Handle both tuple and array formats
        if isinstance(interactions[0], (list, np.ndarray)):
            item_ids = [int(item[0]) for item in interactions]
        else:
            item_ids = [int(item_id) for item_id, _, _ in interactions]
        
        # Create sequences: input is all but last, target is all but first
        item_sequence = item_ids[:-1][-self._max_seq_len:]
        positive_sequence = item_ids[1:][-self._max_seq_len:]
        
        # Negative sampling from 0 to num_items-1 (all valid items)
        negative_sequence = np.random.randint(0, self._num_items, size=(len(item_sequence),)).tolist()

        return {
            'user.ids': [row['user_id']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'positive.ids': positive_sequence,
            'positive.length': len(positive_sequence),
            'negative.ids': negative_sequence,
            'negative.length': len(negative_sequence),
        }


class EvalDataset:
    def __init__(self, dataset: pd.DataFrame, max_seq_len: int):
        self._dataset = dataset
        self._max_seq_len = max_seq_len

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:
        row = self._dataset.iloc[index]
        
        # Extract item IDs from train_interactions
        train_items_raw = row['train_interactions']
        # Handle both tuple and array formats
        if isinstance(train_items_raw[0], (list, np.ndarray)):
            train_items = [int(item[0]) for item in train_items_raw]
        else:
            train_items = [int(item_id) for item_id, _, _ in train_items_raw]
        item_sequence = train_items[-self._max_seq_len:]
        
        # Extract validation/test items
        next_items_raw = row['val_interactions']
        if isinstance(next_items_raw[0], (list, np.ndarray)):
            next_items = [int(item[0]) for item in next_items_raw]
        else:
            next_items = [int(item_id) for item_id, _, _ in next_items_raw]

        return {
            'user.ids': [row['user_id']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': next_items,
            'labels.length': len(next_items),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of samples into batched tensors suitable for model input.

    This function processes a list of dictionaries, each containing keys like '{prefix}.ids'
    and '{prefix}.length' (the length of the sequence for that prefix). For each such prefix, it:
        - Concatenates all '{prefix}.ids' lists from the batch into a single flat list.
        - Collects all '{prefix}.length' values into a list.
        - Converts the resulting lists into torch.LongTensor objects.

    Args:
        batch (List[Dict]): List of sample dictionaries. Each sample must contain keys of the form
            '{prefix}.ids' (list of ints) and '{prefix}.length' (int).

    Returns:
        Dict[str, torch.Tensor]: Dictionary with keys '{prefix}.ids' and '{prefix}.length' for each prefix,
            where values are 1D torch.LongTensor objects suitable for model input.
    """
    processed_batch = {}
    for key in batch[0].keys():
        if key.endswith('.ids'):
            prefix = key.split('.')[0]
            assert '{}.length'.format(prefix) in batch[0]

            processed_batch[f'{prefix}.ids'] = []
            processed_batch[f'{prefix}.length'] = []

            for sample in batch:
                processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

    for part, values in processed_batch.items():
        processed_batch[part] = torch.tensor(values, dtype=torch.long)

    return processed_batch
