"""
Prepare grouped data for SASRec training.

This script saves the grouped train/val/test dataframes and metadata
that SASRec needs for training and evaluation.
"""

import os
import argparse
import pandas as pd
from tecd_retail_recsys.data import DataPreprocessor


def save_grouped_data(preprocessor: DataPreprocessor, output_dir: str = 'processed_data'):
    """
    Preprocess data and save grouped dataframes for SASRec.
    
    Parameters:
    -----------
    preprocessor : DataPreprocessor
        Configured data preprocessor instance
    output_dir : str
        Directory to save processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Run preprocessing
    print("Running data preprocessing...")
    train_df, val_df, test_df = preprocessor.preprocess()
    
    # Group by users
    print("Grouping data by users...")
    train_grouped = preprocessor.group_by_users(train_df, col_name='train_interactions')
    val_grouped = preprocessor.group_by_users(val_df, col_name='val_interactions')
    test_grouped = preprocessor.group_by_users(test_df, col_name='test_interactions')
    
    # Save dataframes
    print(f"Saving grouped data to {output_dir}/...")
    train_grouped.to_parquet(os.path.join(output_dir, 'train_grouped.parquet'))
    val_grouped.to_parquet(os.path.join(output_dir, 'val_grouped.parquet'))
    test_grouped.to_parquet(os.path.join(output_dir, 'test_grouped.parquet'))
    
    # Save metadata
    num_items = len(preprocessor.item_to_idx)
    num_users = len(preprocessor.user_to_idx)
    
    with open(os.path.join(output_dir, 'num_items.txt'), 'w') as f:
        f.write(str(num_items))
    
    with open(os.path.join(output_dir, 'num_users.txt'), 'w') as f:
        f.write(str(num_users))
    
    print(f"Saved data for {num_users} users and {num_items} items")
    print(f"Train: {len(train_grouped)} users")
    print(f"Val: {len(val_grouped)} users")
    print(f"Test: {len(test_grouped)} users")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for SASRec training')
    parser.add_argument('--raw_data_path', type=str, default='t_ecd_small_partial/dataset/small',
                       help='Path to raw TECD data')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--day_begin', type=int, default=1082,
                       help='Starting day')
    parser.add_argument('--day_end', type=int, default=1308,
                       help='Ending day')
    parser.add_argument('--min_user_interactions', type=int, default=1,
                       help='Minimum interactions per user')
    parser.add_argument('--min_item_interactions', type=int, default=20,
                       help='Minimum interactions per item')
    parser.add_argument('--val_days', type=int, default=20,
                       help='Number of days for validation')
    parser.add_argument('--test_days', type=int, default=20,
                       help='Number of days for test')
    parser.add_argument('--users_limit', type=int, default=None,
                       help='Limit number of users (for quick testing)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path=args.raw_data_path,
        processed_data_dir=args.output_dir,
        day_begin=args.day_begin,
        day_end=args.day_end,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        val_days=args.val_days,
        test_days=args.test_days,
        users_limit=args.users_limit
    )
    
    # Process and save data
    save_grouped_data(preprocessor, args.output_dir)


if __name__ == '__main__':
    main()
