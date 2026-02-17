#!/usr/bin/env python3
"""
Run SASRec model training and evaluation.

This script provides a simple interface to:
1. Prepare data for SASRec
2. Train SASRec model
3. Generate recommendations
4. Evaluate the model

Usage examples:
    # Prepare data
    python run_sasrec.py prepare --output_dir processed_data
    
    # Train model
    python run_sasrec.py train --exp_name sasrec_v1 --num_epochs 10 --device cpu
    
    # Generate recommendations
    python run_sasrec.py predict --exp_name sasrec_v1 --split val --output sasrec_val_recs.parquet
"""

import os
import sys
import subprocess
import argparse


def prepare_data(args):
    """Step 1: Prepare data for SASRec training."""
    print("=" * 80)
    print("STEP 1: Preparing data for SASRec")
    print("=" * 80)
    
    cmd = [
        sys.executable, 'prepare_sasrec_data.py',
        '--output_dir', args.output_dir,
    ]
    
    if args.users_limit:
        cmd.extend(['--users_limit', str(args.users_limit)])
    
    subprocess.run(cmd, check=True)
    print("\n✓ Data preparation completed!")


def train_model(args):
    """Step 2: Train SASRec model."""
    print("=" * 80)
    print("STEP 2: Training SASRec model")
    print("=" * 80)
    
    cmd = [
        sys.executable, 'tecd_retail_recsys/models/sasrec/train.py',
        '--exp_name', args.exp_name,
        '--processed_data_dir', args.data_dir,
        '--checkpoint_dir', args.checkpoint_dir,
        '--batch_size', str(args.batch_size),
        '--max_seq_len', str(args.max_seq_len),
        '--embedding_dim', str(args.embedding_dim),
        '--num_heads', str(args.num_heads),
        '--num_layers', str(args.num_layers),
        '--learning_rate', str(args.learning_rate),
        '--dropout', str(args.dropout),
        '--seed', str(args.seed),
        '--device', args.device,
        '--num_epochs', str(args.num_epochs),
    ]
    
    subprocess.run(cmd, check=True)
    print("\n✓ Model training completed!")


def predict(args):
    """Step 3: Generate recommendations."""
    print("=" * 80)
    print("STEP 3: Generating recommendations")
    print("=" * 80)
    
    cmd = [
        sys.executable, 'tecd_retail_recsys/models/sasrec/eval.py',
        '--exp_name', args.exp_name,
        '--processed_data_dir', args.data_dir,
        '--checkpoint_dir', args.checkpoint_dir,
        '--output_path', args.output,
        '--batch_size', str(args.batch_size),
        '--max_seq_len', str(args.max_seq_len),
        '--topk', str(args.topk),
        '--seed', str(args.seed),
        '--device', args.device,
        '--split', args.split,
    ]
    
    subprocess.run(cmd, check=True)
    print(f"\n✓ Recommendations saved to {args.output}!")


def main():
    parser = argparse.ArgumentParser(description='Run SASRec model pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Prepare data command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare data for training')
    prepare_parser.add_argument('--output_dir', type=str, default='processed_data',
                               help='Output directory for processed data')
    prepare_parser.add_argument('--users_limit', type=int, default=None,
                               help='Limit number of users for quick testing')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train SASRec model')
    train_parser.add_argument('--exp_name', type=str, required=True,
                             help='Experiment name')
    train_parser.add_argument('--data_dir', type=str, default='processed_data',
                             help='Directory with processed data')
    train_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                             help='Directory to save checkpoints')
    train_parser.add_argument('--batch_size', type=int, default=256,
                             help='Batch size')
    train_parser.add_argument('--max_seq_len', type=int, default=50,
                             help='Maximum sequence length')
    train_parser.add_argument('--embedding_dim', type=int, default=64,
                             help='Embedding dimension')
    train_parser.add_argument('--num_heads', type=int, default=2,
                             help='Number of attention heads')
    train_parser.add_argument('--num_layers', type=int, default=2,
                             help='Number of transformer layers')
    train_parser.add_argument('--learning_rate', type=float, default=1e-3,
                             help='Learning rate')
    train_parser.add_argument('--dropout', type=float, default=0.2,
                             help='Dropout rate')
    train_parser.add_argument('--num_epochs', type=int, default=10,
                             help='Number of training epochs')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    train_parser.add_argument('--device', type=str, default='cpu',
                             choices=['cpu', 'cuda', 'mps'],
                             help='Device to train on')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate recommendations')
    predict_parser.add_argument('--exp_name', type=str, required=True,
                               help='Experiment name (must match training)')
    predict_parser.add_argument('--data_dir', type=str, default='processed_data',
                               help='Directory with processed data')
    predict_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                               help='Directory with checkpoints')
    predict_parser.add_argument('--output', type=str, required=True,
                               help='Output path for recommendations')
    predict_parser.add_argument('--split', type=str, default='val',
                               choices=['val', 'test'],
                               help='Split to evaluate on')
    predict_parser.add_argument('--batch_size', type=int, default=256,
                               help='Batch size')
    predict_parser.add_argument('--max_seq_len', type=int, default=50,
                               help='Maximum sequence length')
    predict_parser.add_argument('--topk', type=int, default=100,
                               help='Number of recommendations per user')
    predict_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed')
    predict_parser.add_argument('--device', type=str, default='cpu',
                               choices=['cpu', 'cuda', 'mps'],
                               help='Device to run on')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
