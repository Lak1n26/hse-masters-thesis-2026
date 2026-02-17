"""
SASRec wrapper for integration with the retail recommendation system.

This module provides a SASRec class that follows the same interface as other models
(TopPopular, EASE, iALS, etc.) for easy integration into the baseline evaluation.
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tecd_retail_recsys.models.sasrec.model import SASRecEncoder
from tecd_retail_recsys.models.sasrec.data import EvalDataset, collate_fn


class SASRec:
    """
    SASRec (Self-Attentive Sequential Recommendation) model wrapper.
    
    This class provides a consistent interface with other recommendation models
    in the project (EASE, iALS, TopPopular, etc.).
    """
    
    def __init__(self, checkpoint_path: str, max_seq_len: int = 50, 
                 device: str = 'cpu', batch_size: int = 256):
        """
        Initialize SASRec model from a saved checkpoint.
        
        Parameters:
        -----------
        checkpoint_path : str
            Path to the saved model checkpoint (.pth file)
        max_seq_len : int
            Maximum sequence length to consider
        device : str
            Device to run inference on ('cpu', 'cuda', or 'mps')
        batch_size : int
            Batch size for inference
        """
        self.checkpoint_path = checkpoint_path
        self.max_seq_len = max_seq_len
        self.device = device
        self.batch_size = batch_size
        self.trained = False
        self.model = None
        
    def load_model(self):
        """Load the trained model from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        
        self.model = torch.load(self.checkpoint_path, weights_only=False, 
                               map_location=self.device).to(self.device)
        self.model.eval()
        self.trained = True
        
    def fit(self, df, col='train_interactions'):
        """
        Dummy fit method for interface compatibility.
        SASRec should be trained separately using the train.py script.
        """
        print("Warning: SASRec should be trained using the train.py script.")
        print("This fit() method only loads a pre-trained model.")
        self.load_model()
        
    def predict(self, df, topn=10, return_scores=False):
        """
        Generate recommendations for users.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'train_interactions' column containing user histories
        topn : int
            Number of items to recommend per user
        return_scores : bool
            If True, return (item_id, score) tuples; else just item_ids
            
        Returns:
        --------
        list
            List of recommendations for each user
        """
        if not self.trained:
            self.load_model()
        
        # Create dataset for evaluation
        eval_dataset = EvalDataset(dataset=df, max_seq_len=self.max_seq_len)
        
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )
        
        # Generate user embeddings
        user_embeddings_list = []
        
        with torch.inference_mode():
            for batch in eval_dataloader:
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
                
                user_emb = self.model(batch)  # (batch_size, embedding_dim)
                user_embeddings_list.append(user_emb)
            
            user_embeddings = torch.cat(user_embeddings_list, dim=0)
            
            # Get item embeddings: 0 to num_items-1 are real items, num_items is padding
            # We only want to score real items (0 to num_items-1)
            item_embeddings = self.model.item_embeddings.weight.data[:self.model.num_items]  # (num_items, dim)
            
            # Compute scores: (num_users, num_items)
            scores = user_embeddings @ item_embeddings.T
            
            # Get top-N items for each user
            topk_scores, topk_indices = torch.topk(scores, k=topn, dim=1)
        
        # Convert to list format
        predictions = []
        for i in range(len(df)):
            rec_items = topk_indices[i].cpu().tolist()
            
            if return_scores:
                rec_scores = topk_scores[i].cpu().tolist()
                items_with_scores = list(zip(rec_items, rec_scores))
                predictions.append(items_with_scores)
            else:
                predictions.append(rec_items)
        
        return predictions
