import numpy as np
from scipy.sparse import csr_matrix, issparse

class EASE:
    """
    EASE: Embarrassingly Shallow Autoencoders for Sparse Data
    Based on: https://arxiv.org/abs/1905.03375
    """
    def __init__(self, reg_weight: float = 100.0, idx_to_item: dict = None):
        """
        Parameters:
        -----------
        reg_weight : float
            L2 regularization weight
        idx_to_item : dict, optional
            Mapping from matrix column indices to item IDs
            If None, column indices will be used as item IDs
        """
        self.reg_weight = reg_weight
        self.idx_to_item = idx_to_item
        self.trained = False
        self.B = None  # Item-item weight matrix
        self.user_item_matrix = None
        
    def fit(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        n_items = user_item_matrix.shape[1]
        
        # Compute Gram matrix X^T X
        G = self.user_item_matrix.T @ self.user_item_matrix
        
        # Add regularization to diagonal
        G += self.reg_weight * np.eye(n_items)
        
        # Compute P = (X^T X + λI)^(-1)
        P = np.linalg.inv(G)
        
        # Compute B: set diagonal to 0 and normalize
        self.B = P / (-np.diag(P))
        np.fill_diagonal(self.B, 0.0)
        
        self.trained = True
        
    def predict(self, df, topn=10):
        assert self.trained, 'Model not fitted!'
        
        if issparse(self.user_item_matrix):
            user_item_dense = self.user_item_matrix.toarray()
        else:
            user_item_dense = self.user_item_matrix
        
        scores = user_item_dense @ self.B
        predictions = []
        for row_idx, (df_idx, _) in enumerate(df.iterrows()):
            user_scores = scores[row_idx].copy()
            
            # Mask already interacted items
            interacted_items = np.nonzero(user_item_dense[row_idx])[0]
            user_scores[interacted_items] = -np.inf
            
            # Get top-N items
            if topn >= len(user_scores):
                top_indices = np.argsort(user_scores)[::-1]
            else:
                top_indices = np.argpartition(user_scores, -topn)[-topn:]
                top_indices = top_indices[np.argsort(user_scores[top_indices])[::-1]]
            
            if self.idx_to_item is not None:
                top_items = [self.idx_to_item[idx] for idx in top_indices[:topn]]
            else:
                top_items = top_indices[:topn].tolist()
            predictions.append(top_items)
        
        return predictions
