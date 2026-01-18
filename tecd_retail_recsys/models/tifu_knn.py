import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import List, Dict, Optional


class TIFUKNN:
    """
    Temporal-Item-Frequency-Based User-KNN for Next-basket Recommendation
    Based on: https://arxiv.org/abs/2006.00556 (SIGIR 2020)
    
    The model builds Personalized Item Frequency (PIF) vectors for each user
    with temporal decay, then uses KNN for collaborative filtering.
    """
    
    def __init__(
        self,
        n_neighbors: int = 300,
        within_decay_rate: float = 0.9,
        group_decay_rate: float = 0.7,
        alpha: float = 0.7,
        n_groups: int = 7,
        idx_to_item: Dict[int, int] = None
    ):
        """
        Parameters:
        -----------
        n_neighbors : int
            Number of nearest neighbors to consider
        within_decay_rate : float
            Decay rate within baskets/sessions (r_b in paper)
            Older baskets get exponentially lower weight
        group_decay_rate : float
            Decay rate between groups of baskets (r_g in paper)
        alpha : float
            Balance between repetition (user's own history) and exploration (neighbors)
            score = alpha * own_pif + (1-alpha) * neighbors_pif
        n_groups : int
            Number of temporal groups to split user history into
        idx_to_item : dict, optional
            Mapping from matrix column indices to item IDs
        """
        self.n_neighbors = n_neighbors
        self.within_decay_rate = within_decay_rate
        self.group_decay_rate = group_decay_rate
        self.alpha = alpha
        self.n_groups = n_groups
        self.idx_to_item = idx_to_item
        
        self.user_pif = None  # PIF vectors: (n_users, n_items)
        self.user_similarities = None  # Precomputed similarity matrix
        self.neighbor_indices = None  # Precomputed K nearest neighbors for each user
        self.trained = False
        
    def fit(self, df, col='train_interactions'):
        """
        Build PIF (Personalized Item Frequency) vectors for all users.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame where each row is a user, and df[col] contains
            list of (item_id, timestamp) tuples sorted by timestamp
        col : str
            Column name containing interaction history
        """
        n_users = len(df)
        
        # Get all unique items to determine matrix size
        all_items = set()
        for idx, row in df.iterrows():
            for item, _ in row[col]:
                all_items.add(item)
        
        items = sorted(list(all_items))
        self.item_to_idx = {item: idx for idx, item in enumerate(items)}
        self.idx_to_item_local = {idx: item for item, idx in self.item_to_idx.items()}
        n_items = len(items)
        
        print(f"Building TIFU-KNN PIF vectors for {n_users} users and {n_items} items...")
        
        # Initialize PIF matrix
        self.user_pif = np.zeros((n_users, n_items), dtype=np.float32)
        
        # Build PIF vector for each user
        for user_idx, (df_idx, row) in enumerate(tqdm(df.iterrows(), total=n_users, desc="Computing PIF")):
            interactions = row[col]
            if len(interactions) == 0:
                continue
            
            # Group interactions into temporal segments
            # We'll create "baskets" by grouping consecutive interactions
            # For simplicity, treat each interaction as a separate basket
            # (can modify this to group by sessions/days if needed)
            baskets = [[item] for item, _ in interactions]
            n_baskets = len(baskets)
            
            if n_baskets == 0:
                continue
            
            # Split into groups
            group_size = max(1, n_baskets // self.n_groups)
            
            for basket_idx, basket in enumerate(baskets):
                # Determine group index (0 to n_groups-1)
                group_idx = min(self.n_groups - 1, basket_idx // group_size)
                
                # Group decay: older groups get lower weight
                # Most recent group (highest group_idx) should have highest weight
                group_weight = self.group_decay_rate ** (self.n_groups - 1 - group_idx)
                
                # Within-basket decay: older baskets get lower weight
                within_weight = self.within_decay_rate ** (n_baskets - 1 - basket_idx)
                
                # Combined weight
                weight = group_weight * within_weight
                
                # Add weighted frequency to PIF vector
                for item in basket:
                    if item in self.item_to_idx:
                        item_idx = self.item_to_idx[item]
                        self.user_pif[user_idx, item_idx] += weight
        
        # Normalize PIF vectors (optional but often helps)
        row_sums = np.linalg.norm(self.user_pif, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.user_pif = self.user_pif / row_sums
        
        # Precompute user-user similarities and find K nearest neighbors
        print("Precomputing user similarities and neighbors...")
        self._precompute_neighbors()
        
        self.trained = True
        print("TIFU-KNN training completed!")
    
    def _precompute_neighbors(self):
        """
        Precompute K nearest neighbors for each user.
        This significantly speeds up prediction.
        """
        n_users = self.user_pif.shape[0]
        
        # Compute all pairwise similarities at once
        self.user_similarities = cosine_similarity(self.user_pif)
        
        # Set diagonal to -inf (exclude self)
        np.fill_diagonal(self.user_similarities, -np.inf)
        
        # Find K nearest neighbors for each user
        # Use argpartition for efficiency
        k = min(self.n_neighbors, n_users - 1)
        self.neighbor_indices = np.zeros((n_users, k), dtype=np.int32)
        
        print(f"Finding {k} nearest neighbors for each user...")
        for u in tqdm(range(n_users), desc="Finding neighbors"):
            # Get top-k neighbors
            neighbors = np.argpartition(-self.user_similarities[u], k)[:k]
            # Sort by similarity (descending)
            neighbors = neighbors[np.argsort(-self.user_similarities[u, neighbors])]
            self.neighbor_indices[u] = neighbors
        
        # Precompute neighbor PIF averages
        print("Precomputing collaborative signals...")
        self.neighbor_pif = np.zeros_like(self.user_pif)
        for u in tqdm(range(n_users), desc="Computing neighbor PIF"):
            self.neighbor_pif[u] = np.mean(self.user_pif[self.neighbor_indices[u]], axis=0)
        
        self.trained = True
        print("TIFU-KNN training completed!")
        
    def predict(self, df, topn: int = 100) -> List[List[int]]:
        """
        Generate top-N recommendations for users.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with users (same as used in fit)
        topn : int
            Number of recommendations per user
            
        Returns:
        --------
        List of lists with item IDs for each user
        """
        assert self.trained, 'Model not fitted!'
        
        n_users = len(df)
        n_items = self.user_pif.shape[1]
        
        print(f"Generating recommendations for {n_users} users...")
        
        # Compute all scores at once: alpha * own_pif + (1-alpha) * neighbors_pif
        all_scores = self.alpha * self.user_pif + (1.0 - self.alpha) * self.neighbor_pif
        
        # Create mask for interacted items
        print("Masking interacted items...")
        for user_idx, (df_idx, row) in enumerate(tqdm(df.iterrows(), total=n_users, desc="Creating masks")):
            for item, _ in row['train_interactions']:
                if item in self.item_to_idx:
                    item_idx = self.item_to_idx[item]
                    all_scores[user_idx, item_idx] = -np.inf
        
        # Get top-N for all users
        print("Computing top-N recommendations...")
        predictions = []
        
        for user_idx in tqdm(range(n_users), desc="Extracting top-N"):
            user_scores = all_scores[user_idx]
            
            # Get top-N items
            if topn >= n_items:
                top_indices = np.argsort(-user_scores)
            else:
                top_indices = np.argpartition(-user_scores, topn)[:topn]
                top_indices = top_indices[np.argsort(-user_scores[top_indices])]
            
            # Convert matrix indices to item IDs
            if self.idx_to_item is not None:
                # Use provided mapping (for consistency with other models)
                top_items = [self.idx_to_item[self.idx_to_item_local[idx]] 
                           if self.idx_to_item_local[idx] in self.idx_to_item 
                           else self.idx_to_item_local[idx]
                           for idx in top_indices[:topn]]
            else:
                # Use local mapping
                top_items = [self.idx_to_item_local[idx] for idx in top_indices[:topn]]
            
            predictions.append(top_items)
        
        return predictions
