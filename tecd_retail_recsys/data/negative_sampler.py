"""
Negative Sampling Module for Recommendation System Training

Implements multi-strategy negative sampling:
1. Viewed but Not Purchased (50%)
2. Popular Items (30%)
3. Category-Aware Random (20%)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')


def _sample_negatives_for_batch(
    batch_data: List[Tuple],
    user_viewed_items: Dict,
    user_purchased_items: Dict,
    user_categories: Dict,
    item_categories: Dict,
    category_items: Dict,
    item_popularity: Dict,
    all_items: List[str],
    n_viewed: int,
    n_popular: int,
    n_category: int,
    random_seed: int
) -> List[Dict]:
    """
    Process a batch of positive samples in parallel.
    This function must be at module level to be picklable.
    
    Args:
        batch_data: List of (user_id, positive_item, day) tuples
        ... other parameters are sampler state
    
    Returns:
        List of dicts with negative samples
    """
    # Set random seed for this process
    np.random.seed(random_seed)
    
    results = []
    
    for user_id, positive_item, day in batch_data:
        # Get user's interaction set (to exclude)
        user_interactions = user_purchased_items.get(user_id, set())
        
        negative_items = []
        negative_types = []
        
        # Strategy 1: Viewed but not purchased
        viewed = user_viewed_items.get(user_id, set())
        candidates = viewed - user_interactions
        
        if len(candidates) > 0:
            n_sample = min(n_viewed, len(candidates))
            sampled = np.random.choice(list(candidates), size=n_sample, replace=False)
            negative_items.extend(sampled)
            negative_types.extend(['viewed'] * len(sampled))
        
        # Fill remaining with random if needed
        if len(negative_items) < n_viewed:
            remaining_needed = n_viewed - len(negative_items)
            exclude = user_interactions | set(negative_items)
            random_candidates = list(set(all_items) - exclude)
            if len(random_candidates) > 0:
                n_sample = min(remaining_needed, len(random_candidates))
                sampled = np.random.choice(random_candidates, size=n_sample, replace=False)
                negative_items.extend(sampled)
                negative_types.extend(['viewed'] * len(sampled))
        
        # Strategy 2: Popular items
        exclude = user_interactions | set(negative_items)
        pop_candidates = list(set(all_items) - exclude)
        
        if len(pop_candidates) > 0:
            weights = np.array([np.sqrt(item_popularity.get(item, 1)) for item in pop_candidates])
            weights_sum = weights.sum()
            
            if weights_sum > 0 and not np.isnan(weights_sum):
                weights = weights / weights_sum
                if not np.any(np.isnan(weights)):
                    n_sample = min(n_popular, len(pop_candidates))
                    sampled = np.random.choice(pop_candidates, size=n_sample, replace=False, p=weights)
                    negative_items.extend(sampled)
                    negative_types.extend(['popular'] * len(sampled))
                else:
                    # Fallback to uniform
                    n_sample = min(n_popular, len(pop_candidates))
                    sampled = np.random.choice(pop_candidates, size=n_sample, replace=False)
                    negative_items.extend(sampled)
                    negative_types.extend(['popular'] * len(sampled))
            else:
                # Fallback to uniform
                n_sample = min(n_popular, len(pop_candidates))
                sampled = np.random.choice(pop_candidates, size=n_sample, replace=False)
                negative_items.extend(sampled)
                negative_types.extend(['popular'] * len(sampled))
        
        # Strategy 3: Category-aware
        user_cats = user_categories.get(user_id, set())
        if len(user_cats) == 0 and positive_item in item_categories:
            user_cats = {item_categories[positive_item]}
        
        if len(user_cats) > 0:
            cat_candidates = set()
            for category in user_cats:
                cat_candidates.update(category_items.get(category, []))
            
            exclude = user_interactions | set(negative_items)
            cat_candidates = list(cat_candidates - exclude)
            
            if len(cat_candidates) > 0:
                n_sample = min(n_category, len(cat_candidates))
                sampled = np.random.choice(cat_candidates, size=n_sample, replace=False)
                negative_items.extend(sampled)
                negative_types.extend(['category'] * len(sampled))
        
        # Fill remaining with random if still needed
        total_needed = n_viewed + n_popular + n_category
        if len(negative_items) < total_needed:
            remaining_needed = total_needed - len(negative_items)
            exclude = user_interactions | set(negative_items)
            random_candidates = list(set(all_items) - exclude)
            if len(random_candidates) > 0:
                n_sample = min(remaining_needed, len(random_candidates))
                sampled = np.random.choice(random_candidates, size=n_sample, replace=False)
                negative_items.extend(sampled)
                negative_types.extend(['category'] * len(sampled))
        
        results.append({
            'user_id': user_id,
            'item_ids': negative_items,
            'day': day,
            'types': negative_types
        })
    
    return results


class NegativeSampler:
    """
    Multi-strategy negative sampler for recommendation system.
    
    Samples negative items (not purchased) for each positive item (purchased)
    using a combination of strategies to create informative training data.
    """
    
    def __init__(
        self,
        negative_ratio: int = 10,
        viewed_ratio: float = 0.5,
        popular_ratio: float = 0.3,
        category_ratio: float = 0.2,
        view_window_days: int = 14,
        random_seed: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the negative sampler.
        
        Args:
            negative_ratio: Number of negative samples per positive sample (default: 10)
            viewed_ratio: Proportion of "viewed but not purchased" negatives (default: 0.5)
            popular_ratio: Proportion of popular item negatives (default: 0.3)
            category_ratio: Proportion of category-aware negatives (default: 0.2)
            view_window_days: Time window for "viewed but not purchased" in days (default: 14)
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = no parallelization)
        """
        self.negative_ratio = negative_ratio
        self.viewed_ratio = viewed_ratio
        self.popular_ratio = popular_ratio
        self.category_ratio = category_ratio
        self.view_window_days = view_window_days
        self.random_seed = random_seed
        self.n_jobs = cpu_count() if n_jobs == -1 else max(1, n_jobs)
        
        # Calculate actual counts for each strategy
        self.n_viewed = int(negative_ratio * viewed_ratio)
        self.n_popular = int(negative_ratio * popular_ratio)
        self.n_category = negative_ratio - self.n_viewed - self.n_popular
        
        # Internal state
        self.item_popularity: Dict[str, int] = {}
        self.item_categories: Dict[str, str] = {}
        self.category_items: Dict[str, List[str]] = defaultdict(list)
        self.user_viewed_items: Dict[str, set] = defaultdict(set)
        self.user_purchased_items: Dict[str, set] = defaultdict(set)
        self.user_categories: Dict[str, set] = defaultdict(set)
        self.all_items: List[str] = []
        
        np.random.seed(random_seed)
    
    def fit(
        self,
        events_df: pd.DataFrame,
        items_df: pd.DataFrame,
        positive_df: pd.DataFrame
    ):
        """
        Prepare internal structures for negative sampling.
        
        Args:
            events_df: DataFrame with all events (view, click, added-to-cart)
                       Required columns: ['user_id', 'item_id', 'action_type', 'day']
            items_df: DataFrame with item metadata
                      Required columns: ['item_id', 'item_category']
            positive_df: DataFrame with positive samples (purchases)
                        Required columns: ['user_id', 'item_id', 'day']
        """
        print("Fitting negative sampler...")
        
        # 1. Build item metadata
        self._build_item_metadata(items_df)
        
        # 2. Calculate item popularity from all events
        self._calculate_popularity(events_df)
        
        # 3. Build user interaction history
        self._build_user_history(events_df, positive_df)
        
        print(f"Fitted on {len(self.all_items):,} items, "
              f"{len(self.user_purchased_items):,} users")
        print(f"Average views per user: {np.mean([len(v) for v in self.user_viewed_items.values()]):.1f}")
        print(f"Average purchases per user: {np.mean([len(v) for v in self.user_purchased_items.values()]):.1f}")
    
    def _build_item_metadata(self, items_df: pd.DataFrame):
        """Build item category mappings."""
        # Map item_id -> category
        self.item_categories = dict(zip(
            items_df['item_id'],
            items_df['item_category']
        ))
        
        # Map category -> list of items
        for item_id, category in self.item_categories.items():
            self.category_items[category].append(item_id)
        
        self.all_items = list(self.item_categories.keys())
        print(f"Loaded {len(self.all_items):,} items in {len(self.category_items)} categories")
    
    def _calculate_popularity(self, events_df: pd.DataFrame):
        """Calculate item popularity (number of unique users who interacted)."""
        popularity = events_df.groupby('item_id')['user_id'].nunique()
        self.item_popularity = popularity.to_dict()
        
        # Ensure all items have popularity (even if 0)
        for item_id in self.all_items:
            if item_id not in self.item_popularity:
                self.item_popularity[item_id] = 0
        
        # Diagnostic info
        pop_values = list(self.item_popularity.values())
        print(f"Calculated popularity for {len(self.item_popularity):,} items")
        print(f"  Popularity stats: min={min(pop_values)}, max={max(pop_values)}, "
              f"mean={np.mean(pop_values):.1f}, items_with_0_pop={sum(1 for p in pop_values if p == 0)}")
    
    def _build_user_history(self, events_df: pd.DataFrame, positive_df: pd.DataFrame):
        """Build user interaction history."""
        # User viewed/clicked items
        view_click_events = events_df[
            events_df['action_type'].isin(['view', 'click'])
        ]
        for user_id, group in view_click_events.groupby('user_id'):
            self.user_viewed_items[user_id] = set(group['item_id'].unique())
        
        # User purchased items
        for user_id, group in positive_df.groupby('user_id'):
            purchased = set(group['item_id'].unique())
            self.user_purchased_items[user_id] = purchased
            
            # User categories (from purchases)
            for item_id in purchased:
                if item_id in self.item_categories:
                    self.user_categories[user_id].add(self.item_categories[item_id])
        
        print(f"Built history for {len(self.user_purchased_items):,} users")
    
    def sample_negatives(
        self,
        positive_df: pd.DataFrame,
        events_df: Optional[pd.DataFrame] = None,
        batch_size: int = 50000  # Увеличен с 10K до 50K
    ) -> pd.DataFrame:
        """
        Sample negative items for each positive sample.
        
        Args:
            positive_df: DataFrame with positive samples
                        Required columns: ['user_id', 'item_id', 'day']
            events_df: Optional DataFrame with events for temporal filtering
                      If None, uses pre-fitted data
            batch_size: Size of batches for parallel processing
        
        Returns:
            DataFrame with columns: ['user_id', 'item_id', 'day', 'label', 'negative_type']
        """
        print(f"Sampling negatives with ratio 1:{self.negative_ratio}")
        print(f"  - Viewed but not purchased: {self.n_viewed} per positive")
        print(f"  - Popular items: {self.n_popular} per positive")
        print(f"  - Category-aware: {self.n_category} per positive")
        print(f"  - Parallel jobs: {self.n_jobs}")
        print(f"  - Batch size: {batch_size:,}")
        
        all_samples = []
        
        # Add positive samples with label=1
        positive_samples = positive_df.copy()
        positive_samples['label'] = 1
        positive_samples['negative_type'] = 'positive'
        all_samples.append(positive_samples)
        
        # Prepare data for batch processing
        positive_data = positive_df[['user_id', 'item_id', 'day']].values
        total_positives = len(positive_data)
        
        # Split into batches
        batches = []
        for batch_start in range(0, total_positives, batch_size):
            batch_end = min(batch_start + batch_size, total_positives)
            batch = positive_data[batch_start:batch_end].tolist()
            batches.append(batch)
        
        print(f"Processing {len(batches)} batches...")
        
        # Process batches in parallel or sequential
        if self.n_jobs > 1:
            # Parallel processing
            with Pool(processes=self.n_jobs) as pool:
                process_func = partial(
                    _sample_negatives_for_batch,
                    user_viewed_items=dict(self.user_viewed_items),
                    user_purchased_items=dict(self.user_purchased_items),
                    user_categories=dict(self.user_categories),
                    item_categories=self.item_categories,
                    category_items=dict(self.category_items),
                    item_popularity=self.item_popularity,
                    all_items=self.all_items,
                    n_viewed=self.n_viewed,
                    n_popular=self.n_popular,
                    n_category=self.n_category,
                    random_seed=self.random_seed
                )
                
                batch_results = []
                for i, result in enumerate(pool.imap(process_func, batches)):
                    batch_results.extend(result)
                    processed = min((i + 1) * batch_size, total_positives)
                    print(f"Processed {processed:,} / {total_positives:,} positives")
        else:
            # Sequential processing
            batch_results = []
            for i, batch in enumerate(batches):
                result = _sample_negatives_for_batch(
                    batch,
                    user_viewed_items=dict(self.user_viewed_items),
                    user_purchased_items=dict(self.user_purchased_items),
                    user_categories=dict(self.user_categories),
                    item_categories=self.item_categories,
                    category_items=dict(self.category_items),
                    item_popularity=self.item_popularity,
                    all_items=self.all_items,
                    n_viewed=self.n_viewed,
                    n_popular=self.n_popular,
                    n_category=self.n_category,
                    random_seed=self.random_seed + i
                )
                batch_results.extend(result)
                processed = min((i + 1) * batch_size, total_positives)
                print(f"Processed {processed:,} / {total_positives:,} positives")
        
        # Convert results to DataFrame
        for neg_data in batch_results:
            neg_df = pd.DataFrame({
                'user_id': neg_data['user_id'],
                'item_id': neg_data['item_ids'],
                'day': neg_data['day'],
                'label': 0,
                'negative_type': neg_data['types']
            })
            all_samples.append(neg_df)
        
        result = pd.concat(all_samples, ignore_index=True)
        print(f"\nGenerated {len(result):,} total samples:")
        print(result['negative_type'].value_counts())
        print(f"Positive/Negative ratio: 1:{(result['label'] == 0).sum() / (result['label'] == 1).sum():.1f}")
        
        return result
    
    def get_sampling_stats(self) -> Dict:
        """
        Get statistics about the sampling setup.
        
        Returns:
            Dictionary with sampling statistics
        """
        return {
            'negative_ratio': self.negative_ratio,
            'n_viewed': self.n_viewed,
            'n_popular': self.n_popular,
            'n_category': self.n_category,
            'n_jobs': self.n_jobs,
            'total_items': len(self.all_items),
            'total_users': len(self.user_purchased_items),
            'total_categories': len(self.category_items),
            'avg_user_views': np.mean([len(v) for v in self.user_viewed_items.values()]) if self.user_viewed_items else 0,
            'avg_user_purchases': np.mean([len(v) for v in self.user_purchased_items.values()]) if self.user_purchased_items else 0,
        }


def create_train_val_test_samples(
    train_positives: pd.DataFrame,
    val_positives: pd.DataFrame,
    test_positives: pd.DataFrame,
    train_events: pd.DataFrame,
    items_df: pd.DataFrame,
    negative_ratio: int = 10,
    random_seed: int = 42,
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create train/val/test samples with negatives.
    
    Args:
        train_positives: Training positive samples
        val_positives: Validation positive samples
        test_positives: Test positive samples
        train_events: All events for training period (for fitting sampler)
        items_df: Item metadata
        negative_ratio: Number of negatives per positive
        random_seed: Random seed
        n_jobs: Number of parallel jobs (-1 = all CPUs)
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples) DataFrames
    """
    # Initialize sampler
    sampler = NegativeSampler(
        negative_ratio=negative_ratio,
        random_seed=random_seed,
        n_jobs=n_jobs
    )
    
    # Fit on training data
    sampler.fit(
        events_df=train_events,
        items_df=items_df,
        positive_df=train_positives
    )
    
    # Sample negatives for train
    print("\n" + "="*60)
    print("Sampling TRAIN negatives")
    print("="*60)
    train_samples = sampler.sample_negatives(train_positives)
    
    # Sample negatives for validation
    print("\n" + "="*60)
    print("Sampling VALIDATION negatives")
    print("="*60)
    val_samples = sampler.sample_negatives(val_positives)
    
    # Sample negatives for test
    print("\n" + "="*60)
    print("Sampling TEST negatives")
    print("="*60)
    test_samples = sampler.sample_negatives(test_positives)
    
    return train_samples, val_samples, test_samples
