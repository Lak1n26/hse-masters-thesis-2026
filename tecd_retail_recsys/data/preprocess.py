import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class DataPreprocessor:
    def __init__(
        self,
        raw_data_path='t_ecd_small_partial/dataset/small',
        processed_data_dir='processed_data',
        day_begin=1082,
        day_end=1308,
        min_user_interactions=5,
        min_item_interactions=5,
        val_days=1,
        test_days=2,
        users_limit=None
    ):
        self.raw_data_path = raw_data_path
        self.processed_data_dir = processed_data_dir
        self.day_begin = day_begin
        self.day_end = day_end
        self.val_days = val_days
        self.test_days = test_days
        self.users_limit = users_limit

        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions

        self.domain = 'retail'
        self.action_type = 'added-to-cart'
        self.user_to_idx: dict[str, int] = {}
        self.idx_to_user: dict[int, str] = {}
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_item: dict[int, str] = {}


    def load_raw_data(self):
        all_events = []
        events_dir = os.path.join(self.raw_data_path, self.domain, 'events')
        print(f"Loading events from {events_dir}")
        for day in range(self.day_begin, self.day_end + 1):
            file_path = os.path.join(events_dir, f'0{day}.pq')
            if os.path.exists(file_path):
                events = pd.read_parquet(file_path)
                events['day'] = day
                all_events.append(events)
        data = pd.concat(all_events, ignore_index=True)
        print(f"Loaded {len(data):,} total events")
        return data

    def filter_events(self, data):
        filtered = data[data['action_type'] == self.action_type]
        print(f"Filtered to {len(filtered):,} events with action_type='{self.action_type}'")
        return filtered

    def filter_by_interactions(self, data):
        prev_len = 0
        current_len = len(data)

        while prev_len != current_len:
            prev_len = current_len
            user_counts = data["user_id"].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            data = data[data["user_id"].isin(valid_users)]
            item_counts = data["item_id"].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_interactions].index
            data = data[data["item_id"].isin(valid_items)]
            current_len = len(data)

        print(
            f"After filtering (min_user_interactions={self.min_user_interactions}, "
            f"min_item_interactions={self.min_item_interactions}): "
            f"{len(data):,} events, {data['user_id'].nunique():,} users, "
            f"{data['item_id'].nunique():,} items"
        )
        return data

    def preprocess_timestamps(self, data):

        def preprocess_row(row):
            td = pd.to_timedelta(row)
            micros = td / pd.Timedelta(microseconds=1)
            seconds = td.total_seconds()
            return np.int64(seconds)
        
        data['timestamp'] = data['timestamp'].apply(preprocess_row)
        return data

    def create_mappings(self, data):
        unique_users = sorted(data["user_id"].unique())
        unique_items = sorted(data["item_id"].unique())
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        print(
            f"Created mappings: {len(self.user_to_idx)} users, {len(self.item_to_idx)} items"
        )

    def apply_mappings(self, data):
        data = data.copy()
        data["user_id"] = data["user_id"].map(self.user_to_idx)
        data["item_id"] = data["item_id"].map(self.item_to_idx)
        return data

    def temporal_split(self, data):
        """
        Split data by time (Global Temporal Split).
        """
        max_day = data["day"].max()
        test_start = max_day - self.test_days + 1
        val_start = test_start - self.val_days
        train_df = data[data["day"] < val_start]
        val_df = data[(data["day"] >= val_start) & (data["day"] < test_start)]
        test_df = data[data["day"] >= test_start]

        train_users = train_df.user_id.unique()
        val_users = val_df.user_id.unique()
        test_users = test_df.user_id.unique()
        all_included = np.intersect1d(np.intersect1d(train_users, val_users), test_users)
        if self.users_limit is not None:
            all_included = np.random.choice(all_included, size=self.users_limit, replace=False)
        
        train_df = train_df.loc[train_df.user_id.isin(all_included)].copy()
        val_df = val_df.loc[val_df.user_id.isin(all_included)].copy()
        test_df = test_df.loc[test_df.user_id.isin(all_included)].copy()
        
        print(
            f"Temporal split - Train: days < {val_start} ({len(train_df):,} events), "
            f"Val: days {val_start}-{test_start - 1} ({len(val_df):,} events), "
            f"Test: days >= {test_start} ({len(test_df):,} events)"
        )
        print(f'Users in each part (train, val, test) - {all_included.shape[0]}')

        return train_df, val_df, test_df

    def group_by_users(self, data, col_name='interactions'):
        data_grouped = data.groupby('user_id').apply(
            lambda x: [(t1, t2) for t1, t2 in sorted(zip(x.item_id,
                                                        x.timestamp), key=lambda x: x[1])]
        ).reset_index()
        data_grouped.rename({0:col_name}, axis=1, inplace=True)
        return data_grouped

    def get_grouped_data(self, train_df, val_df, test_df):
        train_grouped = self.group_by_users(train_df, col_name='train_interactions')
        val_grouped = self.group_by_users(val_df, col_name='val_interactions')
        test_grouped = self.group_by_users(test_df, col_name='test_interactions')
        joined = train_grouped.merge(val_grouped).merge(test_grouped)
        return joined

    def get_interactions_matrix(self, data, col='train_interactions'):
        user_item_matrix, idx_to_item = self.get_sparse_interactions_matrix(data, col=col)
        return user_item_matrix.toarray(), idx_to_item

    def get_sparse_interactions_matrix(self, data, col='train_interactions'):
        n_users = len(data)
        all_items = set()
        for _, row in data.iterrows():
            for item, _ in row[col]:
                all_items.add(item)

        items = sorted(list(all_items))
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        n_items = len(items)

        row_indices = []
        col_indices = []

        user_indices = {}
        for user_idx, (df_idx, row) in enumerate(data.iterrows()):
            user_indices[df_idx] = user_idx
            for item, _ in row[col]:
                if item in item_to_idx:
                    row_indices.append(user_idx)
                    col_indices.append(item_to_idx[item])

        data = np.ones(len(row_indices))
        user_item_matrix = csr_matrix(
            (data, (row_indices, col_indices)), 
            shape=(n_users, n_items),
            dtype=np.float32
        )
        return user_item_matrix, idx_to_item

    def preprocess(self):
        print("Starting data preprocessing...")
        data = self.load_raw_data()
        data = self.filter_events(data)
        data = self.filter_by_interactions(data)
        data = self.preprocess_timestamps(data)

        self.create_mappings(data)
        data = self.apply_mappings(data)
        train_data, val_data, test_data = self.temporal_split(data)
        return train_data, val_data, test_data

