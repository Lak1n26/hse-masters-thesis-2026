"""
ReCANet - PyTorch Implementation
Repeat Consumption-Aware Neural Network for Next Basket Recommendation
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Set random seeds
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


class ReCANetModel(nn.Module):
    """ReCANet PyTorch Model"""
    
    def __init__(self, num_users, num_items, user_embed_size=32, item_embed_size=128,
                 h1=64, h2=64, h3=64, h4=64, h5=64, history_len=20, dropout=0.2):
        super(ReCANetModel, self).__init__()
        
        self.history_len = history_len
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, user_embed_size)
        self.item_embedding = nn.Embedding(num_items, item_embed_size)
        
        # First dense layer after concatenating user and item embeddings
        self.dense1 = nn.Linear(user_embed_size + item_embed_size, h1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Dense layer after concatenating repeated embeddings with temporal features
        # h1 (repeated) + 1 (recency) + 1 (interval) = h1 + 2
        self.dense2 = nn.Linear(h1 + 2, h1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(h1, h2, batch_first=True)
        self.dropout_lstm1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(h2, h3, batch_first=True)
        self.dropout_lstm2 = nn.Dropout(dropout)
        
        # Final dense layers
        self.dense3 = nn.Linear(h3, h4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(h4, h5)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)
        self.output = nn.Linear(h5, 1)
        # Note: No sigmoid here - we'll use BCEWithLogitsLoss which includes it
        
    def forward(self, item_ids, user_ids, recency, intervals):
        """
        Args:
            item_ids: [batch_size, 1]
            user_ids: [batch_size, 1]
            recency: [batch_size, history_len]
            intervals: [batch_size, history_len]
        """
        # Get embeddings
        item_emb = self.item_embedding(item_ids).squeeze(1)  # [batch_size, item_embed_size]
        user_emb = self.user_embedding(user_ids).squeeze(1)  # [batch_size, user_embed_size]
        
        # Concatenate and pass through first dense layer
        concat_emb = torch.cat([item_emb, user_emb], dim=1)  # [batch_size, user_embed + item_embed]
        x = self.relu1(self.dense1(concat_emb))  # [batch_size, h1]
        x = self.dropout1(x)
        
        # Repeat for each timestep
        x_repeated = x.unsqueeze(1).repeat(1, self.history_len, 1)  # [batch_size, history_len, h1]
        
        # Add temporal features
        recency_expanded = recency.unsqueeze(2)  # [batch_size, history_len, 1]
        intervals_expanded = intervals.unsqueeze(2)  # [batch_size, history_len, 1]
        
        # Concatenate all features
        x_concat = torch.cat([x_repeated, recency_expanded, intervals_expanded], dim=2)  # [batch_size, history_len, h1+2]
        
        # Pass through second dense layer
        x = self.relu2(self.dense2(x_concat))  # [batch_size, history_len, h1]
        x = self.dropout2(x)
        
        # LSTM layers
        lstm1_out, _ = self.lstm1(x)  # [batch_size, history_len, h2]
        lstm1_out = self.dropout_lstm1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)  # [batch_size, history_len, h3]
        lstm2_out = self.dropout_lstm2(lstm2_out)
        
        # Take last timestep
        x = lstm2_out[:, -1, :]  # [batch_size, h3]
        
        # Final dense layers
        x = self.relu3(self.dense3(x))
        x = self.dropout3(x)
        x = self.relu4(self.dense4(x))
        x = self.dropout4(x)
        x = self.output(x)  # No sigmoid - BCEWithLogitsLoss will apply it
        
        return x


class BasketDataset(Dataset):
    """PyTorch Dataset for basket recommendation"""
    
    def __init__(self, items, users, history, history2, labels):
        self.items = torch.LongTensor(items)
        self.users = torch.LongTensor(users)
        self.history = torch.FloatTensor(history)
        self.history2 = torch.FloatTensor(history2)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.items[idx].unsqueeze(0),
            self.users[idx].unsqueeze(0),
            self.history[idx],
            self.history2[idx],
            self.labels[idx]
        )


class ReCANetTorch:
    """ReCANet Wrapper Class - PyTorch Version"""
    
    def __init__(self, train_baskets, test_baskets, valid_baskets, dataset_path,
                 basket_count_min=3, min_item_count=5, user_embed_size=32,
                 item_embed_size=128, h1=64, h2=64, h3=64, h4=64, h5=64,
                 history_len=20, dropout=0.2, job_id=1, device='cpu'):
        
        self.train_baskets = train_baskets
        self.test_baskets = test_baskets
        self.valid_baskets = valid_baskets
        self.dataset_path = dataset_path
        self.basket_count_min = basket_count_min
        self.history_len = history_len
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        print(f"Using device: {self.device}")
        
        # Initialize test users
        basket_per_user = self.train_baskets[['user_id','basket_id']].drop_duplicates() \
            .groupby('user_id').agg({'basket_id':'count'}).reset_index()
        self.test_users = basket_per_user[basket_per_user['basket_id'] >= self.basket_count_min]['user_id'].tolist()
        print(f"Number of test users: {len(self.test_users)}")
        
        # Get all items and users
        self.all_items = self.train_baskets[['item_id']].drop_duplicates()['item_id'].tolist()
        self.all_users = self.train_baskets[['user_id']].drop_duplicates()['user_id'].tolist()
        
        # Filter items
        item_counts = self.train_baskets.groupby(['item_id']).size().to_frame(name='item_count').reset_index()
        item_counts = item_counts[item_counts['item_count'] >= min_item_count]
        item_counts_dict = dict(zip(item_counts['item_id'], item_counts['item_count']))
        print(f"Filtered items: {len(item_counts_dict)}")
        
        self.num_items = len(item_counts_dict) + 1
        self.num_users = len(self.all_users) + 1
        
        # Create ID mappings
        self.item_id_mapper = {}
        self.id_item_mapper = {}
        self.user_id_mapper = {}
        self.id_user_mapper = {}
        
        counter = 0
        for item in self.all_items:
            if item in item_counts_dict:
                self.item_id_mapper[item] = counter + 1
                self.id_item_mapper[counter + 1] = item
                counter += 1
                
        for i, user in enumerate(self.all_users):
            self.user_id_mapper[user] = i + 1
            self.id_user_mapper[i + 1] = user
        
        # Model name for caching
        self.model_name = f"{dataset_path}recanet_torch"
        self.data_path = f"{self.model_name}_{job_id}_{user_embed_size}_{item_embed_size}_{h1}_{h2}_{h3}_{h4}_{h5}_{history_len}"
        
        # Initialize model
        self.model = ReCANetModel(
            num_users=self.num_users,
            num_items=self.num_items,
            user_embed_size=user_embed_size,
            item_embed_size=item_embed_size,
            h1=h1, h2=h2, h3=h3, h4=h4, h5=h5,
            history_len=history_len,
            dropout=dropout
        ).to(self.device)
        
        print(f"Model initialized: {self.num_users} users, {self.num_items} items")
        
    def create_train_data(self):
        """Create training data with temporal features"""
        
        # Check if cached data exists
        cache_file = f"{self.model_name}_{self.history_len}_train_cache.npz"
        if os.path.isfile(cache_file):
            print(f"Loading cached training data from {cache_file}")
            data = np.load(cache_file)
            return (data['train_items'], data['train_users'], data['train_history'],
                    data['train_history2'], data['train_labels'])
        
        print(f"Creating training data...")
        
        basket_items = self.train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
        basket_items_dict = dict(zip(basket_items['basket_id'], basket_items['item_id']))
        basket_items_dict['null'] = []
        
        user_baskets = self.train_baskets[['user_id','date','basket_id']].drop_duplicates(). \
            sort_values(['user_id','date'], ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()
        user_baskets_dict = dict(zip(user_baskets['user_id'], user_baskets['basket_id']))
        
        train_users = []
        train_items = []
        train_history = []
        train_history2 = []
        train_labels = []
        
        print(f'Processing {len(self.test_users)} users...')
        
        for c, user in enumerate(tqdm(self.test_users, desc="Creating training data")):
            if c % 1000 == 0 and c > 0:
                print(f'{c} users processed')
            
            baskets = user_baskets_dict[user]
            item_seq = {}
            
            for i, basket in enumerate(baskets):
                for item in basket_items_dict[basket]:
                    if item not in self.item_id_mapper:
                        continue
                    if item not in item_seq:
                        item_seq[item] = []
                    item_seq[item].append(i)
            
            # Only use last 5 baskets for training (adjustable)
            for i in range(max(0, len(baskets) - 5), len(baskets)):
                label_basket = baskets[i]
                all_history_baskets = baskets[:i]
                items = []
                for basket in all_history_baskets:
                    for item in basket_items_dict[basket]:
                        items.append(item)
                items = list(set(items))
                
                for item in items:
                    if item not in self.item_id_mapper:
                        continue
                    
                    index = np.argmax(np.array(item_seq[item]) >= i)
                    if np.max(np.array(item_seq[item])) < i:
                        index = len(item_seq[item])
                    
                    input_history = item_seq[item][:index].copy()
                    if len(input_history) == 0:
                        continue
                    if len(input_history) == 1 and input_history[0] == -1:
                        continue
                    
                    while len(input_history) < self.history_len:
                        input_history.insert(0, -1)
                    
                    real_input_history = []
                    for x in input_history:
                        if x == -1:
                            real_input_history.append(0)
                        else:
                            real_input_history.append(i - x)
                    
                    real_input_history2 = []
                    for j, x in enumerate(input_history[:-1]):
                        if x == -1:
                            real_input_history2.append(0)
                        else:
                            real_input_history2.append(input_history[j+1] - input_history[j])
                    real_input_history2.append(i - input_history[-1])
                    
                    train_users.append(self.user_id_mapper[user])
                    train_items.append(self.item_id_mapper[item])
                    train_history.append(real_input_history[-self.history_len:])
                    train_history2.append(real_input_history2[-self.history_len:])
                    train_labels.append(float(item in basket_items_dict[label_basket]))
        
        train_items = np.array(train_items)
        train_users = np.array(train_users)
        train_history = np.array(train_history)
        train_history2 = np.array(train_history2)
        train_labels = np.array(train_labels)
        
        # Shuffle
        random_indices = np.random.choice(range(len(train_items)), len(train_items), replace=False)
        train_items = train_items[random_indices]
        train_users = train_users[random_indices]
        train_history = train_history[random_indices]
        train_history2 = train_history2[random_indices]
        train_labels = train_labels[random_indices]
        
        # Cache the data
        np.savez(cache_file, train_items=train_items, train_users=train_users,
                 train_history=train_history, train_history2=train_history2,
                 train_labels=train_labels)
        print(f"Cached training data to {cache_file}")
        
        return train_items, train_users, train_history, train_history2, train_labels
    
    def train(self, epochs=5, batch_size=10000, lr=0.001):
        """Train the model"""
        
        # Prepare data
        train_items, train_users, train_history, train_history2, train_labels = self.create_train_data()
        
        print(f"\nTraining data shape: {train_history.shape}")
        print(f"Positive examples: {np.count_nonzero(train_labels):,}")
        print(f"Total examples: {len(train_labels):,}")
        
        # Create dataset and dataloader
        dataset = BasketDataset(train_items, train_users, train_history, train_history2, train_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Calculate class weight for imbalanced dataset
        pos_weight = (len(train_labels) - np.sum(train_labels)) / np.sum(train_labels)
        print(f"Positive class weight: {pos_weight:.2f} (ratio: {100*np.sum(train_labels)/len(train_labels):.2f}% positive)")
        
        # Loss and optimizer - use BCEWithLogitsLoss with pos_weight for class imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (items, users, hist, hist2, labels) in enumerate(pbar):
                # Move to device
                items = items.to(self.device)
                users = users.to(self.device)
                hist = hist.to(self.device)
                hist2 = hist2.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(items, users, hist, hist2)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                # Apply sigmoid for prediction since BCEWithLogitsLoss doesn't include it in output
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            # Save checkpoint
            checkpoint_path = f"{self.data_path}_epoch_{epoch+1}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        print("\nTraining completed!")
    
    def create_test_data(self, test_data='test'):
        """Create test/validation data"""
        
        cache_file = f"{self.model_name}_{self.history_len}_{test_data}_cache.npz"
        if os.path.isfile(cache_file):
            print(f"Loading cached {test_data} data from {cache_file}")
            data = np.load(cache_file)
            return (data['test_items'], data['test_users'], data['test_history'],
                    data['test_history2'], data['test_labels'])
        
        print(f"Creating {test_data} data...")
        
        train_basket_items = self.train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
        train_basket_items_dict = dict(zip(train_basket_items['basket_id'], train_basket_items['item_id']))
        
        train_user_baskets = self.train_baskets[['user_id','date','basket_id']].drop_duplicates(). \
            sort_values(['user_id','date'], ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()
        train_user_baskets_dict = dict(zip(train_user_baskets['user_id'], train_user_baskets['basket_id']))
        
        train_user_items = self.train_baskets[['user_id','item_id']].drop_duplicates().groupby(['user_id'])['item_id'] \
            .apply(list).reset_index()
        train_user_items_dict = dict(zip(train_user_items['user_id'], train_user_items['item_id']))
        
        if test_data == 'test':
            test_user_items = self.test_baskets.groupby(['user_id'])['item_id'].apply(list).reset_index()
        else:
            test_user_items = self.valid_baskets.groupby(['user_id'])['item_id'].apply(list).reset_index()
        test_user_items_dict = dict(zip(test_user_items['user_id'], test_user_items['item_id']))
        
        test_users = []
        test_items = []
        test_history = []
        test_history2 = []
        test_labels = []
        
        train_basket_items_dict['null'] = []
        
        for c, user in enumerate(tqdm(test_user_items_dict, desc=f"Creating {test_data} data")):
            if user not in train_user_baskets_dict:
                continue
            
            baskets = train_user_baskets_dict[user]
            item_seq = {}
            for i, basket in enumerate(baskets):
                for item in train_basket_items_dict[basket]:
                    if item not in self.item_id_mapper:
                        continue
                    if item not in item_seq:
                        item_seq[item] = []
                    item_seq[item].append(i)
            
            label_items = test_user_items_dict[user]
            items = list(set(train_user_items_dict[user]))
            
            for item in items:
                if item not in self.item_id_mapper:
                    continue
                    
                input_history = item_seq[item][-self.history_len:]
                if len(input_history) == 0:
                    continue
                if len(input_history) == 1 and input_history[0] == -1:
                    continue
                    
                while len(input_history) < self.history_len:
                    input_history.insert(0, -1)
                
                real_input_history = []
                for x in input_history:
                    if x == -1:
                        real_input_history.append(0)
                    else:
                        real_input_history.append(len(baskets) - x)
                
                real_input_history2 = []
                for j, x in enumerate(input_history[:-1]):
                    if x == -1:
                        real_input_history2.append(0)
                    else:
                        real_input_history2.append(input_history[j+1] - input_history[j])
                real_input_history2.append(len(baskets) - input_history[-1])
                
                test_users.append(self.user_id_mapper[user])
                test_items.append(self.item_id_mapper[item])
                test_history.append(real_input_history)
                test_history2.append(real_input_history2)
                test_labels.append(float(item in label_items))
        
        test_items = np.array(test_items)
        test_users = np.array(test_users)
        test_history = np.array(test_history)
        test_history2 = np.array(test_history2)
        test_labels = np.array(test_labels)
        
        # Cache the data
        np.savez(cache_file, test_items=test_items, test_users=test_users,
                 test_history=test_history, test_history2=test_history2,
                 test_labels=test_labels)
        print(f"Cached {test_data} data to {cache_file}")
        
        return test_items, test_users, test_history, test_history2, test_labels
    
    def predict(self, batch_size=5000, epoch=None):
        """
        Generate predictions on test set
        
        Args:
            batch_size: Batch size for prediction
            epoch: Which epoch to use (None = last epoch, or specify 1,2,3...)
        """
        
        print("Generating predictions...")
        
        # Create test data
        test_items, test_users, test_history, test_history2, test_labels = self.create_test_data('test')
        
        # Find which checkpoint to use
        if epoch is None:
            # Use the last available epoch
            checkpoint_files = []
            for e in range(1, 100):  # Check up to 100 epochs
                checkpoint_path = f"{self.data_path}_epoch_{e}.pt"
                if os.path.exists(checkpoint_path):
                    checkpoint_files.append((e, checkpoint_path))
            
            if not checkpoint_files:
                raise ValueError("No checkpoint files found! Train the model first.")
            
            last_epoch, last_checkpoint = checkpoint_files[-1]
            print(f"Using last trained epoch: {last_epoch}")
            checkpoint_to_load = last_checkpoint
        else:
            checkpoint_to_load = f"{self.data_path}_epoch_{epoch}.pt"
            if not os.path.exists(checkpoint_to_load):
                raise ValueError(f"Checkpoint for epoch {epoch} not found: {checkpoint_to_load}")
            print(f"Using specified epoch: {epoch}")
        
        # Load model
        self.model.load_state_dict(torch.load(checkpoint_to_load, map_location=self.device))
        self.model.eval()
        
        # Predict on test set
        y_pred = self._predict_batch(test_items, test_users, test_history, test_history2, batch_size)
        
        # Create prediction dict
        prediction_baskets = {}
        for user in self.test_users:
            if user not in self.user_id_mapper:
                prediction_baskets[user] = []
                continue
                
            user_id = self.user_id_mapper[user]
            indices = np.where(test_users == user_id)[0]
            item_scores = y_pred[indices]
            item_ids = test_items[indices]
            
            item_score_dict = {self.id_item_mapper[item_id]: item_scores[i] 
                               for i, item_id in enumerate(item_ids)}
            sorted_items = sorted(item_score_dict.items(), key=lambda x: x[1], reverse=True)
            top_items = [x[0] for x in sorted_items]
            
            prediction_baskets[user] = top_items
        
        print(f"Generated predictions for {len(prediction_baskets)} users")
        return prediction_baskets
    
    def _predict_batch(self, items, users, history, history2, batch_size):
        """Helper function to predict in batches"""
        
        dataset = BasketDataset(items, users, history, history2, np.zeros(len(items)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        predictions = []
        with torch.no_grad():
            for items_batch, users_batch, hist_batch, hist2_batch, _ in tqdm(dataloader, desc="Predicting"):
                items_batch = items_batch.to(self.device)
                users_batch = users_batch.to(self.device)
                hist_batch = hist_batch.to(self.device)
                hist2_batch = hist2_batch.to(self.device)
                
                outputs = self.model(items_batch, users_batch, hist_batch, hist2_batch)
                # Apply sigmoid since model outputs logits
                probs = torch.sigmoid(outputs)
                predictions.extend(probs.cpu().numpy().flatten())
        
        return np.array(predictions)
