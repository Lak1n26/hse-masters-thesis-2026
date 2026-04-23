
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.metrics import roc_auc_score

def transform_frame(df, categorical_features, numeric_features, cat_maps, scaler, target_col="target"):
    df = df.copy()

    X_cat = np.zeros((len(df), len(categorical_features)), dtype=np.int64)
    for j, col in enumerate(categorical_features):
        mapping = cat_maps[col]
        X_cat[:, j] = df[col].astype(str).fillna("__NA__").map(mapping).fillna(0).astype(np.int64).values

    X_num = scaler.transform(
        df[numeric_features].fillna(0.0).astype(np.float32)
    ).astype(np.float32)

    y = df[target_col].astype(np.float32).values
    return X_cat, X_num, y


class RecDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]
    

def get_embedding_dim(cardinality: int) -> int:
    return min(64, max(8, int(np.sqrt(cardinality)) + 1))


class TwoLayerRecMLP(nn.Module):
    def __init__(self, cat_cardinalities, num_numeric_features, hidden1=256, hidden2=128, dropout=0.15):
        super().__init__()

        self.cat_cols = list(cat_cardinalities.keys())

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(
                num_embeddings=cardinality,
                embedding_dim=get_embedding_dim(cardinality)
            )
            for col, cardinality in cat_cardinalities.items()
        })

        emb_dim_total = sum(get_embedding_dim(card) for card in cat_cardinalities.values())
        input_dim = emb_dim_total + num_numeric_features

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, 1)
        )

    def forward(self, x_cat, x_num):
        embs = []
        for i, col in enumerate(self.cat_cols):
            embs.append(self.embeddings[col](x_cat[:, i]))

        x = torch.cat(embs + [x_num], dim=1)
        logits = self.mlp(x).squeeze(1)
        return logits
    

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    for x_cat, x_num, y in loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        y = y.to(device)

        logits = model(x_cat, x_num)
        loss = criterion(logits, y)

        total_loss += loss.item() * len(y)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy()

    probs = 1 / (1 + np.exp(-all_logits))
    auc = roc_auc_score(all_targets, probs) if len(np.unique(all_targets)) > 1 else np.nan

    return total_loss / len(loader.dataset), auc


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x_cat, x_num, y in loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x_cat, x_num)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def predict_scores(model, df, categorical_features, numeric_features, cat_maps, scaler, device, batch_size=8192):
    model.eval()

    X_cat, X_num, _ = transform_frame(df, categorical_features, numeric_features, cat_maps, scaler)

    preds = []
    for start in range(0, len(df), batch_size):
        end = start + batch_size

        x_cat = torch.tensor(X_cat[start:end], dtype=torch.long, device=device)
        x_num = torch.tensor(X_num[start:end], dtype=torch.float32, device=device)

        logits = model(x_cat, x_num)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    return np.concatenate(preds)

def transform_frame_inference(df, categorical_features, numeric_features, cat_maps, scaler):
    df = df.copy()

    X_cat = np.zeros((len(df), len(categorical_features)), dtype=np.int64)
    for j, col in enumerate(categorical_features):
        mapping = cat_maps[col]
        X_cat[:, j] = (
            df[col]
            .astype(str)
            .fillna("__NA__")
            .map(mapping)
            .fillna(0)
            .astype(np.int64)
            .values
        )

    X_num = scaler.transform(
        df[numeric_features].fillna(0.0).astype(np.float32)
    ).astype(np.float32)

    return X_cat, X_num

@torch.no_grad()
def predict_scores_inference(
    model,
    df,
    categorical_features,
    numeric_features,
    cat_maps,
    scaler,
    device,
    batch_size=8192
):
    model.eval()

    X_cat, X_num = transform_frame_inference(
        df=df,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        cat_maps=cat_maps,
        scaler=scaler
    )

    preds = []

    for start in range(0, len(df), batch_size):
        end = start + batch_size

        x_cat = torch.tensor(X_cat[start:end], dtype=torch.long, device=device)
        x_num = torch.tensor(X_num[start:end], dtype=torch.float32, device=device)

        logits = model(x_cat, x_num)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    return np.concatenate(preds)
