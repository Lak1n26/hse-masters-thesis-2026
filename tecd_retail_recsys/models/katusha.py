import torch
import torch.nn as nn
from torch.utils.data import Dataset
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


class KATUSHA(nn.Module):
    """
    Hybrid model:
    - categorical embeddings -> self-attention
    - numeric features -> separate MLP branch
    - concat -> prediction MLP head
    """

    def __init__(
        self,
        cat_cardinalities,
        num_numeric_features,
        hidden1=256,
        hidden2=128,
        dropout=0.15,
        attn_dim=64,
        num_heads=4,
        attn_dropout=None,
        numeric_hidden=64,
    ):
        super().__init__()

        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")

        if attn_dropout is None:
            attn_dropout = dropout

        self.cat_cols = list(cat_cardinalities.keys())
        self.attn_dim = attn_dim

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(
                num_embeddings=cardinality,
                embedding_dim=get_embedding_dim(cardinality),
            )
            for col, cardinality in cat_cardinalities.items()
        })

        self.embedding_projections = nn.ModuleDict({
            col: nn.Linear(get_embedding_dim(cardinality), attn_dim)
            for col, cardinality in cat_cardinalities.items()
        })

        self.attn_norm = nn.LayerNorm(attn_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(attn_dim)

        self.attn_ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 2, attn_dim),
        )

        self.ffn_dropout = nn.Dropout(dropout)

        self.numeric_branch = nn.Sequential(
            nn.Linear(num_numeric_features, numeric_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(numeric_hidden, numeric_hidden),
            nn.ReLU(),
        )

        categorical_repr_dim = len(self.cat_cols) * attn_dim
        input_dim = categorical_repr_dim + numeric_hidden

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, 1),
        )

    def forward(self, x_cat, x_num):
        cat_tokens = []

        for i, col in enumerate(self.cat_cols):
            emb = self.embeddings[col](x_cat[:, i])
            token = self.embedding_projections[col](emb)
            cat_tokens.append(token)

        cat_tokens = torch.stack(cat_tokens, dim=1)

        attn_input = self.attn_norm(cat_tokens)

        attn_output, _ = self.self_attention(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            need_weights=False,
        )

        cat_tokens = cat_tokens + self.attn_dropout(attn_output)

        ffn_input = self.ffn_norm(cat_tokens)
        cat_tokens = cat_tokens + self.ffn_dropout(self.attn_ffn(ffn_input))

        categorical_repr = cat_tokens.flatten(start_dim=1)
        numeric_repr = self.numeric_branch(x_num)

        x = torch.cat([categorical_repr, numeric_repr], dim=1)

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
    batch_size=8192,
):
    model.eval()

    X_cat, X_num = transform_frame_inference(
        df=df,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        cat_maps=cat_maps,
        scaler=scaler,
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
