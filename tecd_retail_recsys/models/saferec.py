from __future__ import annotations

import math
import copy
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, CosineAnnealingLR

class NBRDataset(Dataset):
    def __init__(
        self,
        user_baskets: Dict[int, List[List[int]]],
        num_items: int,
        max_len: int,
        mode: str = "train",
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.mode = mode

        self.users: List[int] = []
        self.inputs: List[List[List[int]]] = []
        self.targets: List[List[int]] = []

        for uid, baskets in user_baskets.items():
            if len(baskets) < 2:
                continue
            self.users.append(uid)
            self.inputs.append(baskets[:-1])
            self.targets.append(baskets[-1])

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        baskets = self.inputs[idx]
        target_items = self.targets[idx]
        L = self.max_len
        V = self.num_items

        seq = baskets[-L:]  # обрезаем до max_len
        basket_multihot = torch.zeros(L, V)
        mask = torch.zeros(L)
        pad = L - len(seq)
        for k, bsk in enumerate(seq):
            pos = pad + k
            # Нормализация multi-hot по размеру корзины
            valid_items = [item for item in bsk if 0 <= item < V]
            if valid_items:
                val = 1.0 / len(valid_items)
                for item in valid_items:
                    basket_multihot[pos, item] = val
            mask[pos] = 1.0

        target = torch.zeros(V)
        for item in target_items:
            if 0 <= item < V:
                target[item] = 1.0

        freq_vector = basket_multihot.clone()

        return {
            "basket_seq": basket_multihot,
            "target": target,
            "mask": mask,
            "freq_vector": freq_vector,
            "user_id": self.users[idx],
        }


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.drop(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = PointWiseFeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None):
        attn_out, _ = self.attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        x = self.ln1(x + self.drop1(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.drop2(ff_out))
        return x


class SAFERec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        enc_hidden: List[int] | None = None,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.d_model = d_model

        if d_ff is None:
            d_ff = d_model  # В статье: inner hidden = d

        if enc_hidden is None:
            enc_hidden = [d_model]
        layers = []
        in_dim = num_items
        for h_dim in enc_hidden:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        if in_dim != d_model:
            layers.append(nn.Linear(in_dim, d_model))
        self.history_encoder = nn.Sequential(*layers)

        self.position_emb = nn.Embedding(max_len, d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.drop_emb = nn.Dropout(dropout)

        self.item_emb_1 = nn.Embedding(num_items, d_model, padding_idx=None)
        self.item_emb_2 = nn.Embedding(num_items, d_model, padding_idx=None)

        self.freq_proj = nn.Linear(d_model, max_len)

        # Personal Frequency Bias
        self.freq_bias_weight = nn.Parameter(torch.tensor(1.0))

        # Repeat-aware bias
        self.repeat_bias = nn.Parameter(torch.tensor(0.0))
        
        self._init_weights()

        

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(
        self,
        basket_seq: torch.Tensor,
        mask: torch.Tensor,
        freq_vector: torch.Tensor,
    ) -> torch.Tensor:
        B, L, V = basket_seq.shape
        device = basket_seq.device

        W_lat = self.history_encoder(basket_seq)

        positions = torch.arange(L, device=device).unsqueeze(0)
        W_p = W_lat + self.position_emb(positions)
        W_p = self.drop_emb(W_p)

        causal = self._causal_mask(L, device)
        pad_mask = (mask == 0).unsqueeze(1).expand(-1, L, -1)

        h = W_p * mask.unsqueeze(-1)
        for block in self.transformer_blocks:
            h = block(h, attn_mask=causal, key_padding_mask=None)
            h = h * mask.unsqueeze(-1)
        h = self.ln_final(h)


        user_vec = h[:, -1, :]

        I1 = self.item_emb_1.weight
        p_uu = torch.matmul(user_vec, I1.T)

        I2 = self.item_emb_2.weight
        f_i = self.freq_proj(I2)

        h_u = freq_vector.transpose(1, 2)
        p_ui = (h_u * f_i.unsqueeze(0)).sum(dim=-1)

        item_counts = freq_vector.transpose(1, 2).sum(dim=-1)
        seq_len = mask.sum(dim=1, keepdim=True).clamp(min=1)
        p_freq = self.freq_bias_weight * (item_counts / seq_len)

        # repeat mask
        seen_mask = freq_vector.transpose(1, 2).sum(dim=-1).clamp(0, 1)
        logits = p_uu + p_ui + p_freq + self.repeat_bias * seen_mask


        return logits

    def predict_scores(
        self,
        basket_seq: torch.Tensor,
        mask: torch.Tensor,
        freq_vector: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(basket_seq, mask, freq_vector)


def ndcg_at_k(predicted_scores: torch.Tensor,
              target: torch.Tensor,
              k: int = 100) -> float:
    k_safe = min(k, predicted_scores.size(0))
    _, topk_idx = predicted_scores.topk(k_safe)
    relevances = target[topk_idx].cpu().numpy()

    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)

    n_relevant = int(target.sum().item())
    ideal_k = min(n_relevant, k_safe)
    idcg = 0.0
    for i in range(ideal_k):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(predicted_scores: torch.Tensor,
                target: torch.Tensor,
                k: int = 100) -> float:
    k_safe = min(k, predicted_scores.size(0))
    _, topk_idx = predicted_scores.topk(k_safe)
    hits = target[topk_idx].sum().item()
    total = target.sum().item()
    if total == 0:
        return 0.0
    return hits / total


class SAFERecPipeline:
    def __init__(
        self,
        data: Dict[int, List[List[int]]],
        num_items: int,
        max_len: int = 50,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        enc_hidden: List[int] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 5,
        device: str | None = None,
        val_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.raw_data = data
        self.num_items = num_items
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        eligible = {u: bsks for u, bsks in data.items() if len(bsks) >= 3}
        user_ids = list(eligible.keys())
        random.shuffle(user_ids)

        split = int(len(user_ids) * val_ratio)
        val_users = set(user_ids[:split])
        test_users = set(user_ids[split:])


        # Согласно статье: leave-one-basket protocol
        # train: для каждого пользователя все корзины кроме последней
        #        target при обучении — каждая следующая корзина
        # eval:  input = все кроме последней, target = последняя

        self.train_data: Dict[int, List[List[int]]] = {}
        self.val_data: Dict[int, List[List[int]]] = {}
        self.test_data: Dict[int, List[List[int]]] = {}

        for uid in val_users:
            baskets = eligible[uid]
            if len(baskets) >= 3:
                self.train_data[uid] = baskets[:-1]
                self.val_data[uid] = baskets
            else:
                self.train_data[uid] = baskets[:-1]
                self.val_data[uid] = baskets

        for uid in test_users:
            baskets = eligible[uid]
            if len(baskets) >= 3:
                self.train_data[uid] = baskets[:-1]
                self.test_data[uid] = baskets
            else:
                self.train_data[uid] = baskets[:-1]
                self.test_data[uid] = baskets

        for uid, bsks in data.items():
            if len(bsks) == 2 and uid not in self.train_data:
                self.train_data[uid] = bsks

        self.train_dataset = NBRDataset(self.train_data, num_items, max_len, "train")
        self.val_dataset = NBRDataset(self.val_data, num_items, max_len, "eval")
        self.test_dataset = NBRDataset(self.test_data, num_items, max_len, "eval")

        self.model = SAFERec(
            num_items=num_items,
            max_len=max_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            enc_hidden=enc_hidden,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self._full_data = data

        print(f"SAFERecPipeline initialized:")
        print(f"  num_items={num_items}, max_len={max_len}, d_model={d_model}")
        print(f"  n_heads={n_heads}, n_layers={n_layers}, dropout={dropout}")
        print(f"  train users={len(self.train_dataset)}, "
              f"val users={len(self.val_dataset)}, "
              f"test users={len(self.test_dataset)}")
        print(f"  device={self.device}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  total params={total_params:,}")

    def train(self) -> Dict[str, List[float]]:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        history = {"train_loss": [], "val_ndcg100": [], "val_recall100": []}
        best_val_ndcg = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            curr_lr = self.optimizer.param_groups[0]['lr']
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}, lr={curr_lr:.6f}", leave=False):
                basket_seq = batch["basket_seq"].to(self.device)
                target = batch["target"].to(self.device)
                mask = batch["mask"].to(self.device)
                freq_vector = batch["freq_vector"].to(self.device)

                logits = self.model(basket_seq, mask, freq_vector)  # (B, V)

                # default loss
                # loss = F.binary_cross_entropy_with_logits(logits, target)

                # Label smoothing
                eps = 0.1
                target_smooth = target * (1 - eps) + (1 - target) * (eps / self.num_items)
                loss = F.binary_cross_entropy_with_logits(logits, target_smooth)

                # Динамический pos_weight: среднее кол-во негативов / позитивов в батче
                # n_pos = target.sum(dim=1).clamp(min=1)            # (B,)
                # n_neg = self.num_items - n_pos                    # (B,)
                # pw = (n_neg / n_pos).unsqueeze(1).expand_as(target)  # (B, V)
                # loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor([10.0], device=self.device))

                self.optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)

            val_metrics = self._evaluate_dataset(self.val_dataset)
            history["val_ndcg100"].append(val_metrics["ndcg@100"])
            history["val_recall100"].append(val_metrics["recall@100"])

            print(
                f"Epoch {epoch:3d} | "
                f"lr={curr_lr:.6f} | "
                f"loss={avg_loss:.4f} | "
                f"val NDCG@100={val_metrics['ndcg@100']:.4f} | "
                f"val Recall@100={val_metrics['recall@100']:.4f}"
            )

            if val_metrics["ndcg@100"] > best_val_ndcg:
                best_val_ndcg = val_metrics["ndcg@100"]
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best model (val NDCG@100={best_val_ndcg:.4f})")

        return history

    def evaluate(self, on: str = "test") -> Dict[str, float]:
        ds = self.test_dataset if on == "test" else self.val_dataset
        return self._evaluate_dataset(ds)

    def _evaluate_dataset(self, dataset: NBRDataset) -> Dict[str, float]:
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()

        all_ndcg100 = []
        all_recall100 = []
        all_ndcg10 = []
        all_recall10 = []

        with torch.no_grad():
            for batch in loader:
                basket_seq = batch["basket_seq"].to(self.device)
                target = batch["target"].to(self.device)
                mask_b = batch["mask"].to(self.device)
                freq_vector = batch["freq_vector"].to(self.device)

                logits = self.model(basket_seq, mask_b, freq_vector)

                for i in range(logits.size(0)):
                    scores_i = logits[i]
                    target_i = target[i]
                    if target_i.sum() == 0:
                        continue
                    all_ndcg100.append(ndcg_at_k(scores_i, target_i, k=100))
                    all_recall100.append(recall_at_k(scores_i, target_i, k=100))
                    all_ndcg10.append(ndcg_at_k(scores_i, target_i, k=10))
                    all_recall10.append(recall_at_k(scores_i, target_i, k=10))

        return {
            "ndcg@100": np.mean(all_ndcg100) if all_ndcg100 else 0.0,
            "recall@100": np.mean(all_recall100) if all_recall100 else 0.0,
            "ndcg@10": np.mean(all_ndcg10) if all_ndcg10 else 0.0,
            "recall@10": np.mean(all_recall10) if all_recall10 else 0.0,
        }

    def recommend(
        self,
        user_id: int,
        top_k: int = 100,
        baskets: List[List[int]] | None = None,
    ) -> List[Tuple[int, float]]:
        if baskets is None:
            if user_id not in self._full_data:
                raise ValueError(f"User {user_id} not in data")
            baskets = self._full_data[user_id]

        V = self.num_items
        L = self.max_len

        seq = baskets[-L:]
        basket_multihot = torch.zeros(1, L, V)
        mask = torch.zeros(1, L)
        pad = L - len(seq)
        for k, bsk in enumerate(seq):
            pos = pad + k
            for item in bsk:
                if 0 <= item < V:
                    basket_multihot[0, pos, item] = 1.0
            mask[0, pos] = 1.0
        freq_vector = basket_multihot.clone()

        basket_multihot = basket_multihot.to(self.device)
        mask = mask.to(self.device)
        freq_vector = freq_vector.to(self.device)

        scores = self.model.predict_scores(basket_multihot, mask, freq_vector)
        scores = scores.squeeze(0).cpu()

        top_k_safe = min(top_k, scores.size(0))
        topk_vals, topk_idx = scores.topk(top_k_safe)
        result = [
            (int(topk_idx[i].item()), float(topk_vals[i].item()))
            for i in range(top_k)
        ]
        return result

    def recommend_batch(
        self,
        user_ids: List[int],
        top_k: int = 100,
        batch_size: int | None = None,
    ) -> Dict[int, List[Tuple[int, float]]]:
        if batch_size is None:
            batch_size = self.batch_size

        V = self.num_items
        L = self.max_len
        top_k_safe = min(top_k, V)

        self.model.eval()
        results: Dict[int, List[Tuple[int, float]]] = {}

        for start in range(0, len(user_ids), batch_size):
            chunk_uids = user_ids[start : start + batch_size]
            B = len(chunk_uids)

            basket_seq = torch.zeros(B, L, V)
            mask = torch.zeros(B, L)

            for i, uid in enumerate(chunk_uids):
                if uid not in self._full_data:
                    continue
                baskets = self._full_data[uid]
                seq = baskets[-L:]
                pad = L - len(seq)
                for k, bsk in enumerate(seq):
                    pos = pad + k
                    for item in bsk:
                        if 0 <= item < V:
                            basket_seq[i, pos, item] = 1.0
                    mask[i, pos] = 1.0

            freq_vector = basket_seq.clone()

            basket_seq = basket_seq.to(self.device)
            mask = mask.to(self.device)
            freq_vector = freq_vector.to(self.device)

            with torch.no_grad():
                logits = self.model(basket_seq, mask, freq_vector)

            topk_vals, topk_idx = logits.topk(top_k_safe, dim=1)
            topk_vals = topk_vals.cpu()
            topk_idx = topk_idx.cpu()

            for i, uid in enumerate(chunk_uids):
                if uid not in self._full_data:
                    results[uid] = []
                    continue
                results[uid] = [
                    (int(topk_idx[i, j].item()), float(topk_vals[i, j].item()))
                    for j in range(top_k_safe)
                ]

        return results


    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")


def reindex_data(
    data: Dict[int, List[List[int]]]
) -> Tuple[Dict[int, List[List[int]]], int, Dict[int, int], Dict[int, int]]:
    all_items = set()
    for baskets in data.values():
        for bsk in baskets:
            all_items.update(bsk)

    old2new = {item: idx for idx, item in enumerate(sorted(all_items))}
    new2old = {v: k for k, v in old2new.items()}
    num_items = len(old2new)

    new_data = {}
    for uid, baskets in data.items():
        new_data[uid] = [[old2new[i] for i in bsk] for bsk in baskets]

    return new_data, num_items, old2new, new2old
