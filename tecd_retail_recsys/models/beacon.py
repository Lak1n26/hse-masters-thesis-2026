import math
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def build_correlation_matrix(
    baskets: List[List[int]],
    num_items: int,
    N: int = 5,
    mu: float = 0.85,
) -> torch.Tensor:
    """
    Build the item-item correlation matrix C ∈ R^{|V| x |V|}.

    For each pair (i, j) co-occurring in a basket, we compute:
        s(i,j) = co_occur(i,j) / sqrt(freq(i) * freq(j))

    Then we keep only the top-N correlations per item and
    normalize rows to sum to 1.  Finally we apply N-th order
    diffusion:  C = mu * S + (1-mu) * S @ S  (simplified).
    """
    freq = Counter()
    co_occur = Counter()
    for basket in baskets:
        items = list(set(basket))
        for item in items:
            freq[item] += 1
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = (min(items[i], items[j]), max(items[i], items[j]))
                co_occur[pair] += 1

    size = num_items + 1
    rows = defaultdict(dict)
    for (i, j), count in co_occur.items():
        fi, fj = freq[i], freq[j]
        if fi > 0 and fj > 0:
            score = count / math.sqrt(fi * fj)
            rows[i][j] = score
            rows[j][i] = score

    indices_i, indices_j, values = [], [], []
    for item in range(1, size):
        neighbors = rows.get(item, {})
        if not neighbors:
            continue
        top_n = sorted(neighbors.items(), key=lambda x: -x[1])[:N]
        total = sum(v for _, v in top_n)
        for neighbor, score in top_n:
            indices_i.append(item)
            indices_j.append(neighbor)
            values.append(score / total if total > 0 else 0.0)

    if not values:
        return torch.zeros(size, size)

    indices = torch.LongTensor([indices_i, indices_j])
    vals = torch.FloatTensor(values)
    S = torch.sparse_coo_tensor(indices, vals, (size, size)).to_dense()

    S2 = S @ S
    row_sums = S2.sum(dim=1, keepdim=True).clamp(min=1e-8)
    S2 = S2 / row_sums

    C = mu * S + (1 - mu) * S2
    return C


class BeaconDataset(Dataset):
    """
    Expanding window: из каждой последовательности длины N
    генерируем N-1 примеров:
      history = baskets[:t],  target = baskets[t]
    """

    def __init__(
        self,
        sequences: List[List[List[int]]],
        num_items: int,
        max_seq_len: int = 30,
        max_basket_size: int = 50,
    ):
        self.num_items = num_items
        self.max_basket_size = max_basket_size
        self.max_seq_len = max_seq_len

        # разворачиваем все подпоследовательности
        self.samples = []  # list of (history: List[List[int]], target: List[int])
        for seq in sequences:
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                history = seq[max(0, t - max_seq_len): t]
                target = seq[t]
                self.samples.append((history, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, target = self.samples[idx]
        history_multihot = []
        for basket in history:
            vec = torch.zeros(self.num_items + 1)
            for item in basket:
                if 1 <= item <= self.num_items:
                    vec[item] = 1.0
            history_multihot.append(vec)

        target_vec = torch.zeros(self.num_items + 1)
        for item in target:
            if 1 <= item <= self.num_items:
                target_vec[item] = 1.0

        return history_multihot, target_vec



def collate_beacon(batch):
    """
    Pad basket sequences to same length.
    """
    histories, targets = zip(*batch)
    lengths = [len(h) for h in histories]
    max_len = max(lengths)
    dim = targets[0].size(0)

    padded = torch.zeros(len(batch), max_len, dim)
    for i, hist in enumerate(histories):
        for t, vec in enumerate(hist):
            padded[i, t] = vec

    targets = torch.stack(targets)
    lengths = torch.LongTensor(lengths)
    return padded, lengths, targets


class BasketEncoder(nn.Module):
    """
    Encodes a multi-hot basket vector into a dense representation,
    taking into account:
      1. Item importance (learned weights)
      2. Item correlations via the correlation matrix C
    """

    def __init__(self, num_items: int, embed_dim: int):
        super().__init__()
        self.num_items = num_items
        size = num_items + 1  # +1 for padding index 0

        self.importance = nn.Parameter(torch.zeros(size))
        nn.init.uniform_(self.importance, -0.1, 0.1)

        self.encoder = nn.Linear(size, embed_dim)

    def forward(
        self,
        baskets: torch.Tensor,
        corr_matrix: torch.Tensor,
    ) -> torch.Tensor:
        B, T, V = baskets.shape

        imp = torch.sigmoid(self.importance).unsqueeze(0).unsqueeze(0)
        weighted = baskets * imp

        basket_sums = weighted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weighted = weighted / basket_sums

        enriched = torch.matmul(weighted, corr_matrix)

        combined = weighted + enriched

        embedded = self.encoder(combined)
        return embedded


class SequenceEncoder(nn.Module):
    """
    LSTM over basket embeddings to capture inter-basket
    sequential associations.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(
        self,
        basket_embeds: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            basket_embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        return h_n[-1]


# ────────────────────────────────────────────────────────────
# 5.  CORRELATION-SENSITIVE PREDICTOR
# ────────────────────────────────────────────────────────────
class CorrelationPredictor(nn.Module):
    """
    Combines sequential signal from LSTM with correlative signal
    from the correlation matrix:

      y_seq  = sigmoid(W_s @ h + b_s)        — sequential scores
      y_corr = C^T @ y_seq                    — correlation-boosted
      y      = α * y_corr + (1 - α) * y_seq  — final scores

    α controls the balance between endogenous (within-basket
    correlations) and exogenous (across-basket sequential) effects.
    """

    def __init__(self, hidden_dim: int, num_items: int, alpha: float = 0.5):
        super().__init__()
        self.num_items = num_items
        size = num_items + 1
        self.alpha = alpha

        self.fc = nn.Linear(hidden_dim, size)

    def forward(
        self,
        h: torch.Tensor,
        corr_matrix: torch.Tensor,
    ) -> torch.Tensor:
        y_seq = torch.sigmoid(self.fc(h))
        y_corr = torch.matmul(y_seq, corr_matrix)

        y_corr_max = y_corr.max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        y_corr = y_corr / y_corr_max

        y = self.alpha * y_corr + (1 - self.alpha) * y_seq
        return y


class Beacon(nn.Module):
    """
    Full Beacon model:
      Basket Encoder -> Sequence Encoder (LSTM) -> Correlation Predictor
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_lstm_layers: int = 1,
        alpha: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.basket_encoder = BasketEncoder(num_items, embed_dim)
        self.sequence_encoder = SequenceEncoder(embed_dim, hidden_dim, n_lstm_layers, dropout)
        self.predictor = CorrelationPredictor(hidden_dim, num_items, alpha)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "corr_matrix", torch.zeros(num_items + 1, num_items + 1)
        )

    def set_correlation_matrix(self, C: torch.Tensor):
        self.corr_matrix.copy_(C)

    def forward(
        self,
        baskets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        basket_embeds = self.basket_encoder(baskets, self.corr_matrix)
        basket_embeds = self.dropout(basket_embeds)

        h = self.sequence_encoder(basket_embeds, lengths)
        h = self.dropout(h)

        scores = self.predictor(h, self.corr_matrix)
        return scores


def ndcg_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 100) -> float:
    B = scores.size(0)
    scores[:, 0] = float("-inf")  # mask padding

    k = min(k, scores.size(-1))
    _, topk_idx = scores.topk(k, dim=-1)

    ndcgs = []
    for i in range(B):
        target_set = set(targets[i].nonzero(as_tuple=True)[0].tolist()) - {0}
        if not target_set:
            continue
        dcg = 0.0
        for rank, item in enumerate(topk_idx[i].tolist()):
            if item in target_set:
                dcg += 1.0 / math.log2(rank + 2)
        n_rel = min(len(target_set), k)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(n_rel))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcgs) if ndcgs else 0.0


def recall_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int = 100) -> float:
    B = scores.size(0)
    scores[:, 0] = float("-inf")
    k = min(k, scores.size(-1))
    _, topk_idx = scores.topk(k, dim=-1)

    recalls = []
    for i in range(B):
        target_set = set(targets[i].nonzero(as_tuple=True)[0].tolist()) - {0}
        if not target_set:
            continue
        hits = len(target_set & set(topk_idx[i].tolist()))
        recalls.append(hits / len(target_set))
    return np.mean(recalls) if recalls else 0.0


class BeaconTrainer:
    def __init__(
        self,
        data: Dict[int, List[List[int]]],
        *,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_lstm_layers: int = 1,
        alpha: float = 0.5,
        dropout: float = 0.1,
        corr_top_n: int = 5,
        corr_mu: float = 0.85,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        epochs: int = 20,
        holdout_ratio: float = 0.2,
        eval_k: int = 100,
        max_seq_len: int = 30,
        max_basket_size: int = 50,
        device: str = "auto",
        seed: int = 42,
        num_workers: int = 0,
        verbose: bool = True,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_k = eval_k
        self.verbose = verbose
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.max_basket_size = max_basket_size

        all_item_ids = set()
        for uid, baskets in data.items():
            for b in baskets:
                all_item_ids.update(b)

        self.item2idx = {item: idx + 1 for idx, item in enumerate(sorted(all_item_ids))}
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        self.num_items = len(self.item2idx)

        all_sequences = []
        all_baskets_flat = []  # for correlation matrix
        for uid, baskets in data.items():
            mapped_seq = []
            for basket in baskets:
                mapped_basket = list(set(
                    self.item2idx[i] for i in basket if i in self.item2idx
                ))
                if mapped_basket:
                    mapped_seq.append(mapped_basket)
                    all_baskets_flat.append(mapped_basket)
            if len(mapped_seq) >= 2:
                all_sequences.append(mapped_seq)

        random.shuffle(all_sequences)
        n_eval = max(1, int(len(all_sequences) * holdout_ratio))
        eval_sequences = all_sequences[:n_eval]
        train_sequences = all_sequences[n_eval:]

        if self.verbose:
            print(f"Items:            {self.num_items}")
            print(f"User sequences:   {len(all_sequences)}")
            print(f"Train sequences:  {len(train_sequences)}")
            print(f"Eval sequences:   {len(eval_sequences)}")
            print(f"Device:           {self.device}")

        if self.verbose:
            print("Building correlation matrix...")
        corr_matrix = build_correlation_matrix(
            all_baskets_flat, self.num_items, N=corr_top_n, mu=corr_mu
        )

        self.train_ds = BeaconDataset(
            train_sequences, self.num_items, max_seq_len, max_basket_size
        )
        self.eval_sequences = eval_sequences

        if self.verbose:
            eval_ds_temp = BeaconDataset(
                eval_sequences, self.num_items, max_seq_len, max_basket_size
            )
            print(f"Items:            {self.num_items}")
            print(f"User sequences:   {len(all_sequences)}")
            print(f"Train sequences:  {len(train_sequences)}")
            print(f"Eval sequences:   {len(eval_sequences)}")
            print(f"Train samples:    {len(self.train_ds)}")
            print(f"Eval samples:     {len(eval_ds_temp)}")
            print(f"Device:           {self.device}")

        self.model = Beacon(
            num_items=self.num_items,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_lstm_layers=n_lstm_layers,
            alpha=alpha,
            dropout=dropout,
        ).to(self.device)

        self.model.set_correlation_matrix(corr_matrix.to(self.device))

        total_params = sum(p.numel() for p in self.model.parameters())
        if self.verbose:
            print(f"Model params:     {total_params:,}")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

    def train(self):
        loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_beacon,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss, n_batches = 0.0, 0
            for baskets, lengths, targets in loader:
                baskets = baskets.to(self.device)
                lengths = lengths.to(self.device)
                targets = targets.to(self.device)

                scores = self.model(baskets, lengths)

                loss = F.binary_cross_entropy(
                    scores[:, 1:], targets[:, 1:],
                    reduction="mean",
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            self.scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)

            if self.verbose:
                metrics = self._evaluate_internal(silent=True)
                print(
                    f"Epoch {epoch:3d}/{self.epochs} │ "
                    f"loss={avg_loss:.4f} │ "
                    f"NDCG@{self.eval_k}={metrics['ndcg']:.4f} │ "
                    f"Recall@{self.eval_k}={metrics['recall']:.4f}"
                )
        return self

    @torch.no_grad()
    def _evaluate_internal(self, silent: bool = False) -> dict:
        return self._run_evaluation(self.eval_sequences, silent=silent)

    @torch.no_grad()
    def evaluate(
        self,
        data: Dict[int, List[List[int]]],
        eval_k: int = None,
    ) -> dict:
        if eval_k is None:
            eval_k = self.eval_k

        sequences = []
        for uid, baskets in data.items():
            mapped_seq = []
            for basket in baskets:
                mapped_basket = list(set(
                    self.item2idx[i] for i in basket if i in self.item2idx
                ))
                if mapped_basket:
                    mapped_seq.append(mapped_basket)
            if len(mapped_seq) >= 2:
                sequences.append(mapped_seq)

        if not sequences:
            print("No valid sequences (need ≥ 2 baskets per user).")
            return {"ndcg": 0.0, "recall": 0.0, "details": []}

        return self._run_evaluation(sequences, eval_k=eval_k, silent=False)

    @torch.no_grad()
    def _run_evaluation(
        self,
        sequences: List[List[List[int]]],
        eval_k: int = None,
        silent: bool = False,
    ) -> dict:
        if eval_k is None:
            eval_k = self.eval_k
        self.model.eval()

        ds = BeaconDataset(
            sequences, self.num_items, self.max_seq_len, self.max_basket_size
        )
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_beacon, num_workers=0,
        )

        all_ndcg, all_recall = [], []
        details = []

        for baskets, lengths, targets in loader:
            baskets = baskets.to(self.device)
            lengths = lengths.to(self.device)
            targets = targets.to(self.device)

            scores = self.model(baskets, lengths)

            all_ndcg.append(ndcg_at_k(scores.clone(), targets, k=eval_k))
            all_recall.append(recall_at_k(scores.clone(), targets, k=eval_k))

        result = {
            "ndcg": float(np.mean(all_ndcg)) if all_ndcg else 0.0,
            "recall": float(np.mean(all_recall)) if all_recall else 0.0,
        }

        if not silent and self.verbose:
            print(f"Sequences evaluated: {len(ds)}")
            print(f"NDCG@{eval_k}:         {result['ndcg']:.4f}")
            print(f"Recall@{eval_k}:       {result['recall']:.4f}")

        self.model.train()
        return result

    @torch.no_grad()
    def predict(
        self,
        user_baskets: List[List[int]],
        top_k: int = 100,
    ) -> List[Tuple[int, float]]:
        self.model.eval()

        mapped_seq = []
        for basket in user_baskets:
            mapped = list(set(
                self.item2idx[i] for i in basket if i in self.item2idx
            ))
            if mapped:
                mapped_seq.append(mapped)

        if not mapped_seq:
            return []

        V = self.num_items + 1
        history = []
        for basket in mapped_seq:
            vec = torch.zeros(V)
            for item in basket:
                vec[item] = 1.0
            history.append(vec)

        baskets = torch.stack(history).unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([len(history)]).to(self.device)

        scores = self.model(baskets, lengths)[0]
        scores[0] = float("-inf")

        k = min(top_k, V)
        topk_scores, topk_idx = scores.topk(k)

        results = []
        for score, idx in zip(topk_scores.tolist(), topk_idx.tolist()):
            orig_id = self.idx2item.get(idx, idx)
            results.append((orig_id, score))
        return results

    @torch.no_grad()
    def evaluate_as_single_basket(
        self,
        val_data: Dict[int, List[List[int]]],
        top_k: int = 100,
    ) -> dict:
        self.model.eval()

        all_baskets = []
        target_items_orig = set()

        for uid, baskets in val_data.items():
            mapped_seq = []
            for basket in baskets:
                mapped = list(set(
                    self.item2idx[i] for i in basket if i in self.item2idx
                ))
                if mapped:
                    mapped_seq.append(mapped)
            if len(mapped_seq) >= 2:
                all_baskets.extend(mapped_seq[:-1])
                for item in mapped_seq[-1]:
                    target_items_orig.add(item)

        if not all_baskets or not target_items_orig:
            return {"ndcg": 0.0, "recall": 0.0, "recommendations": []}

        all_baskets = all_baskets[-self.max_seq_len:]

        V = self.num_items + 1
        history = []
        for basket in all_baskets:
            vec = torch.zeros(V)
            for item in basket:
                vec[item] = 1.0
            history.append(vec)

        baskets_t = torch.stack(history).unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([len(history)]).to(self.device)

        scores = self.model(baskets_t, lengths)[0]
        scores[0] = float("-inf")

        k = min(top_k, V)
        topk_scores, topk_idx = scores.topk(k)

        ranked = topk_idx.tolist()
        target_set = target_items_orig

        dcg = 0.0
        for rank, item in enumerate(ranked):
            if item in target_set:
                dcg += 1.0 / math.log2(rank + 2)
        n_rel = min(len(target_set), k)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(n_rel))
        ndcg_val = dcg / idcg if idcg > 0 else 0.0

        hits = len(target_set & set(ranked))
        recall_val = hits / len(target_set) if target_set else 0.0

        recommendations = [
            (self.idx2item.get(idx, idx), sc)
            for sc, idx in zip(topk_scores.tolist(), topk_idx.tolist())
        ]

        print(f"Target items:   {len(target_set)}")
        print(f"Hits in top-{k}: {hits}")
        print(f"NDCG@{k}:       {ndcg_val:.4f}")
        print(f"Recall@{k}:     {recall_val:.4f}")

        return {
            "ndcg": ndcg_val,
            "recall": recall_val,
            "hits": hits,
            "recommendations": recommendations,
        }

    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "item2idx": self.item2idx,
            "idx2item": self.idx2item,
            "num_items": self.num_items,
            "model_config": {
                "num_items": self.model.num_items,
                "embed_dim": self.model.embed_dim,
                "hidden_dim": self.model.hidden_dim,
                "n_lstm_layers": self.model.sequence_encoder.lstm.num_layers,
                "alpha": self.model.predictor.alpha,
            },
            "trainer_config": {
                "eval_k": self.eval_k,
                "batch_size": self.batch_size,
                "max_seq_len": self.max_seq_len,
                "max_basket_size": self.max_basket_size,
            },
        }
        torch.save(checkpoint, path)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto"):
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        checkpoint = torch.load(path, map_location=dev, weights_only=False)

        trainer = cls.__new__(cls)
        trainer.item2idx = checkpoint["item2idx"]
        trainer.idx2item = checkpoint["idx2item"]
        trainer.num_items = checkpoint["num_items"]
        trainer.device = dev
        trainer.eval_k = checkpoint["trainer_config"]["eval_k"]
        trainer.batch_size = checkpoint["trainer_config"]["batch_size"]
        trainer.max_seq_len = checkpoint["trainer_config"]["max_seq_len"]
        trainer.max_basket_size = checkpoint["trainer_config"]["max_basket_size"]
        trainer.verbose = True
        trainer.num_workers = 0

        cfg = checkpoint["model_config"]
        trainer.model = Beacon(
            num_items=cfg["num_items"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_lstm_layers=cfg["n_lstm_layers"],
            alpha=cfg["alpha"],
        ).to(dev)

        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()

        trainer.optimizer = torch.optim.Adam(trainer.model.parameters())
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=1
        )
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.eval_sequences = []

        print(f"Loaded from {path} | {trainer.num_items} items | device={dev}")
        return trainer
