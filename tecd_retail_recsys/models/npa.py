"""
Neural Pattern Associator (NPA) for Within-Basket Recommendation

Based on: "Within-basket Recommendation via Neural Pattern Associator"
Paper: https://arxiv.org/abs/2401.16433
"""


import math
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class BasketDataset(Dataset):
    """
    Each sample = one basket.
    During training we use Any-Order Autoregressive:
      – randomly permute the basket
      – pick a random split point k (1 ≤ k < len)
      – input  = first k items  (the "known" part)
      – target = item at position k  (the next item to predict)

    This is repeated with different random permutations each
    epoch, which is equivalent to uniformly sampling over all
    orderings and all split points, as described in the paper.
    """

    def __init__(
        self,
        baskets: List[List[int]],
        num_items: int,
        max_basket_size: int = 128,
    ):
        # удаляем дубликаты внутри корзины, но сохраняем порядок
        self.baskets = []
        for b in baskets:
            seen = set()
            deduped = []
            for item in b:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            if len(deduped) >= 2:            # нужна хотя бы 1 input + 1 target
                self.baskets.append(deduped[:max_basket_size])
        self.num_items = num_items

    def __len__(self):
        return len(self.baskets)

    def __getitem__(self, idx):
        basket = self.baskets[idx].copy()
        random.shuffle(basket)               # any-order
        k = random.randint(1, len(basket) - 1)
        input_items = basket[:k]
        target_item = basket[k]
        return input_items, target_item


def collate_baskets(batch):
    """
    Pad input baskets to the same length
    """
    inputs, targets = zip(*batch)
    max_len = max(len(inp) for inp in inputs)
    padded, masks = [], []
    for inp in inputs:
        pad_len = max_len - len(inp)
        padded.append(inp + [0] * pad_len)
        masks.append([True] * len(inp) + [False] * pad_len)
    return (
        torch.LongTensor(padded),
        torch.BoolTensor(masks),
        torch.LongTensor(targets),
    )


class VQAModule(nn.Module):
    """
    One Vector-Quantized Attention head.

    Conceptual flow:
    1. Item embeddings  →  mean-pool (basket repr.)
    2. Attention 1: basket repr. queries a learnable *codebook*
       of combination-pattern vectors  →  weighted sum  =  *pattern*
    3. Attention 2: pattern queries item embeddings
       →  weighted sum  =  *context vector*
    4. Context is used downstream (next layer or prediction head).

    """

    def __init__(self, embed_dim: int, codebook_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size

        # learnable codebook
        self.codebook = nn.Parameter(
            torch.randn(codebook_size, embed_dim) * 0.02
        )

        # projections for the two attention steps
        self.W_q1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k1 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.W_q2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v2 = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scale = math.sqrt(embed_dim)

    def forward(
        self,
        item_embeds: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step A – basket → codebook attention
        """
        # mean-pool basket items
        mask_f = mask.unsqueeze(-1).float()
        basket_repr = (item_embeds * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        # query from basket repr, keys from codebook
        q1 = self.W_q1(basket_repr).unsqueeze(1)
        k1 = self.W_k1(self.codebook).unsqueeze(0)
        attn1 = (q1 * k1).sum(-1) / self.scale
        attn1 = F.softmax(attn1, dim=-1)

        # weighted codebook vectors → combination pattern
        pattern = torch.matmul(attn1, self.codebook)

        """
        Step B – pattern → item attention  →  context
        """
        q2 = self.W_q2(pattern).unsqueeze(1)
        k2 = self.W_k2(item_embeds)
        v2 = self.W_v2(item_embeds)

        attn2 = (q2 * k2).sum(-1) / self.scale
        attn2 = attn2.masked_fill(~mask, float("-inf"))
        attn2 = F.softmax(attn2, dim=-1)

        context = (attn2.unsqueeze(-1) * v2).sum(1)
        return context


class NPALayer(nn.Module):
    """
    One NPA layer = `n_channels` parallel VQA modules
    """

    def __init__(
        self,
        embed_dim: int,
        n_channels: int = 4,
        codebook_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = nn.ModuleList(
            [VQAModule(embed_dim, codebook_size) for _ in range(n_channels)]
        )
        self.proj = nn.Linear(n_channels * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        item_embeds: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        contexts = [ch(item_embeds, mask) for ch in self.channels]
        cat = torch.cat(contexts, dim=-1)
        context = self.proj(cat)
        context = self.dropout(context)

        updated = item_embeds + context.unsqueeze(1)
        updated = self.norm(updated)

        updated = updated + self.ffn(updated)
        updated = self.norm2(updated)
        return updated, context


class NPA(nn.Module):
    """
    Neural Pattern Associator – full model.

    Architecture:
    • Item embedding table  (num_items → d)
    • N stacked NPA layers  (multi-channel VQA)
    • Prediction head: final context → logits over all items

    Two variants from the paper:
      SC (squashed-context): use only the *last layer's* context
      MC (multi-context): concatenate contexts from all layers
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        n_layers: int = 2,
        n_channels: int = 4,
        codebook_size: int = 64,
        dropout: float = 0.1,
        variant: str = "SC",          # "SC" or "MC"
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.variant = variant.upper()

        # +1 потому что 0 = padding
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.02)

        self.layers = nn.ModuleList(
            [
                NPALayer(embed_dim, n_channels, codebook_size, dropout)
                for _ in range(n_layers)
            ]
        )

        if self.variant == "MC":
            self.context_proj = nn.Linear(n_layers * embed_dim, embed_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_items + 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.item_emb(input_ids)

        all_contexts = []
        for layer in self.layers:
            x, ctx = layer(x, mask)
            all_contexts.append(ctx)

        if self.variant == "MC" and len(all_contexts) > 1:
            context = self.context_proj(torch.cat(all_contexts, dim=-1))
        else:
            context = all_contexts[-1]

        logits = self.pred_head(context)
        return logits


def ndcg_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 100,
) -> float:
    device = logits.device
    B = logits.size(0)
    logits[:, 0] = float("-inf")

    k = min(k, logits.size(-1))
    _, topk_idx = logits.topk(k, dim=-1)

    ndcgs = []
    for i in range(B):
        target_set = set(targets[i].tolist()) - {0}
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


def recall_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 100,
) -> float:
    logits[:, 0] = float("-inf")
    k = min(k, logits.size(-1))
    _, topk_idx = logits.topk(k, dim=-1)
    B = logits.size(0)
    recalls = []
    for i in range(B):
        target_set = set(targets[i].tolist()) - {0}
        if not target_set:
            continue
        hits = len(target_set & set(topk_idx[i].tolist()))
        recalls.append(hits / len(target_set))
    return np.mean(recalls) if recalls else 0.0


class NPATrainer:
    """
    Train and evaluate NPA.
    """

    def __init__(
        self,
        data: Dict[int, List[List[int]]],
        *,
        embed_dim: int = 128,
        n_layers: int = 2,
        n_channels: int = 4,
        codebook_size: int = 64,
        dropout: float = 0.1,
        variant: str = "SC",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 20,
        holdout_ratio: float = 0.2,     # доля корзин на eval
        target_ratio: float = 0.2,      # доля корзины, резервируемая на предикт
        eval_k: int = 100,
        max_basket_size: int = 128,
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
        self.target_ratio = target_ratio
        self.verbose = verbose
        self.num_workers = num_workers

        all_baskets: List[List[int]] = []
        for uid, blist in data.items():
            all_baskets.extend(blist)

        all_item_ids = set()
        for b in all_baskets:
            all_item_ids.update(b)
        self.item2idx: Dict[int, int] = {
            item: idx + 1 for idx, item in enumerate(sorted(all_item_ids))
        }
        self.idx2item: Dict[int, int] = {v: k for k, v in self.item2idx.items()}
        self.num_items = len(self.item2idx)

        mapped_baskets = [
            [self.item2idx[i] for i in b if i in self.item2idx]
            for b in all_baskets
        ]

        random.shuffle(mapped_baskets)
        n_eval = max(1, int(len(mapped_baskets) * holdout_ratio))
        eval_baskets = mapped_baskets[:n_eval]
        train_baskets = mapped_baskets[n_eval:]

        if self.verbose:
            print(f"Items:         {self.num_items}")
            print(f"Total baskets: {len(mapped_baskets)}")
            print(f"Train baskets: {len(train_baskets)}")
            print(f"Eval baskets:  {len(eval_baskets)}")
            print(f"Device:        {self.device}")

        self.train_ds = BasketDataset(train_baskets, self.num_items, max_basket_size)
        self.eval_baskets = [
            b for b in eval_baskets if len(set(b)) >= 3
        ]

        self.model = NPA(
            num_items=self.num_items,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_channels=n_channels,
            codebook_size=codebook_size,
            dropout=dropout,
            variant=variant,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        if self.verbose:
            print(f"Model params:  {total_params:,}")

        self.optimizer = torch.optim.AdamW(
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
            collate_fn=collate_baskets,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss, n_batches = 0.0, 0
            for input_ids, mask, targets in loader:
                input_ids = input_ids.to(self.device)
                mask = mask.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(input_ids, mask)
                loss = F.cross_entropy(logits, targets)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            self.scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)

            if self.verbose:
                metrics = self.evaluate(silent=True)
                print(
                    f"Epoch {epoch:3d}/{self.epochs} │ "
                    f"loss={avg_loss:.4f} │ "
                    f"NDCG@{self.eval_k}={metrics['ndcg']:.4f} │ "
                    f"Recall@{self.eval_k}={metrics['recall']:.4f}"
                )
        return self

    @torch.no_grad()
    def evaluate(self, silent: bool = False) -> dict:
        self.model.eval()

        all_inputs, all_targets_list = [], []
        for basket in self.eval_baskets:
            unique = list(dict.fromkeys(basket))
            n_target = max(1, int(len(unique) * self.target_ratio))
            inp = unique[: len(unique) - n_target]
            tgt = unique[len(unique) - n_target :]
            if len(inp) == 0:
                continue
            all_inputs.append(inp)
            all_targets_list.append(tgt)

        BS = self.batch_size
        ndcgs, recalls = [], []
        for start in range(0, len(all_inputs), BS):
            batch_inp = all_inputs[start : start + BS]
            batch_tgt = all_targets_list[start : start + BS]

            max_len = max(len(x) for x in batch_inp)
            max_tgt = max(len(t) for t in batch_tgt)

            padded = [x + [0] * (max_len - len(x)) for x in batch_inp]
            masks = [[True] * len(x) + [False] * (max_len - len(x)) for x in batch_inp]
            tgt_padded = [t + [0] * (max_tgt - len(t)) for t in batch_tgt]

            input_ids = torch.LongTensor(padded).to(self.device)
            mask = torch.BoolTensor(masks).to(self.device)
            targets = torch.LongTensor(tgt_padded).to(self.device)

            logits = self.model(input_ids, mask)

            for i, inp in enumerate(batch_inp):
                logits[i, inp] = float("-inf")

            ndcgs.append(ndcg_at_k(logits, targets, k=self.eval_k))
            recalls.append(recall_at_k(logits, targets, k=self.eval_k))

        result = {
            "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
        }

        if not silent and self.verbose:
            print(f"Eval  NDCG@{self.eval_k}  = {result['ndcg']:.4f}")
            print(f"Eval  Recall@{self.eval_k} = {result['recall']:.4f}")

        self.model.train()
        return result

    @torch.no_grad()
    def predict(
        self,
        basket: List[int],
        top_k: int = 20,
    ) -> List[Tuple[int, float]]:
        self.model.eval()
        mapped = [self.item2idx[i] for i in basket if i in self.item2idx]
        if not mapped:
            return []

        input_ids = torch.LongTensor([mapped]).to(self.device)
        mask = torch.ones(1, len(mapped), dtype=torch.bool, device=self.device)

        logits = self.model(input_ids, mask)[0]
        logits[0] = float("-inf")
        for idx in mapped:
            logits[idx] = float("-inf")

        scores = F.softmax(logits, dim=-1)
        k = min(top_k, scores.size(-1))
        topk_scores, topk_idx = scores.topk(k)

        results = []
        for sc, idx in zip(topk_scores.tolist(), topk_idx.tolist()):
            orig_id = self.idx2item.get(idx, idx)
            results.append((orig_id, sc))
        return results


    @torch.no_grad()
    def evaluate_external(
        self,
        data: Dict[int, List[List[int]]],
        target_ratio: float = None,
        top_k: int = None,
    ) -> Dict:
        if target_ratio is None:
            target_ratio = self.target_ratio
        if top_k is None:
            top_k = self.eval_k

        self.model.eval()

        all_inputs, all_targets_list, all_meta = [], [], []

        for user_id, baskets in data.items():
            for basket_idx, basket in enumerate(baskets):
                mapped = []
                orig = []
                seen = set()
                for item in basket:
                    if item in self.item2idx and item not in seen:
                        seen.add(item)
                        mapped.append(self.item2idx[item])
                        orig.append(item)

                if len(mapped) < 3:
                    continue

                n_target = max(1, int(len(mapped) * target_ratio))
                inp = mapped[: len(mapped) - n_target]
                tgt = mapped[len(mapped) - n_target :]
                inp_orig = orig[: len(orig) - n_target]
                tgt_orig = orig[len(orig) - n_target :]

                if len(inp) == 0:
                    continue

                all_inputs.append(inp)
                all_targets_list.append(tgt)
                all_meta.append({
                    "user_id": user_id,
                    "basket_idx": basket_idx,
                    "input_items_orig": inp_orig,
                    "target_items_orig": tgt_orig,
                })

        if not all_inputs:
            print("No valid baskets for evaluation.")
            return {"ndcg": 0.0, "recall": 0.0, "details": []}

        BS = self.batch_size
        details = []

        for start in range(0, len(all_inputs), BS):
            batch_inp = all_inputs[start: start + BS]
            batch_tgt = all_targets_list[start: start + BS]
            batch_meta = all_meta[start: start + BS]

            max_len = max(len(x) for x in batch_inp)
            max_tgt = max(len(t) for t in batch_tgt)

            padded = [x + [0] * (max_len - len(x)) for x in batch_inp]
            masks = [[True] * len(x) + [False] * (max_len - len(x)) for x in batch_inp]
            tgt_padded = [t + [0] * (max_tgt - len(t)) for t in batch_tgt]

            input_ids = torch.LongTensor(padded).to(self.device)
            mask = torch.BoolTensor(masks).to(self.device)
            targets = torch.LongTensor(tgt_padded).to(self.device)

            logits = self.model(input_ids, mask)

            for i, inp in enumerate(batch_inp):
                logits[i, inp] = float("-inf")
            logits[:, 0] = float("-inf")

            k_safe = min(top_k, logits.size(-1))
            _, topk_idx = logits.topk(k_safe, dim=-1)

            for i in range(len(batch_inp)):
                target_set = set(batch_tgt[i]) - {0}
                ranked = topk_idx[i].tolist()

                dcg = 0.0
                for rank, item in enumerate(ranked):
                    if item in target_set:
                        dcg += 1.0 / math.log2(rank + 2)
                n_rel = min(len(target_set), k_safe)
                idcg = sum(1.0 / math.log2(r + 2) for r in range(n_rel))
                ndcg_val = dcg / idcg if idcg > 0 else 0.0

                hits = len(target_set & set(ranked))
                recall_val = hits / len(target_set) if target_set else 0.0

                recs = [
                    self.idx2item.get(idx, idx)
                    for idx in ranked[:20]
                ]

                details.append({
                    **batch_meta[i],
                    "ndcg": ndcg_val,
                    "recall": recall_val,
                    "top_recs": recs,
                    "n_targets": len(target_set),
                    "n_hits": hits,
                })

        avg_ndcg = np.mean([d["ndcg"] for d in details])
        avg_recall = np.mean([d["recall"] for d in details])

        print(f"Baskets evaluated: {len(details)}")
        print(f"NDCG@{top_k}:       {avg_ndcg:.4f}")
        print(f"Recall@{top_k}:     {avg_recall:.4f}")

        return {
            "ndcg": float(avg_ndcg),
            "recall": float(avg_recall),
            "details": details,
        }

    @torch.no_grad()
    def predict_autoregressive(
        self,
        seed_items: Optional[List[int]] = None,
        n_steps: int = 20,
        strategy: str = "greedy",      # "greedy" | "top_k" | "nucleus"
        top_k_sample: int = 10,        # для strategy="top_k"
        top_p: float = 0.9,            # для strategy="nucleus"
        temperature: float = 1.0,
        return_scores: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Авторегрессионная генерация корзины.
        """
        self.model.eval()

        basket_idx = []
        if seed_items:
            for item in seed_items:
                if item in self.item2idx:
                    idx = self.item2idx[item]
                    if idx not in basket_idx:
                        basket_idx.append(idx)

        generated = []

        for step in range(n_steps):
            if len(basket_idx) == 0:
                zero_ctx = torch.zeros(1, self.model.embed_dim, device=self.device)
                logits = self.model.pred_head(zero_ctx)[0]
            else:
                input_ids = torch.LongTensor([basket_idx]).to(self.device)
                mask = torch.ones(1, len(basket_idx), dtype=torch.bool, device=self.device)
                logits = self.model(input_ids, mask)[0]

            logits[0] = float("-inf")
            for idx in basket_idx:
                logits[idx] = float("-inf")

            logits = logits / temperature

            if strategy == "greedy":
                next_idx = logits.argmax().item()
                score = F.softmax(logits, dim=-1)[next_idx].item()

            elif strategy == "top_k":
                top_vals, top_inds = logits.topk(top_k_sample)
                probs = F.softmax(top_vals, dim=-1)
                sampled = torch.multinomial(probs, 1).item()
                next_idx = top_inds[sampled].item()
                score = probs[sampled].item()

            elif strategy == "nucleus":
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = probs.sort(descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask_nucleus = cumsum - sorted_probs > top_p
                sorted_probs[mask_nucleus] = 0.0
                sorted_probs /= sorted_probs.sum()
                sampled = torch.multinomial(sorted_probs, 1).item()
                next_idx = sorted_indices[sampled].item()
                score = sorted_probs[sampled].item()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            basket_idx.append(next_idx)
            orig_id = self.idx2item.get(next_idx, next_idx)
            generated.append((orig_id, score))

        return generated


    @torch.no_grad()
    def evaluate_autoregressive(
        self,
        data: Dict[int, List[List[int]]],
        target_ratio: float = None,
        top_k: int = None,
        strategy: str = "greedy",
        top_k_sample: int = 10,
        top_p: float = 0.9,
        temperature: float = 1.0,
        n_runs: int = 1,
    ) -> Dict:
        """
        Авторегрессионная оценка на валидации.
        """
        if target_ratio is None:
            target_ratio = self.target_ratio
        if top_k is None:
            top_k = self.eval_k

        self.model.eval()
        details = []

        for user_id, baskets in data.items():
            for basket_idx, basket in enumerate(baskets):
                seen = set()
                unique = []
                for item in basket:
                    if item in self.item2idx and item not in seen:
                        seen.add(item)
                        unique.append(item)

                if len(unique) < 3:
                    continue

                n_target = max(1, int(len(unique) * target_ratio))
                seed_items = unique[: len(unique) - n_target]
                target_items = unique[len(unique) - n_target:]

                if not seed_items or not target_items:
                    continue

                target_set = set(target_items)
                n_steps = min(top_k, self.num_items - len(seed_items))

                if n_runs == 1:
                    generated = self.predict_autoregressive(
                        seed_items=seed_items,
                        n_steps=n_steps,
                        strategy=strategy,
                        top_k_sample=top_k_sample,
                        top_p=top_p,
                        temperature=temperature,
                    )
                    ranked = [item_id for item_id, _ in generated]

                else:
                    from collections import Counter
                    item_scores = Counter()
                    for _ in range(n_runs):
                        generated = self.predict_autoregressive(
                            seed_items=seed_items,
                            n_steps=n_steps,
                            strategy=strategy,
                            top_k_sample=top_k_sample,
                            top_p=top_p,
                            temperature=temperature,
                        )
                        for rank, (item_id, score) in enumerate(generated):
                            item_scores[item_id] += 1.0 / (rank + 1)

                    ranked = [
                        item_id
                        for item_id, _ in item_scores.most_common(top_k)
                    ]

                ranked_topk = ranked[:top_k]

                dcg = 0.0
                for rank, item_id in enumerate(ranked_topk):
                    if item_id in target_set:
                        dcg += 1.0 / math.log2(rank + 2)
                n_rel = min(len(target_set), top_k)
                idcg = sum(1.0 / math.log2(r + 2) for r in range(n_rel))
                ndcg_val = dcg / idcg if idcg > 0 else 0.0

                hits = len(target_set & set(ranked_topk))
                recall_val = hits / len(target_set)

                hit_rate = 1.0 if hits > 0 else 0.0

                details.append({
                    "user_id": user_id,
                    "basket_idx": basket_idx,
                    "seed_items": seed_items,
                    "target_items": target_items,
                    "top_recs": ranked_topk[:20],
                    "ndcg": ndcg_val,
                    "recall": recall_val,
                    "hit_rate": hit_rate,
                    "n_targets": len(target_set),
                    "n_hits": hits,
                })

        if not details:
            print("No valid baskets for evaluation.")
            return {"ndcg": 0.0, "recall": 0.0, "hit_rate": 0.0, "details": []}

        avg_ndcg = np.mean([d["ndcg"] for d in details])
        avg_recall = np.mean([d["recall"] for d in details])
        avg_hr = np.mean([d["hit_rate"] for d in details])

        print(f"Strategy:         {strategy}" + (f" (x{n_runs} runs)" if n_runs > 1 else ""))
        print(f"Baskets evaluated: {len(details)}")
        print(f"NDCG@{top_k}:       {avg_ndcg:.4f}")
        print(f"Recall@{top_k}:     {avg_recall:.4f}")
        print(f"HitRate@{top_k}:    {avg_hr:.4f}")

        return {
            "ndcg": float(avg_ndcg),
            "recall": float(avg_recall),
            "hit_rate": float(avg_hr),
            "details": details,
        }


