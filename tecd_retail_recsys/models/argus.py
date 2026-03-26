from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



@dataclass
class FullARGUSConfig:
    """Все гиперпараметры модели и обучения."""

    # ---- каталог (заполняется автоматически) ----
    num_items: int = 0
    num_brands: int = 0
    num_categories: int = 0
    num_subcategories: int = 0
    num_subdomains: int = 0
    num_action_types: int = 0
    num_os: int = 0
    num_socdem_clusters: int = 22
    num_regions: int = 91
    num_hours: int = 24
    num_dows: int = 7
    num_price_buckets: int = 100
    num_time_delta_buckets: int = 64

    # ---- размерности предобученных эмбеддингов ----
    item_emb_dim: int = 128
    brand_emb_dim: int = 64

    # ---- архитектура ----
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_interactions: int = 150   # макс кол-во взаимодействий в истории

    # ---- лосс ----
    n_uniform_negatives: int = 2048
    use_in_batch_negatives: bool = True
    temperature: float = 0.05
    feedback_loss_weight: float = 0.3
    label_smoothing: float = 0.0

    # ---- обучение ----
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    epochs: int = 10
    batch_size: int = 128
    eval_batch_size: int = 256
    grad_clip: float = 1.0

    # ---- eval ----
    top_k: int = 100

    # ---- устройство ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


PAD_ID = 0  # индекс для паддинга


class Vocabulary:
    def __init__(self):
        self.val2idx: Dict[Any, int] = {}
        self.idx2val: Dict[int, Any] = {}
        self._next = 1  # 0 = PAD

    def fit(self, values):
        for v in values:
            if v not in self.val2idx:
                self.val2idx[v] = self._next
                self.idx2val[self._next] = v
                self._next += 1
        return self

    def transform(self, value) -> int:
        return self.val2idx.get(value, 0)

    def __len__(self):
        return self._next


def bucketize_price(price: float, n_buckets: int = 100,
                    min_val: float = -10.0, max_val: float = 10.0) -> int:
    bucket = int((price - min_val) / (max_val - min_val) * (n_buckets - 1)) + 1
    return max(1, min(n_buckets, bucket))


def bucketize_time_delta(delta_seconds: float, n_buckets: int = 64) -> int:
    """
    Log-scale bucketization времени с прошлого взаимодействия.
    Бакет 0 = PAD, 1 = 0 секунд, 2..n_buckets = лог-шкала.
    """
    if delta_seconds <= 0:
        return 1
    # log scale: от 1 сек до ~1 года (3e7 сек)
    log_val = math.log1p(delta_seconds)
    max_log = math.log1p(3e7)
    bucket = int(log_val / max_log * (n_buckets - 2)) + 2
    return max(1, min(n_buckets, bucket))


from datetime import datetime

class ArgusDataPreprocessor:
    """
    Ожидаемый формат входных данных — список словарей:
    [
        {
            'timestamp': 1700000000,
            'user_id': 42,
            'item_id': 'SKU_123',
            'subdomain': 'search',
            'action_type': 'add_to_cart',
            'os': 'android',
            'item_brand_id': 'BRAND_5',
            'item_category': 'Electronics',
            'item_subcategory': 'Phones',
            'item_price': 3.5,
            'item_embedding': np.array([...]),    # (item_emb_dim,)
            'brand_embedding': np.array([...]),   # (brand_emb_dim,)
            'user_socdem_cluster': 5,
            'user_region': 77,
        },
        ...
    ]
    """

    def __init__(self, cfg: FullARGUSConfig):
        self.cfg = cfg
        self.vocab_item = Vocabulary()
        self.vocab_brand = Vocabulary()
        self.vocab_category = Vocabulary()
        self.vocab_subcategory = Vocabulary()
        self.vocab_subdomain = Vocabulary()
        self.vocab_action = Vocabulary()
        self.vocab_os = Vocabulary()
        self._fitted = False

    def fit(self, records: List[Dict]):
        self.vocab_item.fit([r['item_id'] for r in records])
        self.vocab_brand.fit([r['item_brand_id'] for r in records])
        self.vocab_category.fit([r['item_category'] for r in records])
        self.vocab_subcategory.fit([r['item_subcategory'] for r in records])
        self.vocab_subdomain.fit([r['subdomain'] for r in records])
        self.vocab_action.fit([r['action_type'] for r in records])
        self.vocab_os.fit([r['os'] for r in records])

        self.cfg.num_items = len(self.vocab_item) - 1       # без PAD
        self.cfg.num_brands = len(self.vocab_brand) - 1
        self.cfg.num_categories = len(self.vocab_category) - 1
        self.cfg.num_subcategories = len(self.vocab_subcategory) - 1
        self.cfg.num_subdomains = len(self.vocab_subdomain) - 1
        self.cfg.num_action_types = len(self.vocab_action) - 1
        self.cfg.num_os = len(self.vocab_os) - 1

        self._fitted = True
        return self

    def transform(self, records: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Группирует записи по user_id, сортирует по timestamp,
        вычисляет time_delta, кодирует фичи.
        """
        assert self._fitted, "Сначала вызовите .fit()"

        # Группировка по user_id
        user_records: Dict[int, List[Dict]] = {}
        for r in records:
            uid = r['user_id']
            if uid not in user_records:
                user_records[uid] = []
            user_records[uid].append(r)

        user_sequences: Dict[int, List[Dict]] = {}
        for uid, recs in user_records.items():
            recs = sorted(recs, key=lambda x: x['timestamp'])

            encoded_seq = []
            prev_ts = None
            for r in recs:
                ts = r['timestamp']
                dt = datetime.utcfromtimestamp(ts)
                delta = (ts - prev_ts) if prev_ts is not None else 0
                prev_ts = ts

                encoded = {
                    # ---- item features ----
                    'item_id':         self.vocab_item.transform(r['item_id']),
                    'brand_id':        self.vocab_brand.transform(r['item_brand_id']),
                    'category_id':     self.vocab_category.transform(r['item_category']),
                    'subcategory_id':  self.vocab_subcategory.transform(r['item_subcategory']),
                    'price_bucket':    bucketize_price(
                        r['item_price'], self.cfg.num_price_buckets
                    ),
                    # 'item_emb':        np.array(r['item_embedding'], dtype=np.float32),
                    # 'brand_emb':       np.array(r['brand_embedding'], dtype=np.float32),

                    # ---- context features ----
                    'socdem_cluster':  r['user_socdem_cluster'] + 1,  # +1 для PAD=0
                    'region':          r['user_region'] + 1,
                    'os_id':           self.vocab_os.transform(r['os']),
                    'subdomain_id':    self.vocab_subdomain.transform(r['subdomain']),
                    'hour':            dt.hour + 1,      # 1..24 (0=PAD)
                    'dow':             dt.weekday() + 1,  # 1..7  (0=PAD)
                    'time_delta_bucket': bucketize_time_delta(
                        delta, self.cfg.num_time_delta_buckets
                    ),

                    # ---- feedback feature ----
                    'action_type':     self.vocab_action.transform(r['action_type']),
                }
                encoded_seq.append(encoded)

            if len(encoded_seq) >= 1:
                user_sequences[uid] = encoded_seq

        return user_sequences


CAT_CONTEXT_KEYS = [
    'socdem_cluster', 'region', 'os_id', 'subdomain_id',
    'hour', 'dow', 'time_delta_bucket',
]
CAT_ITEM_KEYS = [
    'item_id', 'brand_id', 'category_id', 'subcategory_id', 'price_bucket',
]
# DENSE_ITEM_KEYS = ['item_emb', 'brand_emb']  # пока отказался от эмбеддингов
DENSE_ITEM_KEYS = []

FEEDBACK_KEY = 'action_type'


class ARGUSDataset(Dataset):
    def __init__(
        self,
        user_sequences: Dict[int, List[Dict]],
        cfg: FullARGUSConfig,
        is_train: bool = True,
    ):
        self.cfg = cfg
        self.max_len = cfg.max_interactions
        self.is_train = is_train
        self.item_emb_dim = cfg.item_emb_dim
        self.brand_emb_dim = cfg.brand_emb_dim

        self.user_ids: List[int] = []
        self.sequences: List[List[Dict]] = []

        for uid, seq in user_sequences.items():
            if len(seq) < 2:
                continue
            if is_train and len(seq) > 10:
                # Sliding window: шаг = половина max_len
                step = max(self.max_len // 2, 5)
                for start in range(0, max(len(seq) - 5, 1), step):
                    window = seq[start:start + self.max_len]
                    if len(window) >= 2:
                        self.user_ids.append(uid)
                        self.sequences.append(window)
            else:
                if len(seq) > self.max_len:
                    seq = seq[-self.max_len:]
                self.user_ids.append(uid)
                self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        seq_len = len(seq)
        pad_len = self.max_len - seq_len

        result: Dict[str, Any] = {}
        all_cat_keys = CAT_CONTEXT_KEYS + CAT_ITEM_KEYS + [FEEDBACK_KEY]

        for key in all_cat_keys:
            arr = np.zeros(self.max_len, dtype=np.int64)
            for i, inter in enumerate(seq):
                arr[pad_len + i] = inter[key]
            result[key] = torch.from_numpy(arr)

        # Dense features - пока убрал
        # item_emb_arr = np.zeros((self.max_len, self.item_emb_dim), dtype=np.float32)
        # brand_emb_arr = np.zeros((self.max_len, self.brand_emb_dim), dtype=np.float32)
        # for i, inter in enumerate(seq):
        #     item_emb_arr[pad_len + i] = inter['item_emb']
        #     brand_emb_arr[pad_len + i] = inter['brand_emb']
        # result['item_emb_dense'] = torch.from_numpy(item_emb_arr)
        # result['brand_emb_dense'] = torch.from_numpy(brand_emb_arr)

        # Padding mask: True = valid interaction
        mask = np.zeros(self.max_len, dtype=np.bool_)
        mask[pad_len:] = True
        result['padding_mask'] = torch.from_numpy(mask)

        return result


class PreNormBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=False)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        cat_features: Dict[str, int],
        dense_features: Dict[str, int],
        d_model: int,
    ):
        super().__init__()
        self.cat_embeddings = nn.ModuleDict()
        for name, vocab_size in cat_features.items():
            self.cat_embeddings[name] = nn.Embedding(
                vocab_size + 1, d_model, padding_idx=PAD_ID
            )

        self.dense_projections = nn.ModuleDict()
        for name, input_dim in dense_features.items():
            self.dense_projections[name] = nn.Linear(input_dim, d_model, bias=False)

        self.n_features = len(cat_features) + len(dense_features)
        self.scale = 1.0 / math.sqrt(self.n_features) if self.n_features > 0 else 1.0

    def forward(self, cat_inputs: Dict[str, torch.Tensor],
                dense_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for name, emb_layer in self.cat_embeddings.items():
            parts.append(emb_layer(cat_inputs[name]))

        for name, proj_layer in self.dense_projections.items():
            parts.append(proj_layer(dense_inputs[name]))

        if not parts:
            raise ValueError("No features provided")

        out = torch.stack(parts, dim=0).sum(dim=0)
        return out * self.scale


class FullARGUS(nn.Module):
    def __init__(self, cfg: FullARGUSConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.context_encoder = FeatureEncoder(
            cat_features={
                'socdem_cluster': cfg.num_socdem_clusters + 1,
                'region':         cfg.num_regions + 1,
                'os_id':          cfg.num_os,
                'subdomain_id':   cfg.num_subdomains,
                'hour':           cfg.num_hours + 1,
                'dow':            cfg.num_dows + 1,
                'time_delta_bucket': cfg.num_time_delta_buckets + 1,
            },
            dense_features={},
            d_model=d,
        )

        self.item_encoder = FeatureEncoder(
            cat_features={
                'item_id':        cfg.num_items,
                'brand_id':       cfg.num_brands,
                'category_id':    cfg.num_categories,
                'subcategory_id': cfg.num_subcategories,
                'price_bucket':   cfg.num_price_buckets + 1,
            }, 
            dense_features={
            #     'item_emb':  cfg.item_emb_dim,
            #     'brand_emb': cfg.brand_emb_dim,
            },
            d_model=d,
        )

        self.feedback_encoder = FeatureEncoder(
            cat_features={
                'action_type': cfg.num_action_types,
            },
            dense_features={},
            d_model=d,
        )

        self.pos_emb = nn.Embedding(cfg.max_interactions, d)
        # segment: 0=context, 1=item, 2=feedback
        self.seg_emb = nn.Embedding(3, d)

        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.emb_ln = nn.LayerNorm(d)

        self.blocks = nn.ModuleList([
            PreNormBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.final_ln = nn.LayerNorm(d)

        self.item_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d, bias=False),
        )

        self.item_score_emb = nn.Embedding(
            cfg.num_items + 1, d, padding_idx=PAD_ID
        )

        self.feedback_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, cfg.num_action_types + 1),  # +1 для PAD
        )

        max_tokens = 3 * cfg.max_interactions
        causal = torch.triu(
            torch.ones(max_tokens, max_tokens, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_interleaved_sequence(
        self, batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = batch['item_id'].shape[0]
        L = self.cfg.max_interactions
        d = self.cfg.d_model
        device = batch['item_id'].device

        ctx = self.context_encoder(
            cat_inputs={k: batch[k] for k in CAT_CONTEXT_KEYS},
            dense_inputs={},
        )

        item = self.item_encoder(
            cat_inputs={k: batch[k] for k in CAT_ITEM_KEYS},
            dense_inputs={
                # 'item_emb': batch['item_emb_dense'],
                # 'brand_emb': batch['brand_emb_dense'],
            },
        )

        fb = self.feedback_encoder(
            cat_inputs={'action_type': batch['action_type']},
            dense_inputs={},
        )

        tokens = torch.stack([ctx, item, fb], dim=2)
        tokens = tokens.reshape(B, 3 * L, d)

        interaction_pos = torch.arange(L, device=device).repeat_interleave(3)
        pos_e = self.pos_emb(interaction_pos).unsqueeze(0)

        seg_ids = torch.tensor([0, 1, 2], device=device).repeat(L)
        seg_e = self.seg_emb(seg_ids).unsqueeze(0)

        tokens = tokens + pos_e + seg_e
        tokens = self.emb_ln(self.emb_dropout(tokens))

        pad_mask = batch['padding_mask']
        token_pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(B, 3 * L)

        return tokens, token_pad_mask

    def forward(
        self, batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        tokens, token_pad_mask = self._build_interleaved_sequence(batch)
        B, T, d = tokens.shape

        mask = self.causal_mask[:T, :T]

        for block in self.blocks:
            tokens = block(tokens, attn_mask=mask)

        tokens = self.final_ln(tokens)
        return tokens

    def compute_loss(
        self, batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        total loss = item_loss + α * feedback_loss.
        """
        cfg = self.cfg
        hidden = self.forward(batch)
        B, T, d = hidden.shape
        L = cfg.max_interactions
        device = hidden.device

        pad_mask = batch['padding_mask']

        ctx_indices = torch.arange(0, T, 3, device=device)
        ctx_hidden = hidden[:, ctx_indices, :]
        ctx_hidden = self.item_head(ctx_hidden)

        target_item_ids = batch['item_id']

        # item_loss = self._sampled_softmax_loss(
        #     ctx_hidden, target_item_ids, pad_mask
        # )
        item_loss = self._full_softmax_loss(ctx_hidden, target_item_ids, pad_mask)

        item_indices = torch.arange(1, T, 3, device=device)
        item_hidden = hidden[:, item_indices, :]
        fb_logits = self.feedback_head(item_hidden)

        target_actions = batch['action_type']
        fb_logits_flat = fb_logits.reshape(-1, fb_logits.size(-1))
        target_flat = target_actions.reshape(-1)
        pad_flat = pad_mask.reshape(-1).float()

        fb_loss = F.cross_entropy(fb_logits_flat, target_flat, reduction='none', label_smoothing=self.cfg.label_smoothing)
        fb_loss = (fb_loss * pad_flat).sum() / pad_flat.sum().clamp(min=1)

        total_loss = item_loss + cfg.feedback_loss_weight * fb_loss

        metrics = {
            'item_loss': item_loss.item(),
            'feedback_loss': fb_loss.item(),
            'total_loss': total_loss.item(),
        }
        return total_loss, metrics

    def _sampled_softmax_loss(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, d = hidden.shape
        cfg = self.cfg
        device = hidden.device

        pos_emb = self.item_score_emb(target_ids)
        pos_logits = (hidden * pos_emb).sum(-1) / cfg.temperature

        n_neg = min(cfg.n_uniform_negatives, cfg.num_items)
        neg_ids = torch.randint(1, cfg.num_items + 1, (n_neg,), device=device)
        neg_emb = self.item_score_emb(neg_ids)
        neg_logits = torch.matmul(hidden, neg_emb.T) / cfg.temperature

        if cfg.use_in_batch_negatives:
            flat_targets = target_ids[pad_mask].unique()
            if flat_targets.numel() > 0:
                ib_emb = self.item_score_emb(flat_targets)
                ib_logits = torch.matmul(hidden, ib_emb.T) / cfg.temperature
                neg_logits = torch.cat([neg_logits, ib_logits], dim=-1)

        logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
        labels = torch.zeros(B, L, dtype=torch.long, device=device)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction='none',
        )
        pad_flat = pad_mask.reshape(-1).float()
        return (loss * pad_flat).sum() / pad_flat.sum().clamp(min=1)

    def _full_softmax_loss(
        self, hidden, target_ids, pad_mask,
    ):
        logits = torch.matmul(hidden, self.item_score_emb.weight.T)
        logits = logits / self.cfg.temperature
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction='none',
            ignore_index=PAD_ID,
            label_smoothing=self.cfg.label_smoothing,
        )
        pad_flat = pad_mask.reshape(-1).float()
        return (loss * pad_flat).sum() / pad_flat.sum().clamp(min=1)


    @torch.no_grad()
    def score_all_items(
        self,
        batch: Dict[str, torch.Tensor],
        query_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if query_context is not None:
            scores = self._score_with_query(batch, query_context)
        else:
            scores = self._score_last_context(batch)

        scores[:, PAD_ID] = -float('inf')
        return scores

    def _score_last_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.forward(batch)
        B = hidden.shape[0]
        L = self.cfg.max_interactions

        last_fb_hidden = hidden[:, -1, :]
        last_fb_hidden = self.item_head(last_fb_hidden)

        scores = last_fb_hidden @ self.item_score_emb.weight.T
        return scores

    def _score_with_query(
        self,
        batch: Dict[str, torch.Tensor],
        query_context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        tokens, token_pad_mask = self._build_interleaved_sequence(batch)
        B, T, d = tokens.shape
        device = tokens.device
        L = self.cfg.max_interactions

        qc = self.context_encoder(
            cat_inputs={k: query_context[k].unsqueeze(1) for k in CAT_CONTEXT_KEYS},
            dense_inputs={},
        )

        pos_e = self.pos_emb(
            torch.tensor([L - 1], device=device).clamp(max=L - 1)
        ).unsqueeze(0)
        seg_e = self.seg_emb(
            torch.tensor([0], device=device)
        ).unsqueeze(0)

        qc = qc + pos_e + seg_e
        qc = self.emb_ln(self.emb_dropout(qc))

        tokens = torch.cat([tokens, qc], dim=1)

        T_new = T + 1
        mask = torch.triu(
            torch.ones(T_new, T_new, dtype=torch.bool, device=device),
            diagonal=1,
        )

        for block in self.blocks:
            tokens = block(tokens, attn_mask=mask)

        tokens = self.final_ln(tokens)
        query_hidden = tokens[:, -1, :]
        query_hidden = self.item_head(query_hidden)

        scores = query_hidden @ self.item_score_emb.weight.T
        scores[:, PAD_ID] = -float('inf')
        return scores


def ndcg_at_k(
    scores: torch.Tensor,
    targets: List[List[int]],
    k: int = 100,
) -> float:
    B = scores.shape[0]
    ndcgs = []

    for i in range(B):
        gt = targets[i]
        if not gt:
            continue
        relevance = Counter(gt)

        actual_k = min(k, scores[i].shape[0])
        _, topk_ids = scores[i].topk(actual_k)
        topk_ids = topk_ids.cpu().numpy()

        dcg = 0.0
        for rank, item_id in enumerate(topk_ids):
            rel = relevance.get(int(item_id), 0)
            dcg += rel / math.log2(rank + 2)

        ideal_rels = sorted(relevance.values(), reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels[:k]):
            idcg += rel / math.log2(rank + 2)

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def train_val_split(
    user_sequences: Dict[int, List[Dict]],
    n_val_items: int = 1,
) -> Tuple[Dict[int, List[Dict]], Dict[int, List[Dict]]]:
    """
    Leave-last-N-out split по interaction-ам.
    """
    train_data, val_data = {}, {}
    for uid, seq in user_sequences.items():
        if len(seq) < n_val_items + 2:
            train_data[uid] = seq
            continue
        train_data[uid] = seq[:-n_val_items]
        val_data[uid] = seq
    return train_data, val_data


def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = {}
    for key in samples[0]:
        batch[key] = torch.stack([s[key] for s in samples], dim=0)
    return batch


class FullARGUSTrainer:
    def __init__(
        self,
        cfg: FullARGUSConfig,
        preprocessor: ArgusDataPreprocessor,
        train_sequences: Dict[int, List[Dict]],
        val_sequences: Optional[Dict[int, List[Dict]]] = None,
    ):
        self.cfg = cfg
        self.preprocessor = preprocessor

        print(f"[ARGUS] Каталог: {cfg.num_items} items, "
              f"{cfg.num_brands} brands, "
              f"{cfg.num_categories} categories, "
              f"{cfg.num_action_types} action types")

        self.train_ds = ARGUSDataset(train_sequences, cfg, is_train=True)
        self.val_ds = (
            ARGUSDataset(val_sequences, cfg, is_train=False)
            if val_sequences else None
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        self.model = FullARGUS(cfg).to(cfg.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[ARGUS] Параметров модели: {total_params:,}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        total_steps = len(self.train_loader) * cfg.epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.lr,
            total_steps=max(total_steps, 1),
            pct_start=min(cfg.warmup_steps / max(total_steps, 1), 0.1),
            anneal_strategy="cos",
        )

        self.val_sequences = val_sequences

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.cfg.device) for k, v in batch.items()}

    def train(self):
        cfg = self.cfg
        best_ndcg = -1.0

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            epoch_metrics = {'item_loss': 0, 'feedback_loss': 0, 'total_loss': 0}
            n_batches = 0

            for batch in tqdm(self.train_loader):
                batch = self._to_device(batch)

                loss, metrics = self.model.compute_loss(batch)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

                self.optimizer.step()
                self.scheduler.step()

                for k, v in metrics.items():
                    epoch_metrics[k] += v
                n_batches += 1

            for k in epoch_metrics:
                epoch_metrics[k] /= max(n_batches, 1)
            lr_now = self.scheduler.get_last_lr()[0]

            if self.val_ds is not None and len(self.val_ds) > 0:
                val_ndcg = self._evaluate()
                improved = " ★" if val_ndcg > best_ndcg else ""
                if val_ndcg > best_ndcg:
                    best_ndcg = val_ndcg
                print(
                    f"Epoch {epoch}/{cfg.epochs} | "
                    f"item_loss={epoch_metrics['item_loss']:.4f} | "
                    f"fb_loss={epoch_metrics['feedback_loss']:.4f} | "
                    f"lr={lr_now:.2e} | "
                    f"NDCG@{cfg.top_k}={val_ndcg:.4f}{improved}"
                )
            else:
                print(
                    f"Epoch {epoch}/{cfg.epochs} | "
                    f"item_loss={epoch_metrics['item_loss']:.4f} | "
                    f"fb_loss={epoch_metrics['feedback_loss']:.4f} | "
                    f"lr={lr_now:.2e}"
                )

        print(f"\n[ARGUS] Обучение завершено. Best NDCG@{cfg.top_k} = {best_ndcg:.4f}")

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        cfg = self.cfg

        loader = DataLoader(
            self.val_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        all_ndcgs = []
        for batch in loader:
            batch_dev = self._to_device(batch)
            B = batch['item_id'].shape[0]

            targets_list = []
            for b in range(B):
                mask = batch['padding_mask'][b]
                valid_items = batch['item_id'][b][mask].numpy().tolist()
                targets_list.append(valid_items[-1:] if valid_items else [])

            scores = self.model.score_all_items(batch_dev)
            batch_ndcg = ndcg_at_k(scores, targets_list, k=cfg.top_k)
            all_ndcgs.append(batch_ndcg)

        return float(np.mean(all_ndcgs)) if all_ndcgs else 0.0

    @torch.no_grad()
    def recommend(
        self,
        user_sequences: Dict[int, List[Dict]],
        top_k: int = 100,
    ) -> Dict[int, List[Tuple[int, float]]]:
        self.model.eval()
        cfg = self.cfg

        temp_ds = ARGUSDataset(user_sequences, cfg, is_train=False)
        temp_loader = DataLoader(
            temp_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        uid_list = temp_ds.user_ids

        results: Dict[int, List[Tuple[int, float]]] = {}
        ptr = 0

        for batch in temp_loader:
            batch = self._to_device(batch)
            B = batch['item_id'].shape[0]

            scores = self.model.score_all_items(batch)
            actual_k = min(top_k, scores.shape[-1])
            top_scores, top_indices = scores.topk(actual_k, dim=-1)

            top_scores = top_scores.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

            for i in range(B):
                uid = uid_list[ptr + i]
                recs = []
                for rank in range(actual_k):
                    encoded_id = int(top_indices[i, rank])
                    score = float(top_scores[i, rank])
                    original_id = self.preprocessor.vocab_item.idx2val.get(
                        encoded_id, encoded_id
                    )
                    recs.append((original_id, score))
                results[uid] = recs

            ptr += B

        return results

    @torch.no_grad()
    def recommend_simple(
        self,
        user_sequences: Dict[int, List[Dict]],
        top_k: int = 100,
    ) -> Dict[int, List]:
        """
        Упрощенный инференс: возвращает только item_id без скоров.
        """
        full_recs = self.recommend(user_sequences, top_k)
        return {uid: [item for item, _ in recs] for uid, recs in full_recs.items()}
