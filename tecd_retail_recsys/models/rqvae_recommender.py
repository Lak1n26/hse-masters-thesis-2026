"""
RQ-VAE Recommender System
Based on "Recommender Systems with Generative Retrieval" (Rajput et al., 2023)
https://arxiv.org/abs/2305.05065
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster and more stable than LayerNorm"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with multiple forward modes:
    - EMA: Exponential Moving Average updates (default, stable)
    - Gumbel-Softmax: Differentiable sampling (better gradient flow)
    - STE: Straight-Through Estimator (simple)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_gumbel: bool = True
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_gumbel = use_gumbel
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding.weight)
        
        # EMA buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        
    def forward(
        self, 
        inputs: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [batch_size, embedding_dim]
            temperature: for Gumbel-Softmax sampling
        Returns:
            quantized: [batch_size, embedding_dim]
            loss: quantization loss
            encoding_indices: [batch_size]
        """
        # Calculate distances
        distances = (
            torch.sum(inputs**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(inputs, self.embedding.weight.t())
        )
        
        # Get nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        
        if self.training and self.use_gumbel and temperature > 0:
            # Gumbel-Softmax for differentiable sampling
            logits = -distances / max(temperature, 1e-3)
            soft_one_hot = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
            quantized = torch.matmul(soft_one_hot, self.embedding.weight)
        else:
            # Hard quantization (STE)
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            quantized = torch.matmul(encodings, self.embedding.weight)
        
        if self.training:
            # EMA update for codebook
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            batch_cluster_size = encodings.sum(0)
            batch_sum = encodings.t() @ inputs.detach()
            
            self.ema_cluster_size.mul_(self.decay).add_(batch_cluster_size, alpha=1 - self.decay)
            self.ema_w.mul_(self.decay).add_(batch_sum, alpha=1 - self.decay)
            
            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            updated_embedding = self.ema_w / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(updated_embedding)
            
            # Commitment loss
            loss = self.commitment_cost * F.mse_loss(inputs, quantized.detach())
        else:
            e_latent_loss = F.mse_loss(inputs, quantized.detach())
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices


class RQVAE(nn.Module):
    """
    Residual Quantization Variational Autoencoder
    Maps item embeddings to hierarchical semantic IDs
    Enhanced with Gumbel-Softmax for better gradient flow
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_quantizers: int,
        codebook_size: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        use_gumbel: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.use_gumbel = use_gumbel
        
        # Normalization buffers
        self.register_buffer('emb_mean', torch.zeros(input_dim))
        self.register_buffer('emb_std', torch.ones(input_dim))
        
        # Encoder with RMSNorm
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                RMSNorm(hidden_dim),  # RMSNorm instead of LayerNorm
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Residual Quantizers with Gumbel-Softmax support
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                codebook_size, 
                hidden_dims[-1], 
                commitment_cost, 
                ema_decay,
                use_gumbel=use_gumbel
            )
            for _ in range(num_quantizers)
        ])
        
        # Decoder with RMSNorm
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i-1]),
                nn.ReLU(),
                RMSNorm(hidden_dims[i-1]),  # RMSNorm instead of LayerNorm
            ])
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def set_normalization(self, embeddings: torch.Tensor):
        """Compute and store mean/std from the full item embedding matrix"""
        mean = embeddings.mean(dim=0)
        std = embeddings.std(dim=0).clamp(min=1e-6)
        self.emb_mean.copy_(mean)
        self.emb_std.copy_(std)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.emb_mean) / self.emb_std
        
    def encode(
        self, 
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Encode input to semantic IDs using residual quantization with Gumbel-Softmax
        
        Args:
            x: [batch_size, input_dim] — already normalized
            temperature: Gumbel-Softmax temperature (lower = more discrete)
        Returns:
            quantized_sum: [batch_size, hidden_dim]
            semantic_ids: list of [batch_size] for each quantizer level
            vq_loss: quantization loss
        """
        z = self.encoder(x)
        
        residual = z
        quantized_sum = torch.zeros_like(z)
        semantic_ids = []
        vq_loss = 0.0
        
        for quantizer in self.quantizers:
            quantized_st, loss, indices = quantizer(residual, temperature=temperature)
            q_true = quantizer.embedding(indices)
            quantized_sum = quantized_sum + quantized_st  # grad flows via ST/Gumbel
            residual = residual - q_true.detach()          # true residual (no ST noise)
            semantic_ids.append(indices)
            vq_loss = vq_loss + loss
            
        return quantized_sum, semantic_ids, vq_loss
    
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        return self.decoder(quantized)
    
    def forward(
        self, 
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Full forward pass with Gumbel-Softmax temperature control
        """
        x_norm = self.normalize(x)
        quantized, semantic_ids, vq_loss = self.encode(x_norm, temperature=temperature)
        reconstructed = self.decode(quantized)
        recon_loss = F.mse_loss(reconstructed, x_norm)
        
        return reconstructed, semantic_ids, recon_loss, vq_loss
    
    def get_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get semantic IDs for items
        
        Args:
            x: [batch_size, input_dim]
        Returns:
            semantic_ids: [batch_size, num_quantizers]
        """
        with torch.no_grad():
            x_norm = self.normalize(x)
            _, semantic_ids, _ = self.encode(x_norm, temperature=0.0)  # Deterministic
            return torch.stack(semantic_ids, dim=1)


class TransformerEncoderDecoderModel(nn.Module):
    """
    Encoder-Decoder Transformer with User Embeddings and Token Flattening.
    
    Architecture improvements from official implementation:
    1. Separate Encoder (for context) and Decoder (for generation)
    2. User ID embeddings for personalization
    3. RMSNorm for stable training
    4. Separate input projections for encoder/decoder
    5. BOS (beginning-of-sequence) token
    
    The token flattening strategy remains: each item becomes Q consecutive tokens
    (one per quantizer level), enabling the model to condition each level on 
    previous levels within the same item.
    """
    
    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 200,
        num_users: int = 50000,  # Max number of users
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        flat_max = max_seq_length * num_quantizers
        
        # ===== Embeddings =====
        # One embedding table per quantizer level
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model, padding_idx=codebook_size)
            for _ in range(num_quantizers)
        ])
        
        # Level embedding (which quantizer level this token is)
        self.level_embedding = nn.Embedding(num_quantizers, d_model)
        
        # User ID embedding for personalization (NEW!)
        self.user_embedding = nn.Embedding(num_users, d_model)
        
        # BOS token for decoder start (NEW!)
        self.bos_emb = nn.Parameter(torch.randn(d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, flat_max + 1)  # +1 for user token
        
        # ===== Normalization (RMSNorm instead of LayerNorm) =====
        self.norm_encoder = RMSNorm(d_model)
        self.norm_decoder = RMSNorm(d_model)
        
        # ===== Dropout =====
        self.dropout = nn.Dropout(p=dropout)
        
        # ===== Separate Input Projections (from official) =====
        self.in_proj_context = nn.Linear(d_model, d_model, bias=False)  # For encoder
        self.in_proj_decoder = nn.Linear(d_model, d_model, bias=False)  # For decoder
        
        # ===== Encoder-Decoder Transformer =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output layer normalization
        self.layer_norm = RMSNorm(d_model)
        
        # ===== Output heads (one per quantizer level) =====
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size)
            for _ in range(num_quantizers)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    @staticmethod
    def _build_flat_causal_mask(seq_len: int, num_quantizers: int, device: torch.device) -> torch.Tensor:
        """Build causal mask for flattened token sequence"""
        flat_len = seq_len * num_quantizers
        item_idx = torch.arange(flat_len, device=device) // num_quantizers
        level_idx = torch.arange(flat_len, device=device) % num_quantizers
        
        item_i = item_idx.unsqueeze(1)
        item_j = item_idx.unsqueeze(0)
        level_i = level_idx.unsqueeze(1)
        level_j = level_idx.unsqueeze(0)
        
        allowed = (item_j < item_i) | ((item_j == item_i) & (level_j <= level_i))
        mask = torch.zeros(flat_len, flat_len, device=device)
        mask[~allowed] = float('-inf')
        return mask
    
    def forward(
        self,
        semantic_ids_seq: torch.Tensor,
        user_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            semantic_ids_seq: [B, L, Q] integer codes
            user_ids: [B] user IDs
            attention_mask: [B, L] 1=real token, 0=padding (item-level)
        Returns:
            logits: list of Q tensors, each [B, L, codebook_size]
        """
        B, L, Q = semantic_ids_seq.shape
        device = semantic_ids_seq.device
        flat_len = L * Q
        
        # ===== Embed user IDs =====
        user_emb = self.user_embedding(user_ids).unsqueeze(1)  # [B, 1, d_model]
        
        # ===== Embed semantic ID sequence =====
        level_ids = torch.arange(Q, device=device).unsqueeze(0).expand(L, Q)
        level_ids = level_ids.reshape(-1)  # [L*Q]
        
        codes_flat = semantic_ids_seq.reshape(B, flat_len)  # [B, L*Q]
        
        embedded = torch.zeros(B, flat_len, self.d_model, device=device)
        for q in range(Q):
            pos_q = torch.arange(q, flat_len, Q, device=device)
            embedded[:, pos_q, :] = (
                self.embeddings[q](codes_flat[:, pos_q])
                + self.level_embedding(torch.tensor(q, device=device))
            )
        
        # ===== Prepend user embedding to context =====
        context_embedded = torch.cat([user_emb, embedded], dim=1)  # [B, 1+L*Q, d_model]
        context_embedded = self.pos_encoder(context_embedded)
        
        # ===== Encoder: process full context =====
        # Build padding mask for encoder (include user token which is never padding)
        if attention_mask is not None:
            item_mask = attention_mask.unsqueeze(2).expand(B, L, Q)
            flat_mask = item_mask.reshape(B, flat_len)  # [B, L*Q]
            # Prepend 1 for user token (never padding)
            encoder_mask = torch.cat([torch.ones(B, 1, device=device, dtype=flat_mask.dtype), flat_mask], dim=1)
            key_padding_mask_encoder = (encoder_mask == 0)
        else:
            key_padding_mask_encoder = None
        
        # Project context for encoder
        context_projected = self.in_proj_context(self.dropout(self.norm_encoder(context_embedded)))
        
        # Encode
        memory = self.encoder(
            context_projected,
            src_key_padding_mask=key_padding_mask_encoder
        )  # [B, 1+L*Q, d_model]
        
        # ===== Decoder: predict next item =====
        # Decoder input: BOS token
        decoder_input = self.bos_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # [B, 1, d_model]
        decoder_input = self.pos_encoder(decoder_input)
        
        # Project decoder input
        decoder_projected = self.in_proj_decoder(self.dropout(self.norm_decoder(decoder_input)))
        
        # Decode
        output = self.decoder(
            tgt=decoder_projected,
            memory=memory,
            memory_key_padding_mask=key_padding_mask_encoder
        )  # [B, 1, d_model]
        
        output = self.layer_norm(output)
        
        # ===== Generate logits for each quantizer level =====
        # Use single decoder output to predict all levels
        item_repr = output[:, 0, :]  # [B, d_model]
        
        logits = [head(item_repr) for head in self.output_heads]  # List of [B, codebook_size]
        
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RQVAERecommender:
    """
    Complete RQ-VAE Recommender System with Encoder-Decoder Architecture
    
    Improvements:
    1. User ID embeddings for personalization
    2. Encoder-Decoder Transformer (better than decoder-only)
    3. Gumbel-Softmax for RQ-VAE training
    4. RMSNorm for stable training
    """
    
    def __init__(
        self,
        item_embedding_dim: int,
        rqvae_hidden_dims: List[int] = [512, 256],
        num_quantizers: int = 4,
        codebook_size: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        use_gumbel: bool = True,
        transformer_d_model: int = 256,
        transformer_nhead: int = 8,
        transformer_num_encoder_layers: int = 3,
        transformer_num_decoder_layers: int = 3,
        transformer_dim_feedforward: int = 1024,
        transformer_dropout: float = 0.1,
        max_seq_length: int = 200,
        num_users: int = 50000,
        device: str = 'cpu'
    ):
        self.device = device
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.max_seq_length = max_seq_length
        self.num_users = num_users
        
        # Initialize RQ-VAE with Gumbel-Softmax
        self.rqvae = RQVAE(
            input_dim=item_embedding_dim,
            hidden_dims=rqvae_hidden_dims,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
            use_gumbel=use_gumbel
        ).to(device)
        
        # Initialize Encoder-Decoder Transformer with User Embeddings
        self.transformer = TransformerEncoderDecoderModel(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            max_seq_length=max_seq_length,
            num_users=num_users
        ).to(device)
        
        self.item_to_semantic_id: Dict[int, tuple] = {}
        self.semantic_id_to_items: Dict[tuple, List[int]] = {}
        self.user_to_idx: Dict[int, int] = {}  # user_id -> user_idx mapping
        self._prefix_tree: Optional[Dict] = None
        
    def train_rqvae(
        self,
        item_embeddings: torch.Tensor,
        item_ids: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        temperature_schedule: Optional[List[float]] = None
    ):
        """
        Train RQ-VAE with Gumbel-Softmax temperature annealing
        
        Args:
            item_embeddings: [num_items, embedding_dim]
            item_ids: [num_items] original item IDs
            epochs: number of training epochs
            batch_size: batch size
            learning_rate: learning rate
            temperature_schedule: Gumbel-Softmax temperature schedule (defaults to 1.0→0.5)
        """
        logger.info(f"Training RQ-VAE on {len(item_embeddings)} items...")
        
        if temperature_schedule is None:
            # Default: start at 1.0, anneal to 0.5
            temperature_schedule = [
                max(0.5, 1.0 - 0.5 * (epoch / epochs))
                for epoch in range(epochs)
            ]
        
        self.rqvae.train()
        self.rqvae.set_normalization(item_embeddings)
        self.rqvae = self.rqvae.to(self.device)
        
        optimizer = torch.optim.Adam(self.rqvae.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        dataset = torch.utils.data.TensorDataset(item_embeddings, torch.arange(len(item_embeddings)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        for epoch in range(epochs):
            self.rqvae.train()
            total_loss = 0
            total_recon_loss = 0
            total_vq_loss = 0
            
            temperature = temperature_schedule[epoch]
            
            for batch_emb, batch_idx in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                batch_emb = batch_emb.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                reconstructed, semantic_ids, recon_loss, vq_loss = self.rqvae(
                    batch_emb, 
                    temperature=temperature
                )
                
                loss = recon_loss + vq_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rqvae.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_vq_loss += vq_loss.item()
            
            scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)
            avg_vq = total_vq_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, "
                    f"Temp: {temperature:.3f}"
                )
        
        # Create mapping from items to semantic IDs
        logger.info("Creating item to semantic ID mapping...")
        self.rqvae.eval()
        with torch.no_grad():
            all_sem_ids = []
            for i in range(0, len(item_embeddings), batch_size):
                batch = item_embeddings[i:i+batch_size].to(self.device)
                sem_ids = self.rqvae.get_semantic_ids(batch)
                all_sem_ids.append(sem_ids.cpu())
            semantic_ids_all = torch.cat(all_sem_ids, dim=0)
            
            for idx, (item_id, sem_id) in enumerate(zip(item_ids, semantic_ids_all)):
                sem_id_tuple = tuple(sem_id.numpy().tolist())
                self.item_to_semantic_id[int(item_id)] = sem_id_tuple
                
                if sem_id_tuple not in self.semantic_id_to_items:
                    self.semantic_id_to_items[sem_id_tuple] = []
                self.semantic_id_to_items[sem_id_tuple].append(int(item_id))
        
        logger.info(f"Created {len(self.item_to_semantic_id)} item->semantic_id mappings")
        logger.info(f"Number of unique semantic IDs: {len(self.semantic_id_to_items)}")
        
        total_slots = self.codebook_size ** self.num_quantizers
        utilization = len(self.semantic_id_to_items) / min(total_slots, len(item_embeddings))
        logger.info(f"Codebook utilization: {utilization:.2%}")
        
        self._prefix_tree = None

    def _build_prefix_tree(self) -> Dict:
        """Build a trie over all known semantic IDs for fast constrained beam search"""
        tree: Dict = {}
        for sem_id in self.semantic_id_to_items.keys():
            node = tree
            for code in sem_id:
                if code not in node:
                    node[code] = {}
                node = node[code]
        return tree
        
    def train_transformer(
        self,
        user_sequences: List[Tuple[int, List[int]]],  # (user_id, item_ids)
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        warmup_epochs: int = 5
    ):
        """
        Train Encoder-Decoder Transformer with User Embeddings
        
        Args:
            user_sequences: list of (user_id, item_id_sequence) tuples
            epochs: number of training epochs
            batch_size: batch size
            learning_rate: learning rate
            warmup_epochs: number of warmup epochs
        """
        logger.info(f"Training Transformer on {len(user_sequences)} sequences...")
        
        # Build user ID mapping
        unique_users = sorted(set(user_id for user_id, _ in user_sequences))
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        logger.info(f"Found {len(unique_users)} unique users")
        
        # Convert to semantic sequences
        semantic_sequences = []
        user_indices = []
        for user_id, seq in user_sequences:
            if user_id not in self.user_to_idx:
                continue
            sem_seq = []
            for item_id in seq:
                if item_id in self.item_to_semantic_id:
                    sem_seq.append(self.item_to_semantic_id[item_id])
            if len(sem_seq) > 1:
                semantic_sequences.append(sem_seq)
                user_indices.append(self.user_to_idx[user_id])
        
        logger.info(f"Prepared {len(semantic_sequences)} semantic sequences")
        
        self.transformer.train()
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.codebook_size,
            label_smoothing=0.1
        )
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Shuffle sequences
            indices = np.random.permutation(len(semantic_sequences))
            
            progress_bar = tqdm(
                range(0, len(semantic_sequences), batch_size),
                desc=f"Epoch {epoch+1}/{epochs}"
            )
            
            for i in progress_bar:
                batch_indices = indices[i:i+batch_size]
                batch_seqs = [semantic_sequences[idx] for idx in batch_indices]
                batch_users = [user_indices[idx] for idx in batch_indices]
                
                # Prepare batch: for Encoder-Decoder, we use full sequence as context
                # and predict the NEXT item (last item in sequence)
                batch_inputs = []
                batch_targets = []
                batch_user_ids = []
                
                for seq, user_idx in zip(batch_seqs, batch_users):
                    if len(seq) < 2:
                        continue
                    seq = seq[-self.max_seq_length:]
                    # Input: all but last item (context)
                    # Target: last item (what to predict)
                    batch_inputs.append(seq[:-1])
                    batch_targets.append(seq[-1])  # Single item target!
                    batch_user_ids.append(user_idx)
                
                if not batch_inputs:
                    continue
                
                # Pad sequences
                max_len = max(len(s) for s in batch_inputs)
                padded_inputs = []
                masks = []
                
                for inp in batch_inputs:
                    pad_len = max_len - len(inp)
                    pad_token = (self.codebook_size,) * self.num_quantizers
                    padded_inp = inp + [pad_token] * pad_len
                    mask = [1] * len(inp) + [0] * pad_len
                    
                    padded_inputs.append(padded_inp)
                    masks.append(mask)
                
                # Convert to tensors
                input_tensor = torch.tensor(padded_inputs, dtype=torch.long).to(self.device)  # [B, L, Q]
                target_tensor = torch.tensor(batch_targets, dtype=torch.long).to(self.device)  # [B, Q]
                mask_tensor = torch.tensor(masks, dtype=torch.float).to(self.device)
                user_tensor = torch.tensor(batch_user_ids, dtype=torch.long).to(self.device)
                
                # Forward pass
                optimizer.zero_grad(set_to_none=True)
                logits = self.transformer(
                    input_tensor,
                    user_ids=user_tensor,
                    attention_mask=mask_tensor
                )
                # logits: list of Q tensors, each [B, codebook_size]
                
                # Calculate loss for each quantizer level
                loss = 0
                for level in range(self.num_quantizers):
                    level_logits = logits[level]  # [B, codebook_size]
                    level_targets = target_tensor[:, level]  # [B]
                    loss += criterion(level_logits, level_targets)
                
                loss = loss / self.num_quantizers
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
                )
            
            scheduler.step()
            
            if num_batches > 0 and avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    def recommend(
        self,
        user_id: int,
        user_sequence: List[int],
        top_k: int = 100,
        temperature: float = 1.0
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user with personalization
        
        Args:
            user_id: user ID for personalization
            user_sequence: list of item IDs in user's history
            top_k: number of items to recommend
            temperature: sampling temperature
        Returns:
            recommendations: list of (item_id, score) tuples
        """
        self.transformer.eval()
        
        # Map user_id to index (or use 0 for unknown users)
        user_idx = self.user_to_idx.get(user_id, 0)
        
        # Convert to semantic IDs
        sem_seq = []
        for item_id in user_sequence[-self.max_seq_length:]:
            if item_id in self.item_to_semantic_id:
                sem_seq.append(self.item_to_semantic_id[item_id])
        
        if not sem_seq:
            return []
        
        # Build prefix tree once
        if self._prefix_tree is None:
            self._prefix_tree = self._build_prefix_tree()
        
        # Prepare input tensors
        input_tensor = torch.tensor([sem_seq], dtype=torch.long).to(self.device)
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        
        # Generate predictions using Encoder-Decoder Transformer
        with torch.no_grad():
            logits = self.transformer(input_tensor, user_ids=user_tensor)
            # logits: list of Q tensors, each [1, 1, codebook_size] (decoder outputs one token)
            
            log_probs_per_level = [
                F.log_softmax(lgt[0, 0, :] / max(temperature, 1e-3), dim=-1).cpu()
                for lgt in logits
            ]
        
        expand_k = min(top_k * 2, self.codebook_size)
        
        # Beam search constrained by prefix tree
        beams = [([], 0.0, self._prefix_tree)]
        for level_lp in log_probs_per_level:
            next_beams = []
            top_codes = torch.argsort(level_lp, descending=True)[:expand_k].tolist()
            for partial_id, cum_lp, node in beams:
                for code in top_codes:
                    if code in node:
                        next_beams.append((
                            partial_id + [code],
                            cum_lp + level_lp[code].item(),
                            node[code],
                        ))
            if not next_beams:
                return []
            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[:top_k * 2]
        
        # Convert semantic IDs back to item IDs
        unique_recs: Dict[int, float] = {}
        for partial_id, log_prob, _ in beams:
            if len(partial_id) == self.num_quantizers:
                sem_id_tuple = tuple(partial_id)
                items = self.semantic_id_to_items.get(sem_id_tuple)
                if items:
                    for item in items:
                        if item not in unique_recs or log_prob > unique_recs[item]:
                            unique_recs[item] = log_prob
        
        recommendations = sorted(unique_recs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return recommendations
    
    def recommend_batch(
        self,
        user_data: List[Tuple[int, List[int]]],  # (user_id, sequence)
        top_k: int = 100,
        temperature: float = 1.0,
    ) -> List[List[Tuple[int, float]]]:
        """
        Batch inference with user personalization
        
        Args:
            user_data: list of (user_id, item_id_sequence) tuples
            top_k: number of items to recommend per user
            temperature: softmax temperature
        Returns:
            list of (item_id, score) lists, one per user
        """
        self.transformer.eval()
        
        if self._prefix_tree is None:
            self._prefix_tree = self._build_prefix_tree()
        
        # Convert item sequences → semantic ID sequences
        sem_seqs: List[Optional[List[tuple]]] = []
        user_indices: List[int] = []
        
        for user_id, seq in user_data:
            user_idx = self.user_to_idx.get(user_id, 0)
            s = [
                self.item_to_semantic_id[item_id]
                for item_id in seq[-self.max_seq_length:]
                if item_id in self.item_to_semantic_id
            ]
            sem_seqs.append(s if s else None)
            user_indices.append(user_idx)
        
        valid_batch_indices = [i for i, s in enumerate(sem_seqs) if s is not None]
        
        if not valid_batch_indices:
            return [[] for _ in user_data]
        
        valid_seqs = [sem_seqs[i] for i in valid_batch_indices]
        valid_users = [user_indices[i] for i in valid_batch_indices]
        
        # Pad to common length
        max_len = max(len(s) for s in valid_seqs)
        pad_token = (self.codebook_size,) * self.num_quantizers
        
        padded: List[List[tuple]] = []
        masks: List[List[int]] = []
        for s in valid_seqs:
            pad_len = max_len - len(s)
            padded.append(s + [pad_token] * pad_len)
            masks.append([1] * len(s) + [0] * pad_len)
        
        input_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)  # [B, L, Q]
        mask_tensor = torch.tensor(masks, dtype=torch.float).to(self.device)  # [B, L]
        user_tensor = torch.tensor(valid_users, dtype=torch.long).to(self.device)  # [B]
        
        # Single batched transformer forward pass
        with torch.no_grad():
            logits = self.transformer(
                input_tensor,
                user_ids=user_tensor,
                attention_mask=mask_tensor
            )
            # logits: list of Q tensors, each [B, codebook_size] (decoder outputs one token per user)
            
            # Extract log-probs
            log_probs_per_level = []
            for level_logits in logits:
                # level_logits: [B, V]
                lp = F.log_softmax(
                    level_logits / max(temperature, 1e-3), dim=-1
                ).cpu()  # [B, V]
                log_probs_per_level.append(lp)
        
        # Per-user constrained beam search (CPU)
        expand_k = min(top_k * 2, self.codebook_size)
        results: List[List[Tuple[int, float]]] = [[] for _ in user_data]
        
        for batch_idx, orig_idx in enumerate(valid_batch_indices):
            user_log_probs = [lp[batch_idx] for lp in log_probs_per_level]
            
            beams = [([], 0.0, self._prefix_tree)]
            
            for level_lp in user_log_probs:
                next_beams = []
                top_codes = torch.argsort(level_lp, descending=True)[:expand_k].tolist()
                
                for partial_id, cum_lp, node in beams:
                    for code in top_codes:
                        if code in node:
                            next_beams.append((
                                partial_id + [code],
                                cum_lp + level_lp[code].item(),
                                node[code],
                            ))
                
                if not next_beams:
                    break
                next_beams.sort(key=lambda x: x[1], reverse=True)
                beams = next_beams[:top_k * 2]
            else:
                # Collect item-level candidates
                unique_recs: Dict[int, float] = {}
                for partial_id, log_prob, _ in beams:
                    if len(partial_id) != self.num_quantizers:
                        continue
                    sem_id_tuple = tuple(partial_id)
                    items = self.semantic_id_to_items.get(sem_id_tuple)
                    if items is None:
                        continue
                    for item in items:
                        if item not in unique_recs or log_prob > unique_recs[item]:
                            unique_recs[item] = log_prob
                
                results[orig_idx] = sorted(
                    unique_recs.items(), key=lambda x: x[1], reverse=True
                )[:top_k]
        
        return results

    def save(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'rqvae_state_dict': self.rqvae.state_dict(),
            'transformer_state_dict': self.transformer.state_dict(),
            'item_to_semantic_id': self.item_to_semantic_id,
            'semantic_id_to_items': self.semantic_id_to_items,
            'user_to_idx': self.user_to_idx
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.rqvae.load_state_dict(checkpoint['rqvae_state_dict'])
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.item_to_semantic_id = checkpoint['item_to_semantic_id']
        self.semantic_id_to_items = checkpoint['semantic_id_to_items']
        self.user_to_idx = checkpoint.get('user_to_idx', {})
        self._prefix_tree = None
        logger.info(f"Model loaded from {path}")
