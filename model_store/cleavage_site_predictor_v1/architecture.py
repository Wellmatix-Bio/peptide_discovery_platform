from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

PCA_TARGET_DIM = 128
ENCODER_HIDDEN_DIM = PCA_TARGET_DIM
ENCODER_NUM_HEADS = 4
ENCODER_NUM_LAYERS = 5
ENCODER_FFN_DIM = ENCODER_HIDDEN_DIM * 2
ENCODER_DROPOUT = 0.1
LOCAL_ATTN_SPAN = 128  # residue i attends only to [i - span, i + span]
DISTANCE_NUM_RBF = 10
POOL_NUM_RBF = 10
CLEAVAGE_WINDOW_RADIUS = 15  # +/-15 around candidate position t -> a 31-residue window


class DistanceBias(nn.Module):
    """Projects a (B, L, L) distance matrix to a per-head additive attention bias (B, num_heads, L, L) via an RBF kernel + small MLP."""

    def __init__(self, num_heads, num_rbf=DISTANCE_NUM_RBF, max_distance=50.0):
        super().__init__()
        self.num_heads = num_heads
        centers = torch.linspace(0.0, max_distance, num_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = num_rbf / max_distance
        self.proj = nn.Sequential(
            nn.Linear(num_rbf, num_rbf),
            nn.ReLU(),
            nn.Linear(num_rbf, num_heads),
        )

    def forward(self, distance_matrix):
        # distance_matrix: (B, L, L) -> rbf: (B, L, L, num_rbf)
        d = distance_matrix.unsqueeze(-1)
        rbf = torch.exp(-self.rbf_gamma * (d - self.rbf_centers) ** 2)
        bias = self.proj(rbf)  # (B, L, L, num_heads)
        return bias.permute(0, 3, 1, 2)  # (B, num_heads, L, L)


def local_attention_band(length, span, device, dtype):
    """(L, L) boolean mask, True where |i - j| <= span."""
    idx = torch.arange(length, device=device)
    band = (idx[:, None] - idx[None, :]).abs() <= span
    return band


def build_combined_bias(
    distance_bias_module, distance_matrices, mask, local_span=LOCAL_ATTN_SPAN
):
    """One (B, num_heads, L, L) additive bias, masked to the local band and real key positions; shared by every encoder layer."""
    B, L, _ = distance_matrices.shape
    dist_bias = distance_bias_module(distance_matrices)  # (B, H, L, L)

    band = local_attention_band(
        L, local_span, distance_matrices.device, dist_bias.dtype
    )  # (L, L) bool
    outside_band = ~band  # True where attention is disallowed by the span cap

    pad_key = ~mask  # (B, L) True where padded
    neg_inf = torch.finfo(dist_bias.dtype).min

    attn_bias = dist_bias.masked_fill(outside_band[None, None, :, :], neg_inf)
    attn_bias = attn_bias.masked_fill(pad_key[:, None, None, :], neg_inf)
    return attn_bias


class LocalDistanceBiasedEncoderLayer(nn.Module):
    """One pre-norm transformer block taking the shared precomputed attention bias."""

    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        """attn_bias: (B, num_heads, L, L) additive bias, -inf outside the local band and at padded keys."""
        B, L, _ = x.shape
        h = self.norm1(x)
        bias_flat = attn_bias.reshape(B * self.num_heads, L, L)
        attn_out, _ = self.attn(h, h, h, attn_mask=bias_flat, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class EnzymeEncoder(nn.Module):
    """PCA-reduced ESM-2 embedding projected to hidden_dim, then local-attention distance-biased transformer layers to per-residue states."""

    def __init__(
        self,
        embed_dim=PCA_TARGET_DIM,
        hidden_dim=ENCODER_HIDDEN_DIM,
        num_heads=ENCODER_NUM_HEADS,
        num_layers=ENCODER_NUM_LAYERS,
        ffn_dim=ENCODER_FFN_DIM,
        dropout=ENCODER_DROPOUT,
        local_span=LOCAL_ATTN_SPAN,
        use_gradient_checkpointing=True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.distance_bias = DistanceBias(num_heads)
        self.layers = nn.ModuleList(
            [
                LocalDistanceBiasedEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.num_heads = num_heads
        self.local_span = local_span
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, embeddings, distance_matrices, mask):
        """embeddings (B, L, embed_dim), distance_matrices (B, L, L), mask (B, L) -> per-residue states (B, L, hidden_dim)."""
        x = self.input_proj(embeddings)
        attn_bias = build_combined_bias(
            self.distance_bias, distance_matrices, mask, self.local_span
        )

        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, attn_bias, use_reentrant=False)
            else:
                x = layer(x, attn_bias)

        return x


class SubstrateEncoder(nn.Module):
    """Same architecture as EnzymeEncoder, returning per-residue states H^s with no pooling."""

    def __init__(
        self,
        embed_dim=PCA_TARGET_DIM,
        hidden_dim=ENCODER_HIDDEN_DIM,
        num_heads=ENCODER_NUM_HEADS,
        num_layers=ENCODER_NUM_LAYERS,
        ffn_dim=ENCODER_FFN_DIM,
        dropout=ENCODER_DROPOUT,
        local_span=LOCAL_ATTN_SPAN,
        use_gradient_checkpointing=True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.distance_bias = DistanceBias(num_heads)
        self.layers = nn.ModuleList(
            [
                LocalDistanceBiasedEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.num_heads = num_heads
        self.local_span = local_span
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, embeddings, distance_matrices, mask):
        """embeddings (B, L, embed_dim), distance_matrices (B, L, L), mask (B, L) -> per-residue states H^s (B, L, hidden_dim)."""
        x = self.input_proj(embeddings)
        attn_bias = build_combined_bias(
            self.distance_bias, distance_matrices, mask, self.local_span
        )

        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, attn_bias, use_reentrant=False)
            else:
                x = layer(x, attn_bias)

        return x


class ActiveSiteHead(nn.Module):
    """Per-residue active-site logit plus a small Gaussian-kernel + MLP f(.) that turns the active-site probability into an attention-pooling score."""

    def __init__(
        self,
        hidden_dim=ENCODER_HIDDEN_DIM,
        num_rbf=POOL_NUM_RBF,
        min_prob=0.0,
        max_prob=1.0,
    ):
        super().__init__()
        self.site_logit = nn.Linear(hidden_dim, 1)

        centers = torch.linspace(min_prob, max_prob, num_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = num_rbf / (max_prob - min_prob)
        self.pool_score_fn = nn.Sequential(
            nn.Linear(num_rbf, num_rbf),
            nn.ReLU(),
            nn.Linear(num_rbf, 1),
        )

    def forward(self, hidden_states, mask):
        """hidden_states (B, L, hidden_dim), mask (B, L) -> (site_logits (B, L), pooled h^e (B, hidden_dim), pool_weights (B, L))."""
        site_logits = self.site_logit(hidden_states).squeeze(-1)  # (B, L)
        site_probs = torch.sigmoid(site_logits)  # a_hat_i, the paper's pooling input

        clamped = site_probs.clamp(self.rbf_centers.min(), self.rbf_centers.max())
        rbf = torch.exp(
            -self.rbf_gamma * (clamped.unsqueeze(-1) - self.rbf_centers) ** 2
        )  # (B, L, num_rbf)
        pool_scores = self.pool_score_fn(rbf).squeeze(-1)  # (B, L)
        pool_scores = pool_scores.masked_fill(~mask, torch.finfo(pool_scores.dtype).min)
        pool_weights = torch.softmax(pool_scores, dim=-1)  # (B, L)

        pooled = torch.einsum("bl,blh->bh", pool_weights, hidden_states)

        return site_logits, pooled, pool_weights


class CleavageSiteHead(nn.Module):
    """Concatenates a (2*window_radius + 1)-residue window of H^s around position t with the pooled enzyme h^e, then an MLP to a cleavage logit."""

    def __init__(
        self,
        hidden_dim=ENCODER_HIDDEN_DIM,
        window_radius=CLEAVAGE_WINDOW_RADIUS,
        mlp_hidden=128,
    ):
        super().__init__()
        self.window_radius = window_radius
        window_len = 2 * window_radius + 1
        concat_dim = window_len * hidden_dim + hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def _gather_window(self, H_s, substrate_mask, idx):
        """H_s (B, L_s, hidden_dim), substrate_mask, idx (B, N, window_len) -> gathered windows (zeroed outside validity) and a validity mask."""
        B, L_s, H = H_s.shape
        N, window_len = idx.shape[1], idx.shape[2]

        in_bounds = (idx >= 0) & (idx < L_s)
        idx_clamped = idx.clamp(0, L_s - 1)
        idx_flat = idx_clamped.reshape(B, N * window_len)

        gathered = torch.gather(
            H_s, 1, idx_flat.unsqueeze(-1).expand(-1, -1, H)
        ).reshape(B, N, window_len, H)

        real_residue = torch.gather(substrate_mask, 1, idx_flat).reshape(
            B, N, window_len
        )
        valid = in_bounds & real_residue

        gathered = gathered * valid.unsqueeze(-1).float()
        return gathered, valid

    def substrate_window(self, H_s, substrate_mask, positions):
        """positions (B,) 0-indexed -> (B, window_len, hidden_dim) window and (B, window_len) validity mask."""
        offsets = torch.arange(
            -self.window_radius, self.window_radius + 1, device=H_s.device
        )
        idx = (positions[:, None] + offsets[None, :]).unsqueeze(1)  # (B, 1, window_len)
        gathered, valid = self._gather_window(H_s, substrate_mask, idx)
        return gathered.squeeze(1), valid.squeeze(1)

    def forward(self, H_s, substrate_mask, positions, h_e):
        """H_s, substrate_mask, positions (B,), h_e (B, hidden_dim) -> (B,) raw cleavage logit per candidate position."""
        window, _valid = self.substrate_window(
            H_s, substrate_mask, positions
        )  # (B, window_len, H)
        B = window.shape[0]
        combined = torch.cat([window.reshape(B, -1), h_e], dim=-1)
        return self.mlp(combined).squeeze(-1)

    def forward_all_positions(self, H_s, substrate_mask, h_e):
        """H_s, substrate_mask, h_e -> (B, L_s) raw cleavage logits at every position"""
        B, L_s, H = H_s.shape
        offsets = torch.arange(
            -self.window_radius, self.window_radius + 1, device=H_s.device
        )  # (window_len,)
        all_positions = torch.arange(L_s, device=H_s.device)  # (L_s,)
        idx = (all_positions[None, :, None] + offsets[None, None, :]).expand(
            B, -1, -1
        )  # (B, L_s, window_len)

        window, _valid = self._gather_window(
            H_s, substrate_mask, idx
        )  # (B, L_s, window_len, H)
        window_flat = window.reshape(B, L_s, -1)
        h_e_expanded = h_e.unsqueeze(1).expand(-1, L_s, -1)  # (B, L_s, H)
        combined = torch.cat(
            [window_flat, h_e_expanded], dim=-1
        )  # (B, L_s, window_len*H + H)
        return self.mlp(combined).squeeze(-1)  # (B, L_s)


class JointModel(nn.Module):
    """Shared enzyme side (EnzymeEncoder + ActiveSiteHead) plus substrate side (SubstrateEncoder + CleavageSiteHead) used only for L_c."""

    def __init__(self, init_from_active_site_model=None, **encoder_kwargs):
        super().__init__()
        self.enzyme_encoder = EnzymeEncoder(**encoder_kwargs)
        self.active_site_head = ActiveSiteHead(
            hidden_dim=encoder_kwargs.get("hidden_dim", ENCODER_HIDDEN_DIM)
        )
        self.substrate_encoder = SubstrateEncoder(**encoder_kwargs)
        self.cleavage_head = CleavageSiteHead(
            hidden_dim=encoder_kwargs.get("hidden_dim", ENCODER_HIDDEN_DIM)
        )

        if init_from_active_site_model is not None:
            self.enzyme_encoder.load_state_dict(
                init_from_active_site_model.encoder.state_dict()
            )
            self.active_site_head.load_state_dict(
                init_from_active_site_model.head.state_dict()
            )

    def forward_active_site(self, enzyme_embeddings, enzyme_distances, enzyme_mask):
        """L_a path: one enzyme batch -> per-residue active-site logits."""
        enzyme_hidden = self.enzyme_encoder(
            enzyme_embeddings, enzyme_distances, enzyme_mask
        )
        site_logits, h_e, pool_weights = self.active_site_head(
            enzyme_hidden, enzyme_mask
        )
        return {"site_logits": site_logits, "h_e": h_e, "pool_weights": pool_weights}

    def forward_cleavage(
        self,
        enzyme_embeddings,
        enzyme_distances,
        enzyme_mask,
        substrate_embeddings,
        substrate_distances,
        substrate_mask,
    ):
        """L_c path: one (enzyme, substrate) pair batch to per-substrate-residue cleavage logits."""
        enzyme_hidden = self.enzyme_encoder(
            enzyme_embeddings, enzyme_distances, enzyme_mask
        )
        _site_logits, h_e, _pool_weights = self.active_site_head(
            enzyme_hidden, enzyme_mask
        )

        H_s = self.substrate_encoder(
            substrate_embeddings, substrate_distances, substrate_mask
        )
        cleavage_logits = self.cleavage_head.forward_all_positions(
            H_s, substrate_mask, h_e
        )

        return {"cleavage_logits": cleavage_logits, "h_e": h_e, "H_s": H_s}

    @classmethod
    def load_pretrained(cls, checkpoint_path, device="cpu", **encoder_kwargs):
        checkpoint_path = Path(checkpoint_path)
        model = cls(**encoder_kwargs).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        model.load_state_dict(state_dict)

        model.eval()
        return model
