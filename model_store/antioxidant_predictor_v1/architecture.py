"""CNN, BiLSTM and Transformer base networks of the antioxidant ensemble, on one-hot residues; each returns raw logits of shape (B,)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _mask(x: torch.Tensor) -> torch.Tensor:
    """(B, L) bool, True on real residues (padding rows are all zeros)."""
    return x.abs().sum(-1) > 0


class CNN(nn.Module):
    """Conv1D(64, k=3) -> MaxPool1D -> Flatten -> FC(ReLU) -> logit."""

    def __init__(self, in_dim, max_len, filters=64, kernel_size=3, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_dim, filters, kernel_size), nn.ReLU())
        self.pool = nn.MaxPool1d(2)
        flat = filters * ((max_len - kernel_size + 1) // 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        h = self.pool(self.conv(x.transpose(1, 2)))
        return self.fc(h).squeeze(-1)


class BiLSTM(nn.Module):
    """BiLSTM(64) -> Dropout(.5) -> BiLSTM(32) -> Dropout(.5) -> FC(ReLU) -> logit; padding-aware."""

    def __init__(self, in_dim, units1=64, units2=32, dropout=0.5, hidden_dim=32):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, units1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * units1, units2, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(2 * units2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        L = x.size(1)
        lengths = _mask(x).sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm1(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=L)
        out = self.drop(out)
        packed = pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm2(packed)
        h = self.drop(torch.cat([h[0], h[1]], dim=-1))
        return self.fc(h).squeeze(-1)


class TransformerNet(nn.Module):
    """Linear projection + learned positions -> MHSA(4 heads) -> Add&Norm -> FFN -> Add&Norm -> masked mean pool -> FC(ReLU) -> logit."""

    def __init__(self, in_dim, max_len, d_model=64, num_heads=4, ff_dim=128, dropout=0.1, hidden_dim=32):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        mask = _mask(x)
        pos = torch.arange(x.size(1), device=x.device)
        h = self.proj(x) + self.pos(pos)
        a, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = self.norm1(h + self.drop1(a))
        h = self.norm2(h + self.ff(h))
        m = mask.unsqueeze(-1).to(h.dtype)
        h = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.fc(self.drop2(h)).squeeze(-1)
