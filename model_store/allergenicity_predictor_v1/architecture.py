"""MLP head of the allergenicity ensemble: mean-pooled ESM2-150M embedding -> allergen logit."""

from __future__ import annotations

import torch.nn as nn

EMBED_DIM = 640


class MLP(nn.Module):
    """640 -> 480 -> 240 (+Dropout 0.5) -> 64 -> 1 logit; BatchNorm + GELU after each hidden layer."""

    def __init__(self, d_in: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 480), nn.BatchNorm1d(480), nn.GELU(),
            nn.Linear(480, 240), nn.BatchNorm1d(240), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(240, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
