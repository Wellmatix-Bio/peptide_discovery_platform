"""Network definitions for the two members of the allergenicity model; layouts must match the saved weights."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 640
CNN_LEN = 55  # residues; shorter sequences are zero-padded
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


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


class AllergenCNN(nn.Module):
    """One-hot residues (20 channels x CNN_LEN) -> Conv1d(64, k=3) -> AvgPool(3) -> 120 -> 60 -> 2 logits."""

    def __init__(self, length: int = CNN_LEN):
        super().__init__()

        def bn(n: int) -> nn.BatchNorm1d:
            return nn.BatchNorm1d(n, eps=1e-3, momentum=0.01)

        self.net = nn.Sequential(
            nn.Conv1d(20, 64, 3), nn.ReLU(), nn.AvgPool1d(3), nn.Flatten(),
            nn.Linear(64 * ((length - 2) // 3), 120), nn.ReLU(), bn(120),
            nn.Linear(120, 60), nn.ReLU(), bn(60),
            nn.Linear(60, 2),
        )

    def forward(self, x):
        return self.net(x)


def one_hot(sequences: list[str], device: torch.device) -> torch.Tensor:
    """(B, 20, CNN_LEN) float one-hot; padding and truncation as in training."""
    idx = torch.full((len(sequences), CNN_LEN), -1, dtype=torch.long)
    for i, s in enumerate(sequences):
        ids = [AA_IDX.get(c, -1) for c in s[:CNN_LEN]]
        idx[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    idx = idx.to(device)
    return (F.one_hot(idx.clamp(min=0), 20).float() * (idx >= 0).unsqueeze(-1)).transpose(1, 2)
