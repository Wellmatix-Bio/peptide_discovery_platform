"""The MLP member: 14 -> 480 -> 200 -> 64 -> 1, BatchNorm + GELU, dropout 0.5 on the first two blocks."""

from __future__ import annotations

from torch import nn

N_FEATURES = 14


def build_net(n_in: int = N_FEATURES, p_drop: float = 0.5) -> nn.Sequential:
    def block(i: int, o: int, drop: bool = True) -> list[nn.Module]:
        layers: list[nn.Module] = [nn.Linear(i, o), nn.BatchNorm1d(o), nn.GELU()]
        return layers + [nn.Dropout(p_drop)] if drop else layers

    return nn.Sequential(*block(n_in, 480), *block(480, 200), *block(200, 64, drop=False), nn.Linear(64, 1))
