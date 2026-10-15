"""Multi-receptor agonism classifier: P(EC50 < 1000 pM) at GCGR, GLP1R and GIPR from per-token physicochemical features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common.model_sync import sync_model_weights, weights_dir_for

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

RECEPTORS = ("GLP1R", "GIPR", "GCGR")
LABEL_THRESHOLD_PM = 1000.0  # positive label: EC50 below this many picomolar (1 nM)


class TokenMLP(nn.Module):
    """Shared trunk (max_len * 7 -> 256 -> 128 -> 64) and one Linear(64 -> 1) head per receptor; returns logits (B, R)."""

    def __init__(self, in_dim: int, n_receptors: int, hidden=(256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList(nn.Linear(prev, 1) for _ in range(n_receptors))

    def forward(self, x):
        z = self.trunk(x)
        return torch.cat([head(z) for head in self.heads], dim=1)


class AgonismClassifier:
    """Lazy-loaded mean of five fold MLPs; each fold carries its own z-score statistics and padding length.

    Token features come from the shared `FeatureExtractor.get_token_features`, so the extractor is required.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.members = []
        for path in sorted(self.model_dir.glob("fold*.pt")):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = TokenMLP(ck["in_dim"], len(ck["receptors"])).to(self.device)
            model.load_state_dict(ck["state_dict"])
            scaler = {k: np.asarray(v, dtype=np.float32) for k, v in ck["scaler"].items()}
            self.members.append((model.eval(), scaler, int(ck["max_len"])))
            self.columns = list(ck["receptors"])
        if not self.members:
            raise FileNotFoundError(f"No fold*.pt checkpoints in {self.model_dir}")
        self.max_len = min(m[2] for m in self.members)
        self.loaded = True

    @torch.no_grad()
    def predict_proba_batch(
        self, sequences: list[str], feature_extractor
    ) -> dict[str, list[float | None]]:
        """{receptor: [P(EC50 < 1000 pM), ...]}; `None` for a sequence the model cannot encode (unknown residue or more tokens than it was trained on)."""
        if feature_extractor is None:
            raise ValueError("AgonismClassifier needs the shared FeatureExtractor for token features.")
        self._load()
        raws: list[np.ndarray | None] = []
        for sequence in sequences:
            try:
                raw = feature_extractor.get_token_features(sequence)
                raws.append(raw if len(raw) <= self.max_len else None)
            except (ValueError, KeyError):
                raws.append(None)
        ok = [i for i, r in enumerate(raws) if r is not None]
        out: dict[str, list[float | None]] = {r: [None] * len(sequences) for r in self.columns}
        if ok:
            folds = []
            for model, scaler, max_len in self.members:
                X = np.zeros((len(ok), max_len, 7), dtype=np.float32)
                for j, i in enumerate(ok):
                    x = raws[i].copy()
                    x[:, :3] = (x[:, :3] - scaler["mean"]) / scaler["std"]
                    X[j, : len(x)] = x
                logits = model(torch.tensor(X.reshape(len(ok), -1)).to(self.device))
                folds.append(torch.sigmoid(logits).cpu().numpy())
            mean = np.mean(folds, axis=0)
            for c, receptor in enumerate(self.columns):
                for j, i in enumerate(ok):
                    out[receptor][i] = float(mean[j, c])
        return {r: out[r] for r in RECEPTORS if r in out}

    def predict_proba(self, sequence: str, feature_extractor) -> dict[str, float | None]:
        return {r: v[0] for r, v in self.predict_proba_batch([sequence], feature_extractor).items()}
