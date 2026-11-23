"""Shared inference for the cytokine-induction predictors: mean-pooled ESM2-150M features into one or more fitted classifiers, averaged."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from common.applicability import ReferenceProfile, with_similarity
from common.similarity import SimilarityIndex
from common.model_sync import sync_model_weights

if TYPE_CHECKING:
    from pipeline.feature_extractor import FeatureExtractor

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")
ESM_DIM = 640
THRESHOLD = 0.5


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    bad = sorted(set(sequence) - AMINO_ACID_SET)
    if bad:
        raise ValueError(f"sequence contains residues outside the 20 uppercase standard amino acids: {bad}")


def esm_mean_embeddings(sequences: list[str], feature_extractor: "FeatureExtractor") -> np.ndarray:
    """Mean over residue positions of the last hidden layer with BOS/EOS excluded: the training convention."""
    embeddings = feature_extractor.get_esm2_embedding_batch(sequences)
    return np.stack([e.hidden_states[1:-1].mean(axis=0) for e in embeddings])


def _mlp_net(dim: int, hidden: int = 256, dropout: float = 0.3):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    )  # fmt: skip


class Member:
    """One fitted classifier, loaded from the plain dict written by scripts/export_cytokine_models.py."""

    def __init__(self, payload: dict) -> None:
        self.kind = payload["kind"]  # RF | XGBoost | SVM_RBF | LR-L2 | MLP
        self.scaler = payload.get("scaler")
        self.est = payload.get("est")
        self.net = None
        if self.kind == "MLP":
            import torch

            self.net = _mlp_net(int(payload["dim"]))
            self.net.load_state_dict(payload["state"])
            self.net.eval()
            self._torch = torch

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(y=1) for raw (unscaled) features."""
        if self.net is not None:
            with self._torch.no_grad():
                x = self._torch.as_tensor(self.scaler.transform(X), dtype=self._torch.float32)
                return self._torch.sigmoid(self.net(x).squeeze(1)).numpy()
        features = X if self.kind in ("RF", "XGBoost") else self.scaler.transform(X)
        return self.est.predict_proba(features)[:, 1]


class CytokineEnsemble:
    """Mean P(induces the cytokine) over MEMBERS; subclasses set CODE_DIR-relative MEMBERS in their predictor module."""

    MEMBERS: tuple[str, ...] = ()

    def __init__(self, code_dir: Path, model_dir: Path) -> None:
        self.code_dir = Path(code_dir)
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(self.code_dir)
        info = json.loads((self.model_dir / "info.json").read_text(encoding="utf-8"))
        self.length_range = (int(info["training"]["length_min"]), int(info["training"]["length_max"]))
        self.profile = ReferenceProfile.from_info(info)
        self.similarity = SimilarityIndex.load(self.model_dir, info)  # None for weights without a similarity profile
        self.members = {name: Member(joblib.load(self.model_dir / f"{name}.joblib")) for name in self.MEMBERS}
        self.loaded = True

    def _member_probabilities(self, sequences: list[str], feature_extractor: "FeatureExtractor") -> dict[str, np.ndarray]:
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        self._load()
        X = esm_mean_embeddings(sequences, feature_extractor)
        if X.shape[1] != ESM_DIM:
            raise ValueError(f"ESM2 embedding dim {X.shape[1]} != trained dim {ESM_DIM}.")
        return {name: member.predict_proba(X) for name, member in self.members.items()}

    def predict_proba_batch(self, sequences: list[str], feature_extractor: "FeatureExtractor") -> list[float]:
        if not sequences:
            return []
        member_probs = self._member_probabilities(sequences, feature_extractor)
        return [float(v) for v in np.mean(list(member_probs.values()), axis=0)]

    def predict_proba(self, sequence: str, feature_extractor: "FeatureExtractor") -> float:
        return self.predict_proba_batch([sequence], feature_extractor)[0]

    def predict_batch(self, sequences: list[str], feature_extractor: "FeatureExtractor") -> list[dict]:
        """Per sequence: probability, label at 0.5, each member's probability, whether the length is inside the trained range, and its applicability assessment."""
        if not sequences:
            return []
        member_probs = self._member_probabilities(sequences, feature_extractor)
        mean = np.mean(list(member_probs.values()), axis=0)
        low, high = self.length_range
        similar = self.similarity.assess(sequences) if self.similarity else [None] * len(sequences)
        return [
            {
                "probability": float(mean[i]),
                "label": int(mean[i] >= THRESHOLD),
                "members": {name: float(p[i]) for name, p in member_probs.items()},
                "in_length_range": low <= len(sequence) <= high,
                "applicability": with_similarity(self.profile.assess(len(sequence)), similar[i]) if similar[i] else self.profile.assess(len(sequence)),
            }
            for i, sequence in enumerate(sequences)
        ]
