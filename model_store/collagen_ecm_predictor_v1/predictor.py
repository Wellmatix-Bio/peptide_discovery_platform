"""Loader/inference wrapper for the collagen/ECM-activity PU-bagging logistic regression on ESM2-150M embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

# Training length range (2-50 per the notebook; observed 2-50 aa).
MIN_LENGTH, MAX_LENGTH = 2, 50


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    bad = sorted(set(sequence) - AMINO_ACID_SET)
    if bad:
        raise ValueError(
            f"sequence contains residues outside the 20 uppercase standard amino acids: {bad}"
        )
    if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
        raise ValueError(
            f"sequence length {len(sequence)} outside the trained range {MIN_LENGTH}-{MAX_LENGTH}."
        )


def esm_mean_embeddings(
    sequences: list[str], feature_extractor: FeatureExtractor
) -> np.ndarray:
    """Mean over residue positions of the last hidden layer with BOS/EOS excluded: this model's trained convention."""
    embeddings = feature_extractor.get_esm2_embedding_batch(sequences)
    return np.stack([e.hidden_states[1:-1].mean(axis=0) for e in embeddings])


class CollagenECMPredictor:
    """Lazy-loaded mean of 500 bagged logistic regressions on standardised + PCA-reduced embeddings; returns a ranking score, see README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.reducer = joblib.load(self.model_dir / "reducer.joblib")
        bags = np.load(self.model_dir / "bags.npz")
        self.coef = np.asarray(bags["coef"], dtype=float)  # (n_bags, n_components)
        self.intercept = np.asarray(bags["intercept"], dtype=float)  # (n_bags,)
        with open(self.model_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.embedding_dim = int(self.reducer[0].n_features_in_)
        if self.coef.ndim != 2 or self.intercept.shape != (self.coef.shape[0],):
            raise ValueError("bags.npz: coef must be (n_bags, n_components), intercept (n_bags,).")
        self.loaded = True

    def _bag_probabilities(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> np.ndarray:
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        self._load()
        X = esm_mean_embeddings(sequences, feature_extractor)
        if X.shape[1] != self.embedding_dim:
            raise ValueError(
                f"ESM2 embedding dim {X.shape[1]} != trained dim {self.embedding_dim}."
            )
        Z = self.reducer.transform(X)
        return 1.0 / (1.0 + np.exp(-(Z @ self.coef.T + self.intercept)))  # (n, n_bags)

    def predict_proba_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> list[float]:
        """Mean P(active) across bags, one value per sequence."""
        if not sequences:
            return []
        return [
            float(v) for v in self._bag_probabilities(sequences, feature_extractor).mean(1)
        ]

    def predict_proba_with_std_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> tuple[list[float], list[float]]:
        """Mean and across-bag standard deviation of P(active)."""
        if not sequences:
            return [], []
        p = self._bag_probabilities(sequences, feature_extractor)
        return [float(v) for v in p.mean(1)], [float(v) for v in p.std(1)]

    def predict_proba(self, sequence: str, feature_extractor: FeatureExtractor) -> float:
        return self.predict_proba_batch([sequence], feature_extractor)[0]
