"""Loader/inference wrapper for the pigmentation (TIP vs non-TIP) soft-voting ensemble on ESM2-150M embeddings."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

# Observed training length range (2-22 aa).
MIN_LENGTH, MAX_LENGTH = 2, 22


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


class PigmentationPredictor:
    """Lazy-loaded equal-weight soft vote of RF, XGBoost, calibrated RBF-SVC and shrinkage LDA; returns P(TIP), see README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        # Trusted first-party artifact: joblib unpickles arbitrary objects.
        bundle = joblib.load(self.model_dir / "ensemble.joblib")
        self.ensemble = bundle["ensemble"]
        self.threshold = float(bundle["threshold"])
        self.embedding_dim = int(bundle["feature_dimension"])
        self.positive_index = list(self.ensemble.classes_).index(1)
        self.loaded = True

    def predict_proba_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> list[float]:
        """P(TIP) per sequence."""
        if not sequences:
            return []
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
        proba = self.ensemble.predict_proba(X)[:, self.positive_index]
        return [float(v) for v in proba]

    def predict_proba(self, sequence: str, feature_extractor: FeatureExtractor) -> float:
        return self.predict_proba_batch([sequence], feature_extractor)[0]
