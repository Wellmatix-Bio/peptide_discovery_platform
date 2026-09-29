"""Loader/inference wrapper for the synthesis-feasibility ensemble."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import torch

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
MODEL_FILES = ["logistic_regression.pkl", "svm.pkl", "random_forest.pkl"]


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


def esm_embedding_cached(sequence: str, feature_extractor: FeatureExtractor) -> np.ndarray:
    """Mean-pooled ESM2-150M embedding, excluding CLS/EOS special tokens
    (positions 0 and -1 of the cache's already-trimmed per-sequence tensor --
    equivalent to the original get_special_tokens_mask exclusion for this
    tokenizer, which has exactly one BOS and one EOS and no other specials)."""
    embedding = feature_extractor.get_esm2_embedding(sequence)
    mean_pooled = embedding.hidden_states[1:-1].mean(axis=0, keepdims=True)
    return mean_pooled


class SynthesisFeasibilityEnsemble:
    """Lazy-loaded soft-voting ensemble of Logistic Regression, SVM, and
    Random Forest over ESM2 embeddings. Returns a synthesis-feasibility
    probability in [0, 1]. See README.md for weak-generalization caveat."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.models = [joblib.load(self.model_dir / f) for f in MODEL_FILES]
        self.loaded = True

    def predict_proba(self, sequence: str, feature_extractor: FeatureExtractor) -> float:
        _validate_sequence(sequence)
        self._load()
        X = esm_embedding_cached(sequence, feature_extractor)
        probs = [model.predict_proba(X)[:, 1][0] for model in self.models]
        return float(np.mean(probs))
