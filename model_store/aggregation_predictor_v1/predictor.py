"""Loader/inference wrapper for the aggregation-propensity XGBoost classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )
    if len(sequence) < 5:
        raise ValueError("sequence must be at least 5 residues (QSO/SOCN/PAAC/APAAC lag requirement).")


def protscale_features(seq: str, aaindex_scales: dict) -> dict:
    feats = {}
    for name, table in aaindex_scales.items():
        values = np.array([table[c] for c in seq], dtype=float)
        feats[f"ps_{name}_mean"] = values.mean()
        feats[f"ps_{name}_std"] = values.std()
        feats[f"ps_{name}_min"] = values.min()
        feats[f"ps_{name}_max"] = values.max()
    return feats


class AggregationPredictor:
    """Lazy-loaded XGBoost classifier over biopython/propy/AAindex descriptors returning an aggregation-propensity probability; see README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        with open(self.model_dir / "feature_columns.json", "r", encoding="utf-8") as f:
            self.feature_cols = json.load(f)
        with open(self.model_dir / "aaindex1_scales.json", "r", encoding="utf-8") as f:
            self.aaindex_scales = json.load(f)
        self.model = joblib.load(self.model_dir / "xgb_seed_bh.joblib")
        if hasattr(self.model, "set_params"):
            self.model.set_params(device="cpu")
        self.loaded = True

    def predict_aggregation(
        self, sequence: str, feature_extractor: FeatureExtractor
    ) -> float:
        _validate_sequence(sequence)
        self._load()
        seq = sequence.upper()
        feats = {}
        feats.update(feature_extractor.get_propy_biopython_descriptors(seq))
        feats.update(protscale_features(seq, self.aaindex_scales))
        row = pd.DataFrame([feats])
        missing = [col for col in self.feature_cols if col not in row.columns]
        if missing:
            raise ValueError(f"Missing aggregation model features: {missing[:10]}")
        X = row[self.feature_cols]
        proba = self.model.predict_proba(X)[:, 1]
        return float(np.clip(proba[0], 0.0, 1.0))
