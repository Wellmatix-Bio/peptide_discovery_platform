"""Loader/inference wrapper for the peptide-in-solvent solubility classifier."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import joblib
import torch

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
EMBED_DIM = 640

SOLVENT_COLS = [
    "pH_value", "pH_missing_flag", "ionic_strength",
    "dielectric_constant", "protic_aprotic", "polarity_index",
]


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


def normalize_solvent_name(name: str) -> str:
    # matches source pipelineB.ipynb normalize_solvent_name exactly
    name = " ".join(str(name).split())
    name = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", name)
    name = re.sub(r"(?<=[a-zA-Z])(?=\()", " ", name)
    return name.lower()


def esm_embedding_cached(sequence: str, feature_extractor: FeatureExtractor) -> np.ndarray:
    # Mean-pool cached hidden states excluding BOS/EOS, matching source embed_sequences().
    embedding = feature_extractor.get_esm2_embedding(sequence)
    return embedding.hidden_states[1:-1].mean(axis=0)


class SolubilityPredictor:
    """Lazy-loaded XGBoost classifier over ESM2 embeddings and solvent descriptors returning P(soluble); see README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return

        sync_model_weights(CODE_DIR)
        with open(self.model_dir / "solvent_descriptors.json", "r", encoding="utf-8") as f:
            solvent_data = json.load(f)["solvents"]
        self.solvent_lookup = {row["solvent_key"]: row for row in solvent_data}
        self.solvent_display_names = [row["solvent"] for row in solvent_data]

        model_path = self.model_dir / "pair_stratified_8af6a134fd4c5f76.joblib"
        self.model = joblib.load(model_path)
        self.feature_cols = list(self.model.feature_names_in_)
        self.loaded = True

    def _solvent_features(self, solvent: str) -> dict:
        key = normalize_solvent_name(solvent)
        if key not in self.solvent_lookup:
            raise ValueError(
                f"Unknown solvent {solvent!r}. Known solvents: {self.solvent_display_names}"
            )
        row = self.solvent_lookup[key]
        return {col: (0.0 if row[col] is None else float(row[col])) for col in SOLVENT_COLS}

    def predict_proba(
        self,
        sequence: str,
        solvent: str,
        feature_extractor: FeatureExtractor,
    ) -> float:
        _validate_sequence(sequence)
        self._load()
        solvent_feats = self._solvent_features(solvent)
        embedding = esm_embedding_cached(sequence.upper(), feature_extractor)
        features = {f"esm_{i}": value for i, value in enumerate(embedding)}
        features.update(solvent_feats)
        X = pd.DataFrame([features])[self.feature_cols]
        proba = self.model.predict_proba(X.values)[:, 1]
        return float(np.clip(proba[0], 0.0, 1.0))
