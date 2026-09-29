"""Loader/inference wrapper for the AMP classifier ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import joblib
import torch
import xgboost as xgb
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")


def esm_features_cached(sequence: str, feature_extractor: FeatureExtractor, normalize_amino: bool) -> pd.DataFrame:
    return esm_features_cached_batch([sequence], feature_extractor, normalize_amino)


def esm_features_cached_batch(
    sequences: list[str], feature_extractor: FeatureExtractor, normalize_amino: bool
) -> pd.DataFrame:
    """Mean-pool over every cached token INCLUDING BOS/EOS -- this model's
    trained convention (verified: its pooling never zeroes special-token
    positions, unlike mbic/angiogenic/solubility/synthesis_feasibility)."""
    model_sequences = [s.upper() if normalize_amino else s for s in sequences]
    embeddings = feature_extractor.get_esm2_embedding_batch(model_sequences)
    values = np.stack([e.hidden_states.mean(axis=0) for e in embeddings])
    return pd.DataFrame(values, columns=[f"esm2_{i}" for i in range(values.shape[1])])


def modlamp_features(sequence: str) -> pd.DataFrame:
    return modlamp_features_batch([sequence])


def modlamp_features_batch(sequences: list[str]) -> pd.DataFrame:
    seqs = [s.upper() for s in sequences]
    gd = GlobalDescriptor(seqs)
    gd.calculate_all(amide=True)
    global_feats = pd.DataFrame(gd.descriptor, columns=[f"modlamp_{name}" for name in gd.featurenames])

    scalar_scales = ["eisenberg", "gravy", "flexibility", "aasi"]
    multidim_scales = ["z3", "z5", "abhprk"]
    scale_feats = {}
    for scale in scalar_scales:
        pdesc = PeptideDescriptor(seqs, scale)
        pdesc.calculate_global()
        scale_feats[f"modlamp_{scale}_global"] = pdesc.descriptor.flatten()
        pdesc_m = PeptideDescriptor(seqs, scale)
        pdesc_m.calculate_moment()
        scale_feats[f"modlamp_{scale}_moment"] = pdesc_m.descriptor.flatten()
    for scale in multidim_scales:
        pdesc = PeptideDescriptor(seqs, scale)
        pdesc.calculate_global()
        scale_feats[f"modlamp_{scale}_global"] = pdesc.descriptor.flatten()

    return pd.concat([global_feats, pd.DataFrame(scale_feats)], axis=1)


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


class AMPClassifier:
    """Lazy-loaded 5-fold XGBoost + meta-model ensemble. Returns a calibrated
    AMP probability in [0, 1]. See README.md for architecture details."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return

        sync_model_weights(CODE_DIR)
        with open(self.model_dir / "model_card.json", "r", encoding="utf-8") as f:
            self.model_card = json.load(f)
        self.feature_cols = self.model_card["feature_cols"]
        self.normalize_amino = bool(self.model_card.get("normalize_amino", True))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.fold_models = []
        for i in range(int(self.model_card["n_folds"])):
            model = xgb.XGBClassifier()
            model.load_model(self.model_dir / f"xgb_fold{i}.json")
            if hasattr(model, "set_params"):
                model.set_params(device=self.device)
            self.fold_models.append(model)
        self.meta_model = joblib.load(self.model_dir / "meta_model.joblib")
        self.loaded = True

    def predict_proba(self, sequence: str, feature_extractor: FeatureExtractor) -> float:
        _validate_sequence(sequence)
        self._load()
        esm_feats = esm_features_cached(sequence, feature_extractor, self.normalize_amino)
        features = pd.concat([esm_feats, modlamp_features(sequence)], axis=1)
        missing = [col for col in self.feature_cols if col not in features.columns]
        if missing:
            raise ValueError(f"Missing AMP model features: {missing[:10]}")
        X = features[self.feature_cols].values
        raw = np.column_stack(
            [self._predict_fold_proba(model, X) for model in self.fold_models]
        ).mean(axis=1)
        calibrated = self.meta_model.predict_proba(raw.reshape(-1, 1))[:, 1]
        return float(np.clip(calibrated[0], 0.0, 1.0))

    def predict_proba_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> list[float]:
        """Batched predict_proba: one ESM2 forward pass for all sequences."""
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        if not sequences:
            return []
        self._load()
        esm_feats = esm_features_cached_batch(sequences, feature_extractor, self.normalize_amino)
        features = pd.concat([esm_feats, modlamp_features_batch(sequences)], axis=1)
        missing = [col for col in self.feature_cols if col not in features.columns]
        if missing:
            raise ValueError(f"Missing AMP model features: {missing[:10]}")
        X = features[self.feature_cols].values
        raw = np.column_stack(
            [self._predict_fold_proba(model, X) for model in self.fold_models]
        ).mean(axis=1)
        calibrated = self.meta_model.predict_proba(raw.reshape(-1, 1))[:, 1]
        return [float(v) for v in np.clip(calibrated, 0.0, 1.0)]

    def _predict_fold_proba(self, model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
        """P(class=1) for one fold model. Builds a DMatrix explicitly instead
        of calling model.predict_proba(X) directly: when the booster is
        device="cuda" (this env has a GPU) and X is a plain host-memory
        ndarray, predict_proba's inplace-predict path detects the mismatch
        and silently falls back to a DMatrix internally anyway, but not
        before emitting a UserWarning on every call. Doing the DMatrix
        construction ourselves gets XGBoost's own CPU<->GPU handling with no
        warning.

        iteration_range must be capped at best_iteration + 1 to match
        predict_proba(): these fold models were trained with early stopping
        (best_iteration < total boosted rounds), and the sklearn wrapper
        silently restricts prediction to the best iteration -- the raw
        booster does not, and calling booster.predict() without this range
        verifiably returns different (wrong) probabilities from every round
        past the early-stopping point."""
        booster = model.get_booster()
        best_iteration = getattr(model, "best_iteration", None)
        iteration_range = (
            (0, best_iteration + 1) if best_iteration is not None else None
        )
        return booster.predict(xgb.DMatrix(X), iteration_range=iteration_range)
