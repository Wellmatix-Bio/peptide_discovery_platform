"""Loader/inference wrapper for the MBIC SVR+RF ensemble predicting pMBIC = 6 - log10(activity_uM), a log-scale potency (not a probability); see README.md."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import joblib
import torch
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"

PHYSCHEM_COLS = [
    "charge", "charge_density", "gravy", "hydrophobic_moment", "amphipathicity",
    "helical_propensity", "aliphatic_index", "aromaticity", "boman_index",
    "instability_index", "isoelectric_point", "hydrophobic_ratio", "length",
    "d_residue_fraction",
]


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence.upper()):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence.upper()) - AMINO_ACID_SET)}"
        )


def esm_embedding_cached_batch(
    sequences: list[str], feature_extractor: FeatureExtractor
) -> np.ndarray:
    """Mean-pool over cached hidden states excluding BOS/EOS."""
    seqs = [s.strip().upper() for s in sequences]
    embeddings = feature_extractor.get_esm2_embedding_batch(seqs)
    pooled = np.stack([e.hidden_states[1:-1].mean(axis=0) for e in embeddings])
    return pooled.astype(np.float32)


def physchem_descriptors_batch(sequences: list[str], ph: float = 7.4, amide: bool = False, window: int = 18, angle: int = 100) -> list[dict]:
    raws = [s.strip() for s in sequences]
    seqs = [r.upper() for r in raws]
    d_fractions = [
        (sum(1 for c in raw if c.islower()) / len(raw)) if len(raw) else np.nan
        for raw in raws
    ]

    def _global(method, **kw):
        g = GlobalDescriptor(seqs)
        getattr(g, method)(**kw)
        return g.descriptor.ravel()

    def _scaled(scale, method, **kw):
        p = PeptideDescriptor(seqs, scale)
        getattr(p, method)(**kw)
        return p.descriptor.ravel()

    # Eisenberg hydrophobic moment <uH>, max over an alpha-helical (100deg) window.
    moment = _scaled("eisenberg", "calculate_moment", window=window, angle=angle, modality="max")

    # Amphipathicity = <uH> / mean |H|; falls back to mean_h when mean |H| is ~0 to avoid NaN.
    p_abs = PeptideDescriptor(seqs, "eisenberg")
    p_abs.calculate_global(modality="mean")
    mean_h = np.abs(p_abs.descriptor.ravel())
    amphipathicity = np.where(mean_h > 1e-9, moment / np.where(mean_h > 1e-9, mean_h, 1.0), mean_h)

    charge = _global("calculate_charge", ph=ph, amide=amide)
    charge_density = _global("charge_density", ph=ph, amide=amide)
    gravy = _scaled("gravy", "calculate_global", modality="mean")
    helical_propensity = _scaled("levitt_alpha", "calculate_global", modality="mean")
    aliphatic_index = _global("aliphatic_index")
    aromaticity = _global("aromaticity")
    boman_index = _global("boman_index")
    instability_index = _global("instability_index")
    isoelectric_point = _global("isoelectric_point")
    hydrophobic_ratio = _global("hydrophobic_ratio")
    length = _global("length")

    return [
        {
            "charge": float(charge[i]),
            "charge_density": float(charge_density[i]),
            "gravy": float(gravy[i]),
            "hydrophobic_moment": float(moment[i]),
            "amphipathicity": float(amphipathicity[i]),
            "helical_propensity": float(helical_propensity[i]),
            "aliphatic_index": float(aliphatic_index[i]),
            "aromaticity": float(aromaticity[i]),
            "boman_index": float(boman_index[i]),
            "instability_index": float(instability_index[i]),
            "isoelectric_point": float(isoelectric_point[i]),
            "hydrophobic_ratio": float(hydrophobic_ratio[i]),
            "length": float(length[i]),
            "d_residue_fraction": d_fractions[i],
        }
        for i in range(len(sequences))
    ]


class MBICPredictor:
    """Lazy-loaded SVR + RandomForest ensemble over ESM2, modlAMP physchem and species one-hot returning pMBIC; see README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        svr_bundle = joblib.load(self.model_dir / "svr.joblib")
        rf_bundle = joblib.load(self.model_dir / "random_forest.joblib")
        self.svr_model = svr_bundle["model"]
        self.svr_prep = svr_bundle["preprocessing"]
        self.rf_model = rf_bundle["model"]
        self.rf_prep = rf_bundle["preprocessing"]
        self.known_species = set(self.svr_prep["species_ohe"].categories_[0])
        self.loaded = True

    def _featurize(
        self,
        sequences: list[str],
        species: str,
        feature_extractor: FeatureExtractor,
        prep: dict,
    ) -> np.ndarray:
        physchem_rows = physchem_descriptors_batch(sequences)
        physchem_matrix = pd.DataFrame(physchem_rows)[prep["physchem_cols"]].to_numpy(dtype=np.float32)
        physchem_scaled = prep["physchem_scaler"].transform(physchem_matrix)

        esm2_raw = esm_embedding_cached_batch(sequences, feature_extractor)
        esm2_scaled = prep["esm2_scaler"].transform(esm2_raw)
        esm2_pcs = prep["esm2_pca"].transform(esm2_scaled)

        species_oh = prep["species_ohe"].transform(
            pd.DataFrame([{"target_species": species}] * len(sequences))
        )

        return np.concatenate([species_oh, physchem_scaled, esm2_pcs], axis=1)

    def predict_pmbic(
        self, sequence: str, species: str, feature_extractor: FeatureExtractor
    ) -> float:
        """Predict pMBIC for `sequence` against `species`, averaged over the ensemble; unknown species silently become an all-zero one-hot, so treat them with caution."""
        return self.predict_pmbic_batch([sequence], species, feature_extractor)[0]

    def predict_pmbic_batch(
        self,
        sequences: list[str],
        species: str,
        feature_extractor: FeatureExtractor,
    ) -> list[float]:
        """Batched predict_pmbic: one ESM2 forward pass with one fixed `species` for every sequence."""
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        if not sequences:
            return []
        self._load()

        seqs = [s.strip().upper() for s in sequences]

        X_svr = self._featurize(seqs, species, feature_extractor, self.svr_prep)
        X_rf = self._featurize(seqs, species, feature_extractor, self.rf_prep)

        svr_preds = self.svr_model.predict(X_svr)
        rf_preds = self.rf_model.predict(X_rf)

        return [float((s + r) / 2.0) for s, r in zip(svr_preds, rf_preds)]
