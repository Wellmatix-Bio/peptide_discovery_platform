"""Loader/inference wrapper for the aggregation-propensity XGBoost classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from propy.PyPro import GetProDes

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

AROMATIC = set("FWY")
HYDROPHOBIC = set("AVLIMFW")
POLAR = set("STNQCY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
SMALL = set("AGS")

SOCN_QSO_LAG = 4
PAAC_LAMBDA = 4
_PAAC_KEYS = [f"PAAC{i}" for i in range(1, 21 + PAAC_LAMBDA)]
_APAAC_KEYS = [f"APAAC{i}" for i in range(1, 21 + 2 * PAAC_LAMBDA)]


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )
    if len(sequence) < 5:
        raise ValueError("sequence must be at least 5 residues (QSO/SOCN/PAAC/APAAC lag requirement).")


def biopython_features(seq: str) -> dict:
    pa = ProteinAnalysis(seq)
    helix, turn, sheet = pa.secondary_structure_fraction()
    ext_reduced, ext_oxidized = pa.molar_extinction_coefficient()
    n = len(seq)
    return {
        "bp_length": n,
        "bp_molecular_weight": pa.molecular_weight(),
        "bp_aromaticity": pa.aromaticity(),
        "bp_instability_index": pa.instability_index(),
        "bp_gravy": pa.gravy(),
        "bp_isoelectric_point": pa.isoelectric_point(),
        "bp_charge_at_pH7": pa.charge_at_pH(7.0),
        "bp_ss_helix_frac": helix,
        "bp_ss_turn_frac": turn,
        "bp_ss_sheet_frac": sheet,
        "bp_molar_ext_reduced": ext_reduced,
        "bp_molar_ext_oxidized": ext_oxidized,
        "bp_aromatic_frac": sum(c in AROMATIC for c in seq) / n,
        "bp_hydrophobic_frac": sum(c in HYDROPHOBIC for c in seq) / n,
        "bp_polar_frac": sum(c in POLAR for c in seq) / n,
        "bp_positive_frac": sum(c in POSITIVE for c in seq) / n,
        "bp_negative_frac": sum(c in NEGATIVE for c in seq) / n,
        "bp_small_frac": sum(c in SMALL for c in seq) / n,
        "bp_net_charge_frac": (sum(c in POSITIVE for c in seq) - sum(c in NEGATIVE for c in seq)) / n,
    }


def _safe_descriptor(fn, fallback_keys, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except Exception:
        return {k: np.nan for k in fallback_keys}


def pybiomed_features(seq: str) -> dict:
    gp = GetProDes(seq)
    feats = {}
    feats.update(gp.GetAAComp())
    feats.update(gp.GetDPComp())
    feats.update(gp.GetCTD())
    feats.update(gp.GetMoranAuto())
    feats.update(gp.GetGearyAuto())
    feats.update(gp.GetMoreauBrotoAuto())
    feats.update(gp.GetQSO(maxlag=SOCN_QSO_LAG))
    feats.update(gp.GetSOCN(maxlag=SOCN_QSO_LAG))
    feats.update(_safe_descriptor(gp.GetPAAC, _PAAC_KEYS, lamda=PAAC_LAMBDA))
    feats.update(_safe_descriptor(gp.GetAPAAC, _APAAC_KEYS, lamda=PAAC_LAMBDA))
    return feats


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
    """Lazy-loaded XGBoost classifier over biopython/propy/AAindex descriptors.
    Returns an aggregation-propensity probability in [0, 1]. See README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR, use_feature_cache: bool = False):
        self.model_dir = Path(model_dir)
        self.use_feature_cache = use_feature_cache
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
        self, sequence: str, feature_extractor: FeatureExtractor | None = None
    ) -> float:
        _validate_sequence(sequence)
        self._load()
        seq = sequence.upper()
        feats = {}
        if self.use_feature_cache:
            feats.update(feature_extractor.get_propy_biopython_descriptors(seq))
        else:
            feats.update(biopython_features(seq))
            feats.update(pybiomed_features(seq))
        feats.update(protscale_features(seq, self.aaindex_scales))
        row = pd.DataFrame([feats])
        missing = [col for col in self.feature_cols if col not in row.columns]
        if missing:
            raise ValueError(f"Missing aggregation model features: {missing[:10]}")
        X = row[self.feature_cols]
        proba = self.model.predict_proba(X)[:, 1]
        return float(np.clip(proba[0], 0.0, 1.0))
