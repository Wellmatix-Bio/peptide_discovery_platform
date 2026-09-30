"""Loader/inference wrapper for the proliferation/migration VotingClassifier ensembles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import joblib

from common.model_sync import sync_model_weights, weights_dir_for

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity
KD = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# Approximate side-chain pKa values for net charge at pH 7 (Henderson-Hasselbalch)
PKA = {"K": 10.5, "R": 12.5, "H": 6.0, "D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
N_TERM_PKA = 9.0
C_TERM_PKA = 2.0

AROMATIC = set("FWY")
CATIONIC = set("KRH")
ANIONIC = set("DE")
POLAR = set("STNQCY")
ALIPHATIC = set("AVLIMG")

FEATURE_COLS = [
    "length",
    "net_charge",
    "hydrophobicity_mean",
    "hydrophobic_moment",
    "aromatic_fraction",
    "cationic_fraction",
    "anionic_fraction",
    "polar_fraction",
    "aliphatic_fraction",
    "proline_fraction",
    "glycine_fraction",
    "cysteine_fraction",
    "charge_density",
    "isoelectric_proxy",
    "hydrophobic_fraction",
]


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


def net_charge(seq: str, pH: float = 7.0) -> float:
    charge = 1.0 / (1.0 + 10 ** (pH - N_TERM_PKA))
    charge -= 1.0 / (1.0 + 10 ** (C_TERM_PKA - pH))
    for aa in seq:
        if aa in ("K", "R", "H"):
            charge += 1.0 / (1.0 + 10 ** (pH - PKA[aa]))
        elif aa in ("D", "E", "C", "Y"):
            charge -= 1.0 / (1.0 + 10 ** (PKA[aa] - pH))
    return charge


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    # Eisenberg hydrophobic moment assuming an alpha-helix (100 deg/residue)
    angle = np.radians(angle_deg)
    sum_sin = sum(KD[aa] * np.sin(i * angle) for i, aa in enumerate(seq))
    sum_cos = sum(KD[aa] * np.cos(i * angle) for i, aa in enumerate(seq))
    return np.sqrt(sum_sin**2 + sum_cos**2) / len(seq)


def featurize(seq: str) -> dict:
    n = len(seq)
    counts = {aa: seq.count(aa) / n for aa in KD}
    return {
        "length": n,
        "net_charge": net_charge(seq),
        "hydrophobicity_mean": np.mean([KD[aa] for aa in seq]),
        "hydrophobic_moment": hydrophobic_moment(seq),
        "aromatic_fraction": sum(counts[aa] for aa in AROMATIC),
        "cationic_fraction": sum(counts[aa] for aa in CATIONIC),
        "anionic_fraction": sum(counts[aa] for aa in ANIONIC),
        "polar_fraction": sum(counts[aa] for aa in POLAR),
        "aliphatic_fraction": sum(counts[aa] for aa in ALIPHATIC),
        "proline_fraction": counts["P"],
        "glycine_fraction": counts["G"],
        "cysteine_fraction": counts["C"],
        "charge_density": net_charge(seq) / n,
        "isoelectric_proxy": sum(counts[aa] for aa in CATIONIC)
        - sum(counts[aa] for aa in ANIONIC),
        "hydrophobic_fraction": sum(counts[aa] for aa in KD if KD[aa] > 0),
    }


class ProliferationMigrationPredictor:
    """Lazy-loaded pair of VotingClassifier (RF + XGBoost + scaled LogReg) ensembles.
    Returns dominant-mode probabilities for migration and proliferation. See README.md.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.migration_model = joblib.load(
            self.model_dir / "migration_dominant_ensemble.joblib"
        )
        self.proliferation_model = joblib.load(
            self.model_dir / "proliferation_dominant_ensemble.joblib"
        )
        self.loaded = True

    def predict(self, sequence: str) -> dict:
        _validate_sequence(sequence)
        self._load()
        feats = featurize(sequence)
        X = np.array([[feats[col] for col in FEATURE_COLS]], dtype=np.float64)
        migration_proba = float(self.migration_model.predict_proba(X)[0, 1])
        proliferation_proba = float(self.proliferation_model.predict_proba(X)[0, 1])
        return {
            "migration": migration_proba,
            "proliferation": proliferation_proba,
        }
