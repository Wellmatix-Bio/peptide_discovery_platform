"""Loader/inference wrapper for the per-pathway PU-bagged RandomForest classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

from common.model_sync import sync_model_weights, weights_dir_for

# weights_dir_for takes this predictor's own directory (its name is the GCS
# model key); the actual weights live one level down, in model/.
CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR) / "model"

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

LABELS = [
    "MAPK", "ERK", "PI3K_AKT_MTOR", "TGFB_SMAD", "NF_KB", "WNT_BCATENIN",
    "FGFR_JAK2_STAT3", "VEGF_ANGIOGENESIS", "CYTOKINE_MACROPHAGE",
]

PHYSCHEM_COLUMNS = [
    "MW", "Charge", "ChargeDensity", "pI", "InstabilityInd",
    "Aromaticity", "AliphaticInd", "BomanInd", "HydrophRatio",
    "Hydrophobicity", "HydrophobicMoment",
]


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


def compute_physicochemical_descriptors(sequence: str) -> pd.DataFrame:
    """The 11-column PHYSCHEM_COLUMNS feature row, exactly as computed in the source notebook."""
    seqs = [sequence]

    glob = GlobalDescriptor(seqs)
    glob.calculate_all(amide=False)
    glob_df = pd.DataFrame(glob.descriptor, columns=glob.featurenames).drop(columns=["Length"])

    hydrophobicity = PeptideDescriptor(seqs, scalename="Eisenberg")
    hydrophobicity.calculate_global(append=False)
    hydrophobic_moment = PeptideDescriptor(seqs, scalename="Eisenberg")
    hydrophobic_moment.calculate_moment(append=False)

    return glob_df.assign(
        Hydrophobicity=hydrophobicity.descriptor[:, 0],
        HydrophobicMoment=hydrophobic_moment.descriptor[:, 0],
    )[PHYSCHEM_COLUMNS]


class PathwayMappingPredictor:
    """Lazy-loaded per-label PU-bagged RandomForest ensemble (100 bags/label, 9 labels)."""

    def __init__(self, model_dir: Path = MODEL_DIR, labels: list[str] = LABELS):
        self.model_dir = Path(model_dir)
        self.labels = labels
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.bags_by_label: dict[str, list] = {}
        for label in self.labels:
            label_dir = self.model_dir / label
            bag_paths = sorted(label_dir.glob("RandomForestClassifier_k*.joblib"))
            if not bag_paths:
                raise FileNotFoundError(f"No RandomForest bags found for label {label!r} in {label_dir}")
            self.bags_by_label[label] = [joblib.load(path) for path in bag_paths]
        self.loaded = True

    def predict(self, sequence: str) -> dict[str, dict]:
        """Runs every label's RF bag ensemble on `sequence`; returns {label: {"probability", "label_predicted", "n_bags"}}."""
        _validate_sequence(sequence)
        self._load()
        features = compute_physicochemical_descriptors(sequence)

        results = {}
        for label, bags in self.bags_by_label.items():
            bag_probs = [bag.predict_proba(features)[:, 1][0] for bag in bags]
            probability = float(np.mean(bag_probs))
            results[label] = {
                "probability": probability,
                "label_predicted": probability >= 0.5,
                "n_bags": len(bags),
            }
        return results
