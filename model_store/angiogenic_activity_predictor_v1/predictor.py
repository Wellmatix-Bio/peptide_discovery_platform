"""Loader/inference wrapper for the angiogenic-activity SVM+RF+MLP ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM_EMBED_DIM = 1280

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

# Boman (1995) solubility / protein-binding potential scale (kcal/mol), free energy
# of side-chain transfer from cyclohexane to water. Boman Index = mean over residues.
BOMAN = {
    "A": -0.17,
    "R": -0.81,
    "N": -0.42,
    "D": -1.23,
    "C": -0.24,
    "Q": -0.58,
    "E": -2.02,
    "G": -0.01,
    "H": -0.96,
    "I": -0.31,
    "L": -0.56,
    "K": -0.99,
    "M": -0.23,
    "F": -1.13,
    "P": -0.45,
    "S": -0.13,
    "T": -0.14,
    "W": -1.85,
    "Y": -0.94,
    "V": -0.13,
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
    "boman_index",
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

# MLPNet/TorchMLPClassifier must stay importable under this module path — the
# .joblib artifact was pickled against these exact class definitions.
MLP_HIDDEN1 = 32
MLP_HIDDEN2 = 16


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


def boman_index(seq: str) -> float:
    # Negated mean, per convention: higher = more protein-binding potential.
    return -np.mean([BOMAN[aa] for aa in seq])


def featurize(seq: str) -> dict:
    n = len(seq)
    counts = {aa: seq.count(aa) / n for aa in KD}
    return {
        "length": n,
        "net_charge": net_charge(seq),
        "hydrophobicity_mean": np.mean([KD[aa] for aa in seq]),
        "hydrophobic_moment": hydrophobic_moment(seq),
        "boman_index": boman_index(seq),
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


def esm_embedding(sequence: str, tokenizer, esm_model, device: str) -> np.ndarray:
    # mean-pool per-residue hidden states, excluding BOS/EOS, matching source pipeline.ipynb
    with torch.no_grad():
        encoded = tokenizer([sequence], return_tensors="pt", padding=True).to(device)
        hidden_states = esm_model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].clone()
        seq_lengths = mask.sum(dim=1)
        for i, length in enumerate(seq_lengths):
            mask[i, 0] = 0
            mask[i, length - 1] = 0
        mask = mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.cpu().numpy()[0]


def esm_embedding_cached(sequence: str, feature_extractor: FeatureExtractor) -> np.ndarray:
    # mean-pool cached per-token hidden states, excluding BOS/EOS (positions 0
    # and -1 of the cache's already-trimmed tensor), matching source pipeline.ipynb
    embedding = feature_extractor.get_esm2_embedding(sequence)
    return embedding.hidden_states[1:-1].mean(axis=0)


class MLPNet(nn.Module):
    """3 layers deep: hidden1 -> hidden2 -> output(2), softmax at the end."""

    def __init__(
        self, n_features, hidden1=MLP_HIDDEN1, hidden2=MLP_HIDDEN2, n_classes=2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around MLPNet. Applies the same ESM2-block PCA
    reduction as the SVM/RF pipelines (via the fitted esm_pca_/scaler_ attributes
    restored from the pickle) before feeding the network."""

    def __init__(
        self,
        hidden1=MLP_HIDDEN1,
        hidden2=MLP_HIDDEN2,
        lr=1e-3,
        weight_decay=1e-3,
        epochs=300,
        pos_class_weight=5,
        seed=0,
    ):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.pos_class_weight = pos_class_weight
        self.seed = seed

    def predict_proba(self, X):
        self.model_.eval()
        X_reduced = self.esm_pca_.transform(X)
        X_scaled = self.scaler_.transform(X_reduced)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model_(X_t).numpy()
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class AngiogenicActivityPredictor:
    """Lazy-loaded 3-model ensemble (SVM, RandomForest, MLP) over 16 physicochemical
    descriptors + PCA-reduced ESM2 embeddings. Returns per-model and averaged
    P(angiogenic-dominant) in [0, 1]. See README.md for architecture and provenance."""

    def __init__(self, model_dir: Path = MODEL_DIR, use_feature_cache: bool = False):
        self.model_dir = Path(model_dir)
        self.use_feature_cache = use_feature_cache
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        if not self.use_feature_cache:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
            self.esm_model = (
                AutoModel.from_pretrained(ESM_MODEL_NAME).to(self.device).eval()
            )
        self.svm_model = joblib.load(self.model_dir / "production_svm.joblib")
        self.rf_model = joblib.load(self.model_dir / "production_rf.joblib")
        self.mlp_model = joblib.load(self.model_dir / "production_mlp.joblib")
        self.loaded = True

    def predict(
        self, sequence: str, feature_extractor: FeatureExtractor | None = None
    ) -> dict:
        _validate_sequence(sequence)
        self._load()

        feats = featurize(sequence)
        descriptor_vec = np.array(
            [[feats[col] for col in FEATURE_COLS]], dtype=np.float64
        )
        embedding = (
            esm_embedding_cached(sequence, feature_extractor)
            if self.use_feature_cache
            else esm_embedding(sequence, self.tokenizer, self.esm_model, self.device)
        )
        X = np.concatenate([descriptor_vec, embedding.reshape(1, -1)], axis=1)

        svm_proba = float(self.svm_model.predict_proba(X)[0, 1])
        rf_proba = float(self.rf_model.predict_proba(X)[0, 1])
        mlp_proba = float(self.mlp_model.predict_proba(X)[0, 1])
        mean_proba = (svm_proba + rf_proba + mlp_proba) / 3.0

        return {
            "angiogenic": mean_proba,
            "svm": svm_proba,
            "rf": rf_proba,
            "mlp": mlp_proba,
        }
