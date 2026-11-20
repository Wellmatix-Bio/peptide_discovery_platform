"""Loader/inference wrapper for the peptide allergenicity model (IgE-binding peptides); see README.md."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

from .architecture import EMBED_DIM, MLP, AllergenCNN, one_hot

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

# Training peptides were 3-49 residues; the wrapper refuses anything outside that range.
TRAINED_MIN_LENGTH = 3
MAX_LENGTH = 49

N_FOLDS = 5


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    bad = sorted(set(sequence) - AMINO_ACID_SET)
    if bad:
        raise ValueError(
            f"sequence contains residues outside the 20 uppercase standard amino acids: {bad}"
        )
    if not TRAINED_MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
        raise ValueError(
            f"sequence length {len(sequence)} outside the trained range {TRAINED_MIN_LENGTH}-{MAX_LENGTH}."
        )


def esm_mean_embeddings(
    sequences: list[str], feature_extractor: FeatureExtractor
) -> np.ndarray:
    """Mean over residue positions of the last hidden layer with BOS/EOS excluded: this model's trained convention."""
    embeddings = feature_extractor.get_esm2_embedding_batch(sequences)
    return np.stack([e.hidden_states[1:-1].mean(axis=0) for e in embeddings])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class AllergenicityPredictor:
    """Lazy-loaded fold models returning a score and P(allergen) = sigmoid(score); the score is 0 at the decision boundary."""

    version = "v1"

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta = json.loads((self.model_dir / "ensemble.json").read_text(encoding="utf-8"))
        self.scale = meta["scale"]
        # Trusted first-party artifacts: joblib unpickles arbitrary objects.
        self.mlps = []
        for f in range(N_FOLDS):
            bundle = joblib.load(self.model_dir / f"mlp_fold{f}.joblib")
            if int(bundle["d_in"]) != EMBED_DIM:
                raise ValueError(f"expects {bundle['d_in']}-d embeddings, not {EMBED_DIM}.")
            net = MLP(EMBED_DIM).to(self.device)
            net.load_state_dict(bundle["state"])
            self.mlps.append((bundle["scaler"], net.eval()))
        self.cnns = []
        for f in range(N_FOLDS):
            bundle = torch.load(self.model_dir / f"cnn_fold{f}.pt", map_location=self.device, weights_only=True)
            net = AllergenCNN().to(self.device)
            net.load_state_dict(bundle["state"])
            self.cnns.append(net.eval())
        self.loaded = True

    def _scores(self, sequences: list[str], embeddings: np.ndarray) -> np.ndarray:
        """Each member family's fold-mean raw score divided by its training-time spread, then averaged; 0 is the boundary."""
        with torch.no_grad():
            a = np.mean(
                [
                    net(torch.tensor(scaler.transform(embeddings), dtype=torch.float32, device=self.device)).cpu().numpy()
                    for scaler, net in self.mlps
                ],
                axis=0,
            )
            x = one_hot(sequences, self.device)
            logits = [net(x) for net in self.cnns]
            b = np.mean([(lg[:, 1] - lg[:, 0]).cpu().numpy() for lg in logits], axis=0)
        return np.mean([a / self.scale["MLP"], b / self.scale["CNN"]], axis=0)

    def predict_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> list[dict]:
        """Per sequence: `score` (0 = decision boundary) and `p_allergen` = sigmoid(score), an uncalibrated 0-1 value, 0.5 at the boundary."""
        if not sequences:
            return []
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        self._load()
        scores = self._scores(sequences, esm_mean_embeddings(sequences, feature_extractor))
        return [
            {"score": float(s), "p_allergen": float(p)}
            for s, p in zip(scores, _sigmoid(scores))
        ]

    def predict_proba_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> list[float]:
        return [r["p_allergen"] for r in self.predict_batch(sequences, feature_extractor)]

    def predict_proba(self, sequence: str, feature_extractor: FeatureExtractor) -> float:
        return self.predict_proba_batch([sequence], feature_extractor)[0]
