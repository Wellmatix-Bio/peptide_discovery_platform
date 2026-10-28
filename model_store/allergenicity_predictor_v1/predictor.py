"""Loader/inference wrapper for the allergenicity ensemble (SVM + LDA + MLP) on mean-pooled ESM2-150M embeddings."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

from .architecture import EMBED_DIM, MLP

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

# ESM2's context limit; the training notebook truncated here, this wrapper refuses instead.
MAX_LENGTH = 1022
# Shortest sequence in the training data (AlgPred 2.0 full-length proteins); nothing shorter was ever seen.
TRAINED_MIN_LENGTH = 50

MEMBERS = ("svm", "lda", "mlp")


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    bad = sorted(set(sequence) - AMINO_ACID_SET)
    if bad:
        raise ValueError(
            f"sequence contains residues outside the 20 uppercase standard amino acids: {bad}"
        )
    if len(sequence) > MAX_LENGTH:
        raise ValueError(f"sequence length {len(sequence)} exceeds the ESM2 limit of {MAX_LENGTH}.")


def esm_mean_embeddings(
    sequences: list[str], feature_extractor: FeatureExtractor
) -> np.ndarray:
    """Mean over residue positions of the last hidden layer with BOS/EOS excluded: this model's trained convention."""
    embeddings = feature_extractor.get_esm2_embedding_batch(sequences)
    return np.stack([e.hidden_states[1:-1].mean(axis=0) for e in embeddings])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class AllergenicityPredictor:
    """Lazy-loaded equal-weight mean of three allergen probabilities, see README.md."""

    version = "v1"

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Trusted first-party artifacts: joblib unpickles arbitrary objects.
        self.svm = joblib.load(self.model_dir / "SVM.joblib")
        self.lda = joblib.load(self.model_dir / "LDA.joblib")
        bundle = joblib.load(self.model_dir / "MLP.pt")
        self.mlp_scaler = bundle["scaler"]
        self.mlp = MLP(int(bundle["d_in"])).to(self.device)
        self.mlp.load_state_dict(bundle["state"])
        self.mlp.eval()
        if int(bundle["d_in"]) != EMBED_DIM:
            raise ValueError(f"MLP expects {bundle['d_in']}-d embeddings, not {EMBED_DIM}.")
        self.loaded = True

    def _member_probabilities(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        """Each member's P(allergen); 0.5 is that member's own class boundary (SVM margin 0, MLP logit 0, LDA posterior 0.5)."""
        with torch.no_grad():
            x = torch.tensor(self.mlp_scaler.transform(embeddings), dtype=torch.float32, device=self.device)
            mlp_logit = self.mlp(x).cpu().numpy()
        return {
            "svm": _sigmoid(self.svm.decision_function(embeddings)),
            "lda": self.lda.predict_proba(embeddings)[:, 1],
            "mlp": _sigmoid(mlp_logit),
        }

    def predict_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> list[dict]:
        """P(allergen) per sequence, the mean of the three members, with each member's value."""
        if not sequences:
            return []
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        self._load()
        members = self._member_probabilities(esm_mean_embeddings(sequences, feature_extractor))
        mean = np.mean([members[name] for name in MEMBERS], axis=0)
        return [
            {
                "p_allergen": float(mean[i]),
                "members": {name: float(members[name][i]) for name in MEMBERS},
            }
            for i in range(len(sequences))
        ]

    def predict_proba_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> list[float]:
        return [r["p_allergen"] for r in self.predict_batch(sequences, feature_extractor)]

    def predict_proba(self, sequence: str, feature_extractor: FeatureExtractor) -> float:
        return self.predict_proba_batch([sequence], feature_extractor)[0]
