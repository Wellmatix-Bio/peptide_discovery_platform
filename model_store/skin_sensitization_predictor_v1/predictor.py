"""Loader/inference wrapper for the skin-sensitization ensemble (RF + LDA + SVM + MLP) on 14 molecular descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from common.model_sync import sync_model_weights, weights_dir_for

from .architecture import N_FEATURES, build_net
from .features import FEATURES, feature_vector, peptide_smiles

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

MEMBERS = ("rf", "lda", "svm", "mlp")

# A peptide is called a sensitizer at or above this ensemble probability. Fixed with the model, not a run parameter.
DECISION_THRESHOLD = 0.55


class SkinSensitizationPredictor:
    """Lazy-loaded equal-weight mean of four sensitization probabilities, see README.md."""

    version = "v1"
    threshold = DECISION_THRESHOLD

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        # Trusted first-party artifacts: joblib unpickles arbitrary objects.
        self.rf = joblib.load(self.model_dir / "RF.joblib")
        self.lda = joblib.load(self.model_dir / "LDA.joblib")
        self.svm = joblib.load(self.model_dir / "SVM.joblib")
        bundle = joblib.load(self.model_dir / "MLP.pt")
        if int(bundle["n_in"]) != N_FEATURES:
            raise ValueError(f"MLP expects {bundle['n_in']} features, not {N_FEATURES}.")
        self.mlp_scaler = bundle["scaler"]
        self.mlp = build_net(int(bundle["n_in"]), float(bundle["p_drop"]))
        self.mlp.load_state_dict(bundle["state"])
        self.mlp.eval()
        self.loaded = True

    def features(self, sequences: list[str], modifications: list[dict[str, Any] | None] | None = None) -> np.ndarray:
        """The 14 descriptors per peptide (see `features.FEATURES`)."""
        mods = modifications or [None] * len(sequences)
        if len(mods) != len(sequences):
            raise ValueError("modifications must have one entry per sequence.")
        rows = []
        for i, (sequence, mod) in enumerate(zip(sequences, mods)):
            try:
                rows.append(feature_vector(peptide_smiles(sequence, mod)))
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        return np.asarray(rows, dtype=float)

    def _member_probabilities(self, x: np.ndarray) -> dict[str, np.ndarray]:
        with torch.no_grad():
            logits = self.mlp(torch.as_tensor(self.mlp_scaler.transform(x), dtype=torch.float32)).squeeze(1).numpy()
        return {
            "rf": self.rf.predict_proba(x)[:, 1],
            "lda": self.lda.predict_proba(x)[:, 1],
            "svm": self.svm.predict_proba(x)[:, 1],
            "mlp": 1.0 / (1.0 + np.exp(-logits)),
        }

    def predict_batch(
        self, sequences: list[str], modifications: list[dict[str, Any] | None] | None = None
    ) -> list[dict]:
        """P(sensitizer) per peptide, the mean of the four members, with each member's value."""
        if not sequences:
            return []
        x = self.features(sequences, modifications)
        self._load()
        members = self._member_probabilities(x)
        mean = np.mean([members[name] for name in MEMBERS], axis=0)
        return [
            {
                "p_sensitizer": float(mean[i]),
                "members": {name: float(members[name][i]) for name in MEMBERS},
            }
            for i in range(len(sequences))
        ]

    def predict_proba_batch(
        self, sequences: list[str], modifications: list[dict[str, Any] | None] | None = None
    ) -> list[float]:
        return [r["p_sensitizer"] for r in self.predict_batch(sequences, modifications)]

    def predict_proba(self, sequence: str, modifications: dict[str, Any] | None = None) -> float:
        return self.predict_proba_batch([sequence], [modifications])[0]
