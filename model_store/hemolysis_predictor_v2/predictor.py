"""Hemolysis ensemble (multi-task network + SVR + SVM on frozen ESM2-150M embeddings): pHC50 regression and P(hemolytic) classification."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch

from common.model_sync import sync_model_weights, weights_dir_for

from .architecture import HEMOLYTIC_PHC50, LocalGlobalMultiTaskPredictor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

#: P(hemolytic) above this means HC50 < 100 uM; the ensemble was evaluated at this threshold.
HEMOLYTIC_PROBABILITY_THRESHOLD = 0.5
#: Longest peptide in the training data; longer inputs are extrapolation.
MAX_TRAIN_LENGTH = 100

_STANDARD = re.compile("[ACDEFGHIKLMNPQRSTVWY]+")


def _validate(sequences: Sequence[str]) -> None:
    bad = [s for s in sequences if not _STANDARD.fullmatch(s)]
    if bad:
        raise ValueError(
            f"{len(bad)} sequence(s) are empty or have non-standard residues, e.g. {bad[0]!r}"
        )


class HemolysisPredictor:
    """Lazy-loaded ensemble; `predict_phc50_batch` is the regression (ranking) output, `predict_hemolytic_proba_batch` the classifier (filtering) output."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.network = LocalGlobalMultiTaskPredictor.load(self.model_dir / "network.pt")
        classical = joblib.load(self.model_dir / "svm_svr.joblib")
        self.svr, self.svm = classical["svr"], classical["svm"]
        self.loaded = True

    @staticmethod
    def _residue_embeddings(sequences: list[str], feature_extractor) -> list[torch.Tensor]:
        """(L, 640) per sequence from the shared ESM2 extractor, without BOS/EOS."""
        embeddings = feature_extractor.get_esm2_embedding_batch(sequences)
        return [torch.from_numpy(e.hidden_states[1:-1]).float() for e in embeddings]

    def predict_batch(self, sequences: list[str], feature_extractor) -> list[dict]:
        """Per sequence: ensemble `phc50` (mean of SVR and network), ensemble `p_hemolytic` (mean of SVM and network), and `hc50_um`."""
        if not sequences:
            return []
        sequences = [s.strip().upper() for s in sequences]
        _validate(sequences)
        self._load()
        residues = self._residue_embeddings(sequences, feature_extractor)
        pooled = np.stack([r.mean(0).numpy() for r in residues])
        phc50_net, p_net = self.network.predict_both(residues).T
        phc50_svr = self.svr.predict(pooled)
        p_svm = np.asarray(self.svm.predict_proba(pooled))[:, 1]
        phc50 = (phc50_svr + phc50_net) / 2
        p = (p_svm + p_net) / 2
        return [
            {
                "phc50": float(a),
                "p_hemolytic": float(b),
                "hc50_um": float(10.0 ** (6.0 - a)),
                "phc50_svr": float(c),
                "phc50_network": float(d),
                "p_svm": float(e),
                "p_network": float(f),
            }
            for a, b, c, d, e, f in zip(phc50, p, phc50_svr, phc50_net, p_svm, p_net)
        ]

    def predict_phc50_batch(self, sequences: list[str], feature_extractor) -> list[float]:
        """Ensemble pHC50 = 6 - log10(HC50 in uM); higher = more hemolytic. Use for ranking."""
        return [r["phc50"] for r in self.predict_batch(sequences, feature_extractor)]

    def predict_hemolytic_proba_batch(self, sequences: list[str], feature_extractor) -> list[float]:
        """Ensemble P(HC50 < 100 uM). Use for filtering."""
        return [r["p_hemolytic"] for r in self.predict_batch(sequences, feature_extractor)]
