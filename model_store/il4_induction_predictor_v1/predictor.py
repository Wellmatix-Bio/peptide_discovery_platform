"""IL-4 induction predictor, first version: mean of 1 classifier on mean-pooled ESM2-150M features."""

from __future__ import annotations

from pathlib import Path

from common.cytokine_ensemble import CytokineEnsemble
from common.model_sync import weights_dir_for

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)


class Il4InductionPredictor(CytokineEnsemble):
    """P(peptide induces IL-4) in the training assays' context; members: XGBoost_posw."""

    MEMBERS = ('XGBoost_posw',)

    def __init__(self, model_dir: Path = MODEL_DIR):
        super().__init__(CODE_DIR, model_dir)
