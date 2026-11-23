"""The self-contained run config for an immunomodulation studio job; shared by the API that writes it and the worker that reads it."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


#: The studio's six endpoints, in display order: cytokine -> (model_store package, predictor class).
CYTOKINE_MODELS = {
    "TNF-alpha": ("tnf_alpha_induction_predictor_v1", "TnfAlphaInductionPredictor"),
    "IL-6": ("il6_induction_predictor_v1", "Il6InductionPredictor"),
    "IL-10": ("il10_induction_predictor_v1", "Il10InductionPredictor"),
    "IL-4": ("il4_induction_predictor_v1", "Il4InductionPredictor"),
    "IL-13": ("il13_induction_predictor_v1", "Il13InductionPredictor"),
    "IFN-gamma": ("ifn_gamma_induction_predictor_v1", "IfnGammaInductionPredictor"),
}

#: One file per weight folder that the worker needs in the model store, relative to it. Checked at job creation, so a
#: bucket missing a model fails the request instead of failing a GPU job minutes after it starts.
REQUIRED_WEIGHT_FILES = [f"{package}/info.json" for package, _ in CYTOKINE_MODELS.values()] + ["esm2_t30_150M/config.json"]


#: One run may hold up to this many peptides; the worker scores them in batches of `batch_size` (the API's IMMUNO_BATCH_SIZE overrides the default).
IMMUNO_MAX_PEPTIDES = 10000
DEFAULT_BATCH_SIZE = 100


class ImmunoRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1)
    artifacts_dir: str
    model_store: str
    peptides: list[str] = Field(min_length=1)  # already parsed and validated by common.sequence_utils
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1)  # peptides the models process at a time
