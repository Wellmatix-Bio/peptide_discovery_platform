# Cosmetic primary-function scoring shared by Stage 4 (GA decision) and Stage 6 (function gates): one cached model per function.
from __future__ import annotations

from typing import Any, Callable

from pipeline.feature_extractor import FeatureExtractor

FUNCTIONS = ("collagen_ecm", "antioxidant", "pigmentation")

_models: dict[str, Any] = {}


def _load(function: str) -> Callable[[list[str], FeatureExtractor], list[float]]:
    """Batch scorer for one function; the model is built on first use and kept."""
    if function == "collagen_ecm":
        from model_store.collagen_ecm_predictor_v1 import CollagenECMPredictor

        model = _models.setdefault(function, CollagenECMPredictor())
        return model.predict_proba_batch
    if function == "antioxidant":
        from model_store.antioxidant_predictor_v1 import AntioxidantPredictor

        model = _models.setdefault(function, AntioxidantPredictor())
        return lambda sequences, fe: model.predict_proba_batch(sequences)
    if function == "pigmentation":
        from model_store.pigmentation_predictor_v1 import PigmentationPredictor

        model = _models.setdefault(function, PigmentationPredictor())
        return model.predict_proba_batch
    raise ValueError(f"unsupported cosmetic function {function!r}; expected one of {FUNCTIONS}")


def score_function(
    function: str, sequences: list[str], feature_extractor: FeatureExtractor
) -> list[float]:
    """The function's model output per sequence: a ranking score (collagen_ecm) or a probability (antioxidant, pigmentation)."""
    if not sequences:
        return []
    return [float(v) for v in _load(function)(sequences, feature_extractor)]
