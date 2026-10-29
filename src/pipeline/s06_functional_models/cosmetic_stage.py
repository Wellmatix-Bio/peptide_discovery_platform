# Stage 6 (cosmetics): primary-function gate and antimicrobial reject filter; secondary functions are scored and recorded.
from __future__ import annotations

from typing import Any

from common.logging import get_logger
from common.model_registry import ModelRef
from pipeline.base import CandidateStage, RunContext, StageError
from pipeline.cosmetic_functions import FUNCTIONS, score_function
from pipeline.s06_functional_models.stage import _get_amp_model
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage6CosmeticParams

logger = get_logger(__name__)

MODEL_BY_FUNCTION = {
    "collagen_ecm": "collagen_ecm_predictor",
    "antioxidant": "antioxidant_predictor",
    "pigmentation": "pigmentation_predictor",
}


def _functions(params: Stage6CosmeticParams, ctx: RunContext) -> tuple[str, list[str]]:
    """Primary and secondary functions: stage params win, the cosmetic brief fills what they leave unset."""
    brief_functions = getattr(ctx.brief, "functions", None)
    primary = params.primary_function or getattr(brief_functions, "primary", None)
    secondary = params.secondary_functions
    if secondary is None:
        secondary = list(getattr(brief_functions, "secondary", None) or [])
    return primary, [f for f in dict.fromkeys(secondary) if f != primary]


class CosmeticStage6(CandidateStage):
    """Opt-in; replaces the wound-healing functional models for a cosmetic run."""

    name = "s06_cosmetic_functional_models"
    produces = {"sequence"}

    def models_used(self) -> list[ModelRef]:
        return [ModelRef(name="amp_classifier", version="v1")] + [
            ModelRef(name=name, version="v1") for name in MODEL_BY_FUNCTION.values()
        ]

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        params = Stage6CosmeticParams.model_validate(config.params)
        primary, secondary = _functions(params, ctx)
        if primary not in FUNCTIONS:
            raise StageError(
                self.name,
                f"primary function {primary!r} is not one of {FUNCTIONS}; set it in the stage params or the cosmetic brief.",
            )
        if any(not c.sequence for c in candidates):
            raise StageError(self.name, "received a candidate with no sequence")
        thresholds = {f: getattr(params, f"{f}_threshold") for f in FUNCTIONS}
        fe = ctx.feature_extractor

        def gate(function: str, pool: list[Candidate]) -> list[dict[str, Any]]:
            scores = score_function(function, [c.sequence for c in pool], fe)
            return [
                {"score": s, "threshold": thresholds[function], "pass": s >= thresholds[function]}
                for s in scores
            ]

        # A. Primary function gate.
        results = gate(primary, candidates)
        standing: list[Candidate] = []
        for candidate, result in zip(candidates, results):
            candidate.predictions["functional_scores"] = {primary: result}
            candidate.predictions["primary_function_pass"] = result["pass"]
            if result["pass"]:
                standing.append(candidate)
        rejected = {"primary_function_below_threshold": len(candidates) - len(standing)}

        # B. Antimicrobial reject filter, on the candidates still standing.
        survivors: list[Candidate] = []
        if standing:
            probabilities = _get_amp_model().predict_proba_batch(
                [c.sequence for c in standing], fe
            )
            for candidate, probability in zip(standing, probabilities):
                candidate.predictions["antimicrobial"] = {
                    "probability": probability,
                    "reject_min": params.antimicrobial_reject_min,
                }
                if probability >= params.antimicrobial_reject_min:
                    continue
                survivors.append(candidate)
        rejected["antimicrobial"] = len(standing) - len(survivors)

        # Secondary functions are listed in the brief but never reject; they are scored for the record.
        for function in secondary:
            for candidate, result in zip(survivors, gate(function, survivors)):
                candidate.predictions["functional_scores"][function] = result

        # C. Decision: everything still here proceeds to Stage 7.
        for candidate in survivors:
            candidate.predictions["functional_verdict"] = {
                "overall": "proceed",
                "primary_function": primary,
                "secondary_functions": secondary,
            }
        logger.info(
            "s06_cosmetic.done",
            extra={"n_in": len(candidates), "n_out": len(survivors), "rejected": rejected},
        )
        return survivors
