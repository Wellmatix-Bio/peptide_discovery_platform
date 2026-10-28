# Stage 4 (cosmetics): seed-based GA per incoming candidate, followed by a validity check that rejects invalid peptides.
from __future__ import annotations

from typing import Any

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext, StageError
from pipeline.cosmetic_functions import FUNCTIONS, score_function
from pipeline.s04_generation.cosmetic_validity import validity_filter
from pipeline.s04_generation.routeA_cosmetic import PrimaryPredictor, RouteACosmetic
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage4CosmeticParams

logger = get_logger(__name__)


def _setting(params: Stage4CosmeticParams, ctx: RunContext, name: str, brief_path: str) -> Any:
    """Stage params win; the cosmetic brief supplies whatever they leave unset."""
    value = getattr(params, name)
    if value is not None:
        return value
    obj: Any = ctx.brief
    for part in brief_path.split("."):
        obj = getattr(obj, part, None)
    return obj


def build_primary_predictor(
    function: str, threshold: float, ctx: RunContext
) -> PrimaryPredictor:
    """Active/inactive decision for the brief's primary function: the model score at or above `threshold`."""
    if function not in FUNCTIONS:
        raise StageError(CosmeticStage4.name, f"unsupported primary function {function!r}")
    return lambda sequences: [
        score >= threshold
        for score in score_function(function, sequences, ctx.feature_extractor)
    ]


class CosmeticStage4(CandidateStage):
    """Stage 4 for cosmetic briefs; opt-in, replaces the wound-healing generation stage."""

    name = "s04_cosmetic_candidate_generation"
    produces = {"sequence"}

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        params = Stage4CosmeticParams.model_validate(config.params)
        function = _setting(params, ctx, "primary_function", "functions.primary")
        min_length = _setting(params, ctx, "min_length", "peptide.min_length")
        max_length = _setting(params, ctx, "max_length", "peptide.max_length")
        ph = _setting(params, ctx, "product_ph", "product.ph")
        missing = [
            n
            for n, v in (
                ("primary_function", function),
                ("min_length", min_length),
                ("max_length", max_length),
                ("product_ph", ph),
            )
            if v is None
        ]
        if missing:
            raise StageError(
                self.name, f"set {missing} in the stage params or the cosmetic brief."
            )

        # Seeds are known actives, so an invalid one is dropped before it can shape the reference distribution.
        seeds, seed_rejections = validity_filter(
            [c for c in candidates if c.sequence], min_length, max_length
        )
        if seed_rejections:
            logger.warning(
                "stage4_cosmetic.invalid_seeds",
                extra={"rejected": dict(seed_rejections)},
            )
        if not seeds:
            logger.warning("stage4_cosmetic.no_valid_seeds")
            return []

        route = RouteACosmetic(
            {
                "seed_candidates": [
                    {
                        "id": c.id,
                        "sequence": c.sequence,
                        "modifications": c.predictions.get("modifications"),
                    }
                    for c in seeds
                ],
                "primary_function": function,
                "predict_active": build_primary_predictor(
                    function, params.primary_threshold, ctx
                ),
                "min_length": min_length,
                "max_length": max_length,
                "product_ph": ph,
                "random_seed": ctx.seed,
                "pop_size": params.pop_size,
                "n_generations": params.n_generations,
                "tournament_size": params.tournament_size,
                "crossover_prob": params.crossover_prob,
                "top_k": params.top_k,
            }
        )
        generated = route.run()
        valid, rejected = validity_filter(generated, min_length, max_length)
        logger.info(
            "stage4_cosmetic.validity",
            extra={
                "generated": len(generated),
                "valid": len(valid),
                "rejected": dict(rejected),
            },
        )
        return valid
