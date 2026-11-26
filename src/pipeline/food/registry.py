"""The food pipeline's stages. Same return shape as `registry.build_stages`: (setup stages, candidate stages)."""

from __future__ import annotations

from pipeline.base import CandidateStage, SetupStage
from pipeline.food.digestion.stage import FoodDigestion
from pipeline.food.generation.stage import FoodGeneration
from pipeline.food.gut_stability.stage import FoodGutStability
from pipeline.food.ranking.stage import FoodRankingDecision
from pipeline.food.screening.stage import FoodScreening
from schemas.run_config import RunConfig

#: Filled as each food stage is written, in run order.
FOOD_SETUP_STAGE_CLASSES: list[type[SetupStage]] = []
FOOD_CANDIDATE_STAGE_CLASSES: list[type[CandidateStage]] = [FoodDigestion, FoodGeneration, FoodScreening, FoodGutStability, FoodRankingDecision]


def build_food_stages(config: RunConfig) -> tuple[list[SetupStage], list[CandidateStage]]:
    """Enabled food stages, in declared order; a stage absent from the run config is enabled, as in `build_stages`."""
    setup = [cls() for cls in FOOD_SETUP_STAGE_CLASSES if config.for_stage(cls.name).enabled]
    candidate = [cls() for cls in FOOD_CANDIDATE_STAGE_CLASSES if config.for_stage(cls.name).enabled]
    return setup, candidate
