"""Runs the food pipeline's stages with the platform's runner."""

from __future__ import annotations

from dataclasses import fields

from pipeline.base import RunContext
from pipeline.food.context import FoodRunContext
from pipeline.food.io import load_food_candidates_jsonl
from pipeline.food.registry import build_food_stages
from runner import PipelineRunner
from schemas.candidate import Candidate
from schemas.run_config import RunConfig


class FoodPipelineRunner(PipelineRunner):
    def __init__(self, config: RunConfig, status_path: str | None = None) -> None:
        super().__init__(config, status_path, stage_builder=build_food_stages)

    def _build_context(self, config: RunConfig) -> RunContext:
        ctx = super()._build_context(config)
        return FoodRunContext(**{f.name: getattr(ctx, f.name) for f in fields(ctx)})

    def _load_entry_candidates(self) -> list[Candidate]:
        """Empty unless `seed_candidates_path` names a food candidates JSONL (a stage's file under `candidates/`), to enter mid-pipeline."""
        if not self.config.seed_candidates_path:
            return []
        return load_food_candidates_jsonl(self.config.seed_candidates_path)
