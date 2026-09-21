# Loads config, resolves stage order, and executes the pipeline.

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import os
from common.audit import AuditWriter
from common.io import BoundaryWriter, load_candidates_fasta, write_final_candidates
from common.logging import get_logger
from common.model_registry import ModelRegistry
from common.stats import write_run_stats
from pipeline.base import CandidateStage, CandidateStageResult, RunContext, SetupStage
from pipeline.feature_extractor import FeatureExtractor
from registry import build_stages
from schemas.candidate import Candidate
from schemas.knowledge_base import KnowledgeBase
from schemas.run_config import RunConfig

logger = get_logger(__name__)

# Maps each setup stage to the RunContext field its payload populates.
SETUP_STAGE_TARGET_FIELD = {
    "s01_therapeutic_product_brief": "brief",
    "s02_wound_biology_and_targets": "objectives",
    # "s03_data_integration": "standardized_records",
}


class ConfigError(RuntimeError):
    """Raised when a run config cannot produce a valid pipeline."""


class PipelineRunner:
    """Executes setup stages, then candidate stages, for one run config."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.ctx = self._build_context(config)
        self.setup_stages, self.candidate_stages = build_stages(config)
        self._cancelled = False

    # ------------------------------------------------------------------

    def run(self) -> list[Candidate]:
        for stage in self.setup_stages:
            self._check_cancelled()
            result = stage.execute(self.config.for_stage(stage.name), self.ctx)
            target_field = SETUP_STAGE_TARGET_FIELD.get(stage.name)
            if target_field is not None:
                self.ctx = replace(self.ctx, **{target_field: result.payload})

        candidates = self._load_entry_candidates()
        self._validate_contracts(candidates)
        n_start = len(candidates)
        stage_results: list[CandidateStageResult] = []
        run_started = time.perf_counter()

        for stage in self.candidate_stages:
            self._check_cancelled()
            stage_result = stage.execute(
                candidates, self.config.for_stage(stage.name), self.ctx
            )
            stage_results.append(stage_result)
            if not stage_result.candidates:
                logger.warning("run.exhausted", extra={"stage": stage.name})
                break
            candidates = stage_result.candidates

        run_duration = time.perf_counter() - run_started
        logger.info(
            "run.done", extra={"run_id": self.config.run_id, "n": len(candidates)}
        )

        run_dir = Path(self.config.artifacts_dir) / "runs" / self.config.run_id
        stats_path = write_run_stats(
            run_dir,
            run_id=self.config.run_id,
            n_start=n_start,
            n_final=len(candidates),
            duration_s=run_duration,
            stage_results=stage_results,
        )
        logger.info("run.stats_written", extra={"path": str(stats_path)})

        final_path = write_final_candidates(run_dir, candidates)
        logger.info("run.candidates_written", extra={"path": str(final_path)})

        return candidates

    def cancel(self) -> None:
        """Request a stop. Takes effect between stages, not mid-stage."""
        self._cancelled = True

    # ------------------------------------------------------------------

    def _build_context(self, config: RunConfig) -> RunContext:
        run_dir = Path(config.artifacts_dir) / "runs" / config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config.snapshot(run_dir / "config_snapshot.yaml")

        ctx = RunContext(
            run_id=config.run_id,
            schema_version=config.schema_version,
            audit=AuditWriter(run_dir / "audit_log.jsonl"),
            boundary=BoundaryWriter(run_dir),
            models=ModelRegistry(config.model_store),
            feature_extractor=FeatureExtractor(),
            use_feature_cache=config.use_feature_cache,
            seed=config.seed or 42,
        )

        return ctx

    def _load_entry_candidates(self) -> list[Candidate]:
        """Empty for a full run (s04 generates them); loaded when entering mid-pipeline."""
        candidates = load_candidates_fasta(
            self.config.seed_candidates_path, schema_version=self.ctx.schema_version
        )
        logger.info("run.input_loaded", extra={"n": len(candidates)})
        return candidates

    def _validate_contracts(self, candidates: list[Candidate]) -> None:
        """Fail at startup if any enabled stage's `requires` is unsatisfiable."""
        available = candidates[0].present_fields() if candidates else set()
        errors: list[str] = []

        for stage in self.candidate_stages:
            missing = stage.requires - available
            if missing:
                errors.append(f"  {stage.name}: missing {sorted(missing)}")
            available |= stage.produces

        if errors:
            raise ConfigError(
                "pipeline validation failed\n\n"
                + "\n".join(errors)
                + "\n\nNo stages executed."
            )

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise RuntimeError(f"run {self.config.run_id} cancelled")


def load_config(config_path: str | Path) -> RunConfig:
    """Load a run config from a base.yaml file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return RunConfig.load(config_path)
