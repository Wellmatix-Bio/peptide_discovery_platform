# Loads config, resolves stage order, and executes the pipeline.

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from common.audit import AuditWriter
from common.env import DEV_MODE
from common import storage
from common.io import BoundaryWriter, load_candidates_fasta, write_final_candidates
from common.logging import get_logger
from common.model_registry import ModelRegistry
from common.stats import write_run_stats
from pipeline.base import CandidateStage, CandidateStageResult, RunContext, SetupStage
from pipeline.feature_extractor import FeatureExtractor
from registry import build_stages
from schemas.candidate import Candidate
from schemas.knowledge_base import KnowledgeBase
from schemas.run_config import SCHEMA_VERSION, RunConfig

logger = get_logger(__name__)

# Maps each setup stage to the RunContext field its payload populates.
SETUP_STAGE_TARGET_FIELD = {
    "s01_therapeutic_product_brief": "brief",
    "s02_wound_biology_and_targets": "objectives",
    # "s03_data_integration": "standardized_records",
}


class ConfigError(RuntimeError):
    """Raised when a run config cannot produce a valid pipeline."""


def build_run_services(
    run_id: str,
    run_dir: str,
    model_store: str,
    feature_extractor: FeatureExtractor,
    *,
    seed: int = 42,
) -> RunContext:
    """Shared by PipelineRunner._build_context and worker.py's run_job."""
    return RunContext(
        run_id=run_id,
        audit=AuditWriter(f"{run_dir}/audit_log.jsonl"),
        boundary=BoundaryWriter(run_dir),
        models=ModelRegistry(model_store),
        feature_extractor=feature_extractor,
        seed=seed,
    )


class PipelineRunner:
    """Executes setup stages, then candidate stages, for one run config."""

    def __init__(self, config: RunConfig, status_path: str | None = None) -> None:
        self.config = config
        self.ctx = self._build_context(config)
        self.setup_stages, self.candidate_stages = build_stages(config)
        self._cancelled = False
        self._progress_path = status_path

    # ------------------------------------------------------------------
    def _update_progress(self, stage: str, progress: float) -> None:
        if self._progress_path:
            storage.write_text(
                self._progress_path,
                json.dumps(
                    {
                        "run_id": self.config.run_id,
                        "status": "running",
                        "progress": progress,
                        "stage": stage,
                    }
                ),
            )

    def run(self) -> list[Candidate]:
        num_stages = len(self.setup_stages) + len(self.candidate_stages)
        for idx, stage in enumerate(self.setup_stages):
            self._check_cancelled()
            if not DEV_MODE:
                self._update_progress(stage.name, idx / num_stages)
            result = stage.execute(self.config.for_stage(stage.name), self.ctx)
            target_field = SETUP_STAGE_TARGET_FIELD.get(stage.name)
            if target_field is not None:
                self.ctx = replace(self.ctx, **{target_field: result.payload})

        candidates = self._load_entry_candidates()
        self._validate_contracts(candidates)
        n_start = len(candidates)
        stage_results: list[CandidateStageResult] = []
        run_started = time.perf_counter()

        for idx, stage in enumerate(self.candidate_stages):
            self._check_cancelled()
            if not DEV_MODE:
                self._update_progress(
                    stage.name, (len(self.setup_stages) + idx) / num_stages
                )
            stage_result = stage.execute(
                candidates, self.config.for_stage(stage.name), self.ctx
            )
            stage_results.append(stage_result)
            candidates = stage_result.candidates
            if not stage_result.candidates:
                logger.warning("run.exhausted", extra={"stage": stage.name})
                break

        run_duration = time.perf_counter() - run_started
        logger.info(
            "run.done", extra={"run_id": self.config.run_id, "n": len(candidates)}
        )

        run_dir = storage.join(self.config.artifacts_dir, "runs", self.config.run_id)
        if DEV_MODE:
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
        run_dir = storage.join(config.artifacts_dir, "runs", config.run_id)
        storage.ensure_dir(run_dir)
        config.snapshot(storage.join(run_dir, "config_snapshot.yaml"))

        return build_run_services(
            run_id=config.run_id,
            run_dir=str(run_dir),
            model_store=config.model_store,
            feature_extractor=FeatureExtractor(),
            seed=config.seed,
        )

    def _load_entry_candidates(self) -> list[Candidate]:
        """Empty for a full run (s04 generates them); loaded when entering mid-pipeline."""
        candidates = load_candidates_fasta(
            self.config.seed_candidates_path, schema_version=SCHEMA_VERSION
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
