"""Base stage definitions for the wmx pipeline.

There are two kinds of stage, and they are not interchangeable:

- `SetupStage` (s01-s03): run once, before any candidates exist. They take
  the run config and produce a payload (brief, target/deficit definitions,
  standardized records) that later stages consume. There is no candidate
  list to enrich, so `run()` does not take or return one.
- `CandidateStage` (s04-s14): run once candidates exist. They take the
  current candidate list and return the surviving candidates — a stage MAY
  filter candidates out directly in `run()` based on its own thresholds, and
  must log what it removed and why.

Earlier both were forced through a single `Stage.run(candidates, config, ctx)`
signature; s01/s02 had no candidates to return so they returned bare dicts,
which silently violated the base contract. Splitting the base class makes
that mismatch a type error instead of a runtime surprise.

Everything in `execute()` — precondition checks, audit logging and boundary
serialisation — is framework code and must not be overridden.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from common.audit import AuditWriter
from common.io import BoundaryWriter
from common.logging import get_logger
from common.model_registry import ModelRef, ModelRegistry
from pipeline.feature_extractor import FeatureExtractor
from schemas.brief import Brief
from schemas.objectives import ObjectiveVector
from schemas.candidate import Candidate
from schemas.knowledge_base import KnowledgeBase
from schemas.run_config import StageConfig

logger = get_logger(__name__)


class StageError(RuntimeError):
    """Raised when a stage cannot execute.

    Carries `stage` and `cause` so callers (and the audit log) can tell which
    stage failed and why without re-parsing the message string.
    """

    def __init__(
        self, stage: str, reason: str, *, cause: Exception | None = None
    ) -> None:
        self.stage = stage
        self.reason = reason
        self.cause = cause
        super().__init__(f"[{stage}] failed: {reason}")


class PreconditionError(StageError):
    """Raised when incoming candidates lack fields a candidate stage requires."""

    def __init__(self, stage: str, missing: Sequence[str]) -> None:
        self.missing = sorted(missing)
        reason = (
            f"missing required fields {self.missing}. "
            f"Enable the producing stage or supply them in the input file."
        )
        super().__init__(stage, reason)


@dataclass(frozen=True)
class RunContext:
    """Shared services handed to every stage.

    Injected rather than imported so stages remain decoupled from global state
    and can be unit-tested against a fake context.
    """

    run_id: str
    schema_version: int
    audit: AuditWriter
    boundary: BoundaryWriter
    models: ModelRegistry
    feature_extractor: FeatureExtractor
    use_feature_cache: bool = False
    seed: int = 42

    # populated by setup stages / loaded at startup
    brief: Brief | None = None
    objectives: ObjectiveVector | None = None
    standardized_records: list[dict[str, Any]] | None = None
    knowledge_base: KnowledgeBase | None = None


@dataclass
class SetupResult:
    """Outcome of a single setup-stage execution."""

    stage: str
    payload: dict[str, Any]
    duration_s: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class CandidateStageResult:
    """Outcome of a single candidate-stage execution."""

    stage: str
    candidates: list[Candidate]
    n_in: int
    n_out: int
    n_removed: int
    duration_s: float
    # models_used: list[ModelRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Stage(ABC):
    """Shared identity and contract metadata for both stage kinds.

    Not runnable on its own — subclass `SetupStage` or `CandidateStage`.
    """

    #: Stage identifier, e.g. "s08_safety_developability". Must match the
    #: key used in run configs.
    name: str

    #: If False, a run config cannot disable this stage.
    optional: bool = True

    def validate_config(self, config: StageConfig) -> None:
        """Optional stage-specific config validation, called before execution."""
        return None

    @classmethod
    def contract(cls) -> Mapping[str, Any]:
        """Machine-readable contract, used by `wmx stages` and by docs checks."""
        return {"name": cls.name, "optional": cls.optional}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class SetupStage(Stage):
    """Base class for s01-s03: stages that run before any candidates exist.

    Subclasses must set `name` and implement `run()`. `run()` receives the
    run config and returns a JSON-serialisable payload — the brief, target
    definitions, or standardized records that downstream stages read back out
    of `RunContext`/config, not a candidate list.
    """

    @abstractmethod
    def run(self, config: StageConfig, ctx: RunContext) -> Any:
        """Stage-specific logic. Return the payload this stage produces."""

    def execute(self, config: StageConfig, ctx: RunContext) -> SetupResult:
        """Run the stage with audit and boundary serialisation."""
        self.validate_config(config)

        started = time.perf_counter()
        logger.info("setup_stage.start", extra={"stage": self.name})

        try:
            payload = self.run(config, ctx)
        except Exception as exc:  # noqa: BLE001 — re-raised with stage context
            ctx.audit.record_failure(self.name, exc, run_id=ctx.run_id)
            raise StageError(self.name, str(exc), cause=exc) from exc

        duration = time.perf_counter() - started
        result = SetupResult(stage=self.name, payload=payload, duration_s=duration)

        ctx.audit.record_setup(result, run_id=ctx.run_id)

        logger.info(
            "setup_stage.done",
            extra={"stage": self.name, "duration_s": round(duration, 2)},
        )
        return result


class CandidateStage(Stage):
    """Base class for s04-s14: stages that operate on the candidate list.

    Subclasses must set `name`, `requires`, `produces` and implement `run()`.

    Contract for `run()`:
      - It MAY add or modify fields on candidates.
      - It MAY filter candidates out directly, based on its own thresholds.
        Return only the survivors — `execute()` logs however many were removed.
      - It MUST populate every field listed in `produces`, or record a warning
        explaining the omission.
    """

    #: Candidate fields that must already be present on input.
    requires: set[str] = set()

    #: Candidate fields this stage adds.
    produces: set[str] = set()

    @abstractmethod
    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        """Stage-specific logic. Return the surviving candidates, enriched."""

    def models_used(self) -> list[ModelRef]:
        """Models invoked during the last `run()`, for provenance.

        Override in stages that call registered models.
        """
        return []

    def execute(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> CandidateStageResult:
        """Run the stage with preconditions, audit and serialisation."""
        self._check_preconditions(candidates)
        self.validate_config(config)

        n_in = len(candidates)
        started = time.perf_counter()
        logger.info("stage.start", extra={"stage": self.name, "n_in": n_in})

        try:
            survivors = self.run(candidates, config, ctx)
        except Exception as exc:  # noqa: BLE001 — re-raised with stage context
            ctx.audit.record_failure(self.name, exc, run_id=ctx.run_id)
            raise StageError(self.name, str(exc), cause=exc) from exc

        warnings = self._check_postconditions(survivors)
        duration = time.perf_counter() - started

        result = CandidateStageResult(
            stage=self.name,
            candidates=survivors,
            n_in=n_in,
            n_out=len(survivors),
            n_removed=n_in - len(survivors),
            duration_s=duration,
            # models_used=self.models_used(),
            warnings=warnings,
        )

        ctx.audit.record_stage(result, run_id=ctx.run_id)
        ctx.boundary.write(self.name, survivors, run_id=ctx.run_id)
        # self.release_models()

        logger.info(
            "stage.done",
            extra={
                "stage": self.name,
                "n_out": result.n_out,
                "n_removed": result.n_removed,
                "duration_s": round(duration, 2),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Contract checks
    # ------------------------------------------------------------------

    def _check_preconditions(self, candidates: Sequence[Candidate]) -> None:
        """Fail fast if the incoming candidates lack required fields.

        The runner performs the same check statically before any stage runs;
        this is the per-candidate safety net for data arriving from disk.
        """
        if not candidates:
            return

        sample = candidates[0]
        missing = {f for f in self.requires if not _has_field(sample, f)}
        if missing:
            raise PreconditionError(self.name, list(missing))

    def _check_postconditions(self, candidates: Sequence[Candidate]) -> list[str]:
        """Warn if declared outputs are missing. Non-fatal by design: a model
        may legitimately abstain on a given candidate.
        """
        if not candidates:
            return []

        sample = candidates[0]
        missing = {f for f in self.produces if not _has_field(sample, f)}
        if missing:
            msg = f"{self.name} declared but did not produce: {sorted(missing)}"
            logger.warning("stage.postcondition", extra={"stage": self.name})
            return [msg]
        return []

    @classmethod
    def contract(cls) -> Mapping[str, Any]:
        return {
            **super().contract(),
            "requires": sorted(cls.requires),
            "produces": sorted(cls.produces),
        }

    def __repr__(self) -> str:
        return f"<CandidateStage {self.name} requires={sorted(self.requires)}>"


def _has_field(candidate: Candidate, name: str) -> bool:
    """Field presence check tolerant of both attribute and mapping storage."""
    value = getattr(candidate, name, None)
    if value is None and isinstance(getattr(candidate, "predictions", None), dict):
        value = candidate.predictions.get(name)
    return value is not None
