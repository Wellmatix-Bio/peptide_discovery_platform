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
    """Raised when a stage cannot execute; carries `stage` and `cause` for the audit log."""

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


#: Removals are tallied from this stage number on; earlier stages build the candidate pool rather than screen it.
FIRST_COUNTED_STAGE = 5


class RejectionTally:
    """Run-wide removals from stage 5 on: total removed and reason -> count, summed from each stage's own counter."""

    def __init__(self) -> None:
        self.n_removed = 0
        self.reasons: dict[str, int] = {}

    def add(self, n_removed: int, reasons: dict[str, int]) -> None:
        self.n_removed += n_removed
        for reason, count in reasons.items():
            self.reasons[reason] = self.reasons.get(reason, 0) + count

    def as_dict(self) -> dict[str, Any]:
        return {"n_removed": self.n_removed, "reasons": dict(self.reasons)}


@dataclass(frozen=True)
class RunContext:
    """Shared services handed to every stage, injected so stages can be tested against a fake context."""

    run_id: str
    audit: AuditWriter
    boundary: BoundaryWriter
    models: ModelRegistry
    feature_extractor: FeatureExtractor
    seed: int = 42

    # populated by setup stages / loaded at startup
    brief: Brief | None = None
    objectives: ObjectiveVector | None = None
    standardized_records: list[dict[str, Any]] | None = None
    knowledge_base: KnowledgeBase | None = None

    #: Filled by each candidate stage as it finishes, saved with the run's results.
    rejections: RejectionTally = field(default_factory=RejectionTally)


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
    #: Why candidates were removed, reason -> count. One candidate can fail several checks, so the counts can add up to
    #: more than n_removed.
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    # models_used: list[ModelRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Stage(ABC):
    """Shared identity and contract metadata for both stage kinds; subclass `SetupStage` or `CandidateStage`."""

    #: Stage identifier, e.g. "s08_safety_developability"; must match the run-config key.
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
    """Base class for s01-s03, which run before any candidates exist; `run()` takes the run config and returns a JSON-serialisable payload."""

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
    """Base class for s04-s14; `run()` may modify candidates, may filter them out (returning survivors), and must populate every field in `produces` or warn."""

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

    def note_rejection(self, *reasons: str) -> None:
        """Count one removed candidate under each reason it failed; call once per removed candidate from `run()`."""
        counts = self.__dict__.setdefault("_rejection_reasons", {})
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1

    def models_used(self) -> list[ModelRef]:
        """Models invoked during the last `run()`, for provenance; override in stages that call registered models."""
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
        self._rejection_reasons = {}
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
            rejection_reasons=dict(self._rejection_reasons),
            # models_used=self.models_used(),
            warnings=warnings,
        )

        ctx.audit.record_stage(result, run_id=ctx.run_id)
        if int(self.name[1:3]) >= FIRST_COUNTED_STAGE:
            ctx.rejections.add(result.n_removed, result.rejection_reasons)
        ctx.boundary.write(self.name, survivors, run_id=ctx.run_id)
        # self.release_models()

        logger.info(
            "stage.done",
            extra={
                "stage": self.name,
                "n_out": result.n_out,
                "n_removed": result.n_removed,
                "rejection_reasons": result.rejection_reasons,
                "duration_s": round(duration, 2),
            },
        )
        return result

    # Contract checks

    def _check_preconditions(self, candidates: Sequence[Candidate]) -> None:
        """Fail fast if the incoming candidates lack required fields (the per-candidate safety net for data from disk)."""
        if not candidates:
            return

        sample = candidates[0]
        missing = {f for f in self.requires if not _has_field(sample, f)}
        if missing:
            raise PreconditionError(self.name, list(missing))

    def _check_postconditions(self, candidates: Sequence[Candidate]) -> list[str]:
        """Warn if declared outputs are missing; non-fatal because a model may abstain on a candidate."""
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
