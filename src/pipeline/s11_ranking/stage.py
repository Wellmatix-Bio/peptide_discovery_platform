# Stage 11: Multi-Objective Ranking.
#
# Pure ranking, not filtering: a weighted multi-objective score over Stage
# 5-9 outputs, with Stage-1-driven weight adjustment and missing-score
# renormalization. No hard rejection gates -- safety/complexity filtering
# already happens in Stage 8/9; Stage 11 only ever scores and orders. No ML
# here either. MVP scope: no formulation/delivery (Stage 10), novelty/IP, or
# commercial feasibility.
from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext, StageError
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

logger = get_logger(__name__)

RANKING_VERSION = "3.0.0"

Component = Literal[
    "wound_closure",
    "antimicrobial",
    "immunomodulation",
    "angiogenesis",
    "collagen_ecm",
    "safety",
    "stability",
    "synthesis_feasibility",
    "mechanistic_confidence",
]
COMPONENTS: tuple[Component, ...] = (
    "wound_closure",
    "antimicrobial",
    "immunomodulation",
    "angiogenesis",
    "collagen_ecm",
    "safety",
    "stability",
    "synthesis_feasibility",
    "mechanistic_confidence",
)

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Unit = Annotated[Finite, Field(ge=0, le=1)]
Nonnegative = Annotated[Finite, Field(ge=0)]
Positive = Annotated[Finite, Field(gt=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductObjective(StrictModel):
    """Stage 1's product brief, as far as Stage 11 needs it."""

    wound_context: list[str] = Field(default_factory=list)
    desired_functions: list[str] = Field(default_factory=list)


class ComponentScores(StrictModel):
    """The nine ranking components. None means missing -- never coerced to
    0: missing values are excluded from scoring and renormalized around,
    not penalized as zero."""

    wound_closure: Finite | None = None
    antimicrobial: Finite | None = None
    immunomodulation: Finite | None = None
    angiogenesis: Finite | None = None
    collagen_ecm: Finite | None = None
    safety: Finite | None = None
    stability: Finite | None = None
    synthesis_feasibility: Finite | None = None
    mechanistic_confidence: Finite | None = None


class RankingInput(StrictModel):
    """One candidate's Stage 11 input."""

    candidate_id: str = Field(min_length=1)
    sequence: str = Field(min_length=1)
    stage1: ProductObjective = Field(default_factory=ProductObjective)
    scores: ComponentScores = Field(default_factory=ComponentScores)


class WeightAdjustment(StrictModel):
    component: Component
    reason: str
    modifier: Positive


class ComponentContribution(StrictModel):
    score: Unit | None
    weight: Unit
    contribution: Unit | None


class RankingResult(StrictModel):
    """One candidate's full Stage 11 output: every input, intermediate, and
    final value needed to reproduce or audit the ranking decision without
    re-running anything."""

    candidate_id: str
    sequence: str
    ranking_version: str
    stage1: ProductObjective
    status: Literal["ranked", "insufficient_evidence"]
    final_score: Unit | None
    rank: int | None = None
    evidence_coverage: Unit
    incomplete_evidence: bool
    original_component_scores: ComponentScores
    normalized_scores: ComponentScores
    baseline_weights: dict[Component, Unit]
    adjusted_weights: dict[Component, Unit]
    component_contributions: dict[Component, ComponentContribution]
    missing_components: list[Component]
    weight_adjustments: list[WeightAdjustment]


class BatchResult(StrictModel):
    ranking_version: str
    configuration: dict[str, Any]
    total_candidates: int
    insufficient_evidence: int
    ranked_candidates: list[RankingResult]
    insufficient_evidence_candidates: list[RankingResult]


# ----------------------------------------------------------------------
# Config: biological policy, kept separate from the scoring/weighting logic
# below so it can be edited (or swapped via config.params) without touching
# code.
# ----------------------------------------------------------------------


class NormalizerConfig(StrictModel):
    """A ScoreNormalizer's configuration. "probability": the raw value is
    already [0,1], higher-is-better unless flipped. "linear": maps [lower,
    upper] -> [0,1] first -- for regression outputs (MIC, migration
    percentage, stability half-life, synthesis difficulty, ...) that aren't
    natively [0,1]. No biological thresholds are hardcoded here or anywhere
    in the scoring engine; every bound comes from config."""

    kind: Literal["probability", "linear"] = "probability"
    lower: Finite | None = None
    upper: Finite | None = None
    higher_is_better: StrictBool = True
    clip: StrictBool = False

    @model_validator(mode="after")
    def _validate_bounds(self):
        if self.kind == "linear":
            if self.lower is None or self.upper is None or self.lower >= self.upper:
                raise ValueError("linear normalization requires lower < upper")
            if not math.isfinite(self.upper - self.lower):
                raise ValueError("normalization interval must be finite")
        elif self.lower is not None or self.upper is not None or self.clip:
            raise ValueError(
                "probability normalization does not accept bounds or clipping"
            )
        return self


class ModifierRule(StrictModel):
    """One Stage-1-driven weight adjustment: a condition on desired_functions
    or wound_context, and the multiplicative modifiers it applies. Multiple
    matching rules all apply (multiplicatively) -- there is deliberately no
    independent weight set per wound type; every adjustment starts from
    baseline_weights and multiplies from there."""

    reason: str = Field(min_length=1)
    field: Literal["desired_functions", "wound_context"]
    any_of: list[str] = Field(default_factory=list)
    all_of: list[str] = Field(default_factory=list)
    none_of: list[str] = Field(default_factory=list)
    modifiers: dict[Component, Positive] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_condition(self):
        if not self.any_of and not self.all_of:
            raise ValueError("a modifier rule needs any_of or all_of")
        return self

    def matches(self, values: list[str]) -> bool:
        value_set = set(values)
        if self.any_of and not (value_set & set(self.any_of)):
            return False
        if self.all_of and not set(self.all_of) <= value_set:
            return False
        if self.none_of and value_set & set(self.none_of):
            return False
        return True


class InputSources(StrictModel):
    """Dot paths (relative to Candidate.predictions) telling Stage 11 where
    to read each component score from. No implicit composites and no
    derivation -- a path is a literal lookup, nothing else. A component left
    unmapped is always None for every candidate (missing-is-not-zero
    handling applies)."""

    scores: dict[Component, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_paths(self):
        for path in self.scores.values():
            if not path or any(not part for part in path.split(".")):
                raise ValueError(
                    f"source path {path!r} must be nonempty dot-separated keys"
                )
        return self


class RankingConfig(StrictModel):
    """The full Stage 11 policy: weights, modifiers, normalizers, and input
    wiring, loadable from YAML/JSON. Validation enforces: baseline weights
    sum to 1, are non-negative, unknown score names raise."""

    baseline_weights: dict[Component, Nonnegative]
    modifier_rules: list[ModifierRule] = Field(default_factory=list)
    normalizers: dict[Component, NormalizerConfig] = Field(default_factory=dict)
    input_sources: InputSources = Field(default_factory=InputSources)

    @model_validator(mode="after")
    def _validate_weights(self):
        if set(self.baseline_weights) != set(COMPONENTS):
            raise ValueError(
                "baseline_weights must contain exactly all nine components"
            )
        total = sum(self.baseline_weights.values())
        if not math.isfinite(total) or not math.isclose(
            total, 1.0, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("baseline weights must sum to 1.0")
        reasons = [rule.reason for rule in self.modifier_rules]
        if len(reasons) != len(set(reasons)):
            raise ValueError("modifier rule reasons must be unique")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "RankingConfig":
        """Load a ranking policy from a YAML file. No implicit default path
        -- like every other stage's config file (e.g. Stage 2's
        deficit_rules_path), the path must come from config.params."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Ranking config file not found: {source}")
        return cls.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))


# ----------------------------------------------------------------------
# ScoreNormalizer: converts a raw value into a [0,1] score, higher = better.
# Missing values pass through untouched.
# ----------------------------------------------------------------------


class ScoreNormalizer:
    def __init__(self, config: NormalizerConfig):
        self.config = config

    def normalize(self, value: float | None) -> float | None:
        """Preserve missing values; return a finite score in [0, 1]."""
        if value is None:
            return None
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("score must be a finite number")
        config = self.config
        score = value
        if config.kind == "linear":
            # _validate_bounds guarantees lower/upper are set when kind == "linear".
            assert config.lower is not None and config.upper is not None
            score = (value - config.lower) / (config.upper - config.lower)
            if config.clip:
                score = max(0.0, min(1.0, score))
        if not 0 <= score <= 1:
            raise ValueError(
                f"score {value} falls outside configured normalization bounds"
            )
        return score if config.higher_is_better else 1.0 - score


# ----------------------------------------------------------------------
# WeightAdjuster.
# ----------------------------------------------------------------------


class WeightAdjuster:
    """baseline_weights * matching modifiers, renormalized to sum to 1.0:
    adjusted_weight_i = baseline_weight_i * modifier_i, then
    normalized_weight_i = adjusted_weight_i / sum(all adjusted weights)."""

    def __init__(self, config: RankingConfig):
        self.config = config

    def adjust(
        self, stage1: ProductObjective
    ) -> tuple[dict[Component, float], list[WeightAdjustment]]:
        weights: dict[Component, float] = dict(self.config.baseline_weights)
        adjustments: list[WeightAdjustment] = []

        for rule in self.config.modifier_rules:
            values = (
                stage1.desired_functions
                if rule.field == "desired_functions"
                else stage1.wound_context
            )
            if not rule.matches(values):
                continue
            for component, modifier in rule.modifiers.items():
                weights[component] *= modifier
                adjustments.append(
                    WeightAdjustment(
                        component=component, reason=rule.reason, modifier=modifier
                    )
                )

        total = math.fsum(weights.values())
        if total <= 0:
            raise ValueError("all adjusted weights collapsed to zero or below")
        normalized: dict[Component, float] = {
            component: weights[component] / total for component in COMPONENTS
        }
        return normalized, adjustments


# ----------------------------------------------------------------------
# CandidateScorer: missing-value handling, final score. No gating -- every
# candidate is scored; only zero evidence makes a rank impossible.
# ----------------------------------------------------------------------


class CandidateScorer:
    def __init__(self, config: RankingConfig):
        self.config = config
        self.adjuster = WeightAdjuster(config)
        self.normalizers = {
            component: ScoreNormalizer(
                config.normalizers.get(component, NormalizerConfig())
            )
            for component in COMPONENTS
        }

    def score(self, candidate: RankingInput) -> RankingResult:
        adjusted_weights, adjustments = self.adjuster.adjust(candidate.stage1)

        normalized: dict[Component, float | None] = {
            component: self.normalizers[component].normalize(
                getattr(candidate.scores, component)
            )
            for component in COMPONENTS
        }
        missing: list[Component] = [
            component for component in COMPONENTS if normalized[component] is None
        ]

        # evidence_coverage: fraction of the intended weighted evidence that
        # was actually available (e.g. 0.90 = 90% present).
        evidence_coverage = min(
            1.0,
            math.fsum(
                adjusted_weights[c] for c in COMPONENTS if normalized[c] is not None
            ),
        )
        status = "insufficient_evidence" if evidence_coverage == 0 else "ranked"

        # Renormalize remaining weights around only the available
        # components, then compute each component's literal contribution so
        # contributions sum to final_score exactly.
        contributions: dict[Component, ComponentContribution] = {}
        for component in COMPONENTS:
            component_score = normalized[component]
            weight = (
                adjusted_weights[component] / evidence_coverage
                if component_score is not None and evidence_coverage
                else 0.0
            )
            weight = min(1.0, weight)
            contributions[component] = ComponentContribution(
                score=component_score,
                weight=weight,
                contribution=(
                    weight * component_score
                    if component_score is not None and evidence_coverage
                    else None
                ),
            )

        final_score = None
        if status == "ranked":
            final_score = min(
                1.0, math.fsum(c.contribution or 0.0 for c in contributions.values())
            )

        return RankingResult(
            candidate_id=candidate.candidate_id,
            sequence=candidate.sequence,
            ranking_version=RANKING_VERSION,
            stage1=candidate.stage1.model_copy(deep=True),
            status=status,
            final_score=final_score,
            evidence_coverage=evidence_coverage,
            incomplete_evidence=bool(missing),
            original_component_scores=candidate.scores.model_copy(deep=True),
            normalized_scores=ComponentScores(**normalized),
            baseline_weights=dict(self.config.baseline_weights),
            adjusted_weights=adjusted_weights,
            component_contributions=contributions,
            missing_components=missing,
            weight_adjustments=adjustments,
        )


# ----------------------------------------------------------------------
# CandidateRanker: sort + rank + batch summary.
# ----------------------------------------------------------------------


class CandidateRanker:
    """Sorts ranked candidates by final_score descending; ties broken by
    higher safety, then higher evidence_coverage, then higher stability,
    then candidate_id ascending."""

    @staticmethod
    def _tie_break_key(result: RankingResult) -> tuple[float, float, float, float, str]:
        safety: float = (
            result.normalized_scores.safety
            if result.normalized_scores.safety is not None
            else -1.0
        )
        stability: float = (
            result.normalized_scores.stability
            if result.normalized_scores.stability is not None
            else -1.0
        )
        return (
            -(result.final_score or 0.0),
            -safety,
            -result.evidence_coverage,
            -stability,
            result.candidate_id,
        )

    def rank(self, results: list[RankingResult]) -> list[RankingResult]:
        ranked = [r for r in results if r.status == "ranked"]
        ranked.sort(key=self._tie_break_key)
        return [r.model_copy(update={"rank": i + 1}) for i, r in enumerate(ranked)]


# ----------------------------------------------------------------------
# Stage11Service: orchestrates scoring + ranking for a batch.
# ----------------------------------------------------------------------


class Stage11Service:
    def __init__(self, config: RankingConfig):
        self.config = config
        self.scorer = CandidateScorer(config)
        self.ranker = CandidateRanker()

    def rank_batch(self, candidates: list[RankingInput]) -> BatchResult:
        scored = [self.scorer.score(candidate) for candidate in candidates]
        ranked = self.ranker.rank(scored)
        insufficient = [r for r in scored if r.status == "insufficient_evidence"]

        return BatchResult(
            ranking_version=RANKING_VERSION,
            configuration=self.config.model_dump(mode="json"),
            total_candidates=len(scored),
            insufficient_evidence=len(insufficient),
            ranked_candidates=ranked,
            insufficient_evidence_candidates=insufficient,
        )


# ----------------------------------------------------------------------
# Pipeline adapter: Candidate.predictions <-> RankingInput/RankingResult.
# ----------------------------------------------------------------------


def _read_path(predictions: dict[str, Any], path: str) -> Any:
    value: Any = predictions
    for key in path.split("."):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"source path {path!r} traverses a non-mapping value")
        value = value.get(key)
    return value


class Stage11(CandidateStage):
    name = "s11_ranking"
    requires = {"sequence"}
    produces = {"ranking"}

    #: Set on every run() call; the last batch's full BatchResult, for
    #: callers (e.g. reports) that want batch-level totals beyond what
    #: CandidateStageResult carries.
    last_batch_result: BatchResult | None = None

    def run(
        self, candidates: list[Candidate], config: StageConfig, ctx: RunContext
    ) -> list[Candidate]:
        self.last_batch_result = None
        unknown = set(config.params) - {"ranking_config", "ranking_config_path"}
        if unknown:
            raise ValueError(f"unknown Stage 11 parameters: {sorted(unknown)}")
        if "ranking_config" in config.params and "ranking_config_path" in config.params:
            raise ValueError("provide ranking_config or ranking_config_path, not both")
        if (
            "ranking_config" not in config.params
            and "ranking_config_path" not in config.params
        ):
            raise StageError(
                self.name,
                "ranking_config or ranking_config_path is not specified in the config.",
            )

        policy = (
            RankingConfig.model_validate(config.params["ranking_config"])
            if "ranking_config" in config.params
            else RankingConfig.load(config.params["ranking_config_path"])
        )
        stage1 = ProductObjective(
            wound_context=ctx.brief.wound_context if ctx.brief else [],
            desired_functions=ctx.brief.desired_functions if ctx.brief else [],
        )

        inputs = []
        for candidate in candidates:
            if not candidate.sequence:
                raise ValueError(f"Stage 11 candidate {candidate.id} has no sequence")
            inputs.append(
                RankingInput(
                    candidate_id=candidate.id,
                    sequence=candidate.sequence,
                    stage1=stage1,
                    scores=ComponentScores(
                        **{
                            component: _read_path(candidate.predictions, path)
                            for component, path in policy.input_sources.scores.items()
                        }
                    ),
                )
            )

        batch = Stage11Service(policy).rank_batch(inputs)
        self.last_batch_result = batch

        by_id = {candidate.id: candidate for candidate in candidates}
        ordered = batch.ranked_candidates + batch.insufficient_evidence_candidates
        for result in ordered:
            by_id[result.candidate_id].predictions["ranking"] = {
                **result.model_dump(mode="json"),
                "configuration": batch.configuration,
            }

        logger.info(
            "s11.done",
            extra={
                "total": batch.total_candidates,
                "insufficient_evidence": batch.insufficient_evidence,
            },
        )
        return [by_id[result.candidate_id] for result in ordered]
