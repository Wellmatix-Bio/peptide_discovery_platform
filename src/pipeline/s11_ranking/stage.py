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
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

logger = get_logger(__name__)

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
# Weights of these follow the brief; the other components are always on.
OBJECTIVE_COMPONENTS: tuple[Component, ...] = (
    "wound_closure",
    "antimicrobial",
    "immunomodulation",
    "angiogenesis",
    "collagen_ecm",
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
    flag_values: dict[str, Any] = Field(default_factory=dict)
    brief_limits: dict[str, Finite] = Field(default_factory=dict)


class WeightAudit(StrictModel):
    """How one component's weight was set from the brief. `relevance` is 1.0
    for quality components; `requested` is True only when the brief's
    desired_functions selected the component."""

    baseline_weight: Unit
    relevance: Unit
    reason: str
    requested: bool
    final_weight: Unit


class ComponentContribution(StrictModel):
    score: Unit | None
    weight: Unit
    contribution: Unit | None


class FlagCutAudit(StrictModel):
    flag: str
    raw_value: Any
    severity: Unit
    cut: Unit


class ComponentPenaltyAudit(StrictModel):
    score_before_penalty: Unit | None
    flags: list[FlagCutAudit]
    penalized_score: Unit | None


class RankingResult(StrictModel):
    """One candidate's full Stage 11 output: every input, intermediate, and
    final value needed to reproduce or audit the ranking decision without
    re-running anything. normalized_scores holds the flag-penalized scores;
    the pre-penalty values are in flag_penalties."""

    candidate_id: str
    sequence: str
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
    # Requested by desired_functions but no score available (e.g. unmapped collagen_ecm).
    requested_components_without_data: list[Component] = Field(default_factory=list)
    weight_audit: dict[Component, WeightAudit] = Field(default_factory=dict)
    flag_penalties: dict[Component, ComponentPenaltyAudit] = Field(default_factory=dict)


class BatchResult(StrictModel):
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


class Bound(StrictModel):
    """One side of a numeric flag: severity 0 at `line`, 1 at `limit`. A limit
    below the line means lower-is-worse."""

    line: Finite
    limit: Finite

    @model_validator(mode="after")
    def _distinct(self):
        if self.line == self.limit:
            raise ValueError("flag line and limit must differ")
        return self


class FlagRule(StrictModel):
    """One flag-to-component penalty. Numeric flags use `bounds` (two for a
    two-sided flag); categorical flags use `levels` (str(raw) -> severity).
    `center` turns the raw value into |raw - center| first. `brief_line_field`
    shifts the bounds so `line` equals that brief value, keeping the span."""

    component: Component
    source: str = Field(min_length=1)
    p: Unit = 0.3
    bounds: list[Bound] = Field(default_factory=list)
    levels: dict[str, Unit] = Field(default_factory=dict)
    center: Finite | None = None
    brief_line_field: str | None = None

    @model_validator(mode="after")
    def _validate_rule(self):
        if any(not part for part in self.source.split(".")):
            raise ValueError(f"source path {self.source!r} must be dot-separated keys")
        if bool(self.bounds) == bool(self.levels):
            raise ValueError("a flag rule needs exactly one of bounds or levels")
        if self.levels and (self.center is not None or self.brief_line_field):
            raise ValueError("center/brief_line_field only apply to numeric flags")
        return self


class FlagPenaltyConfig(StrictModel):
    """Empty `flags` disables the step."""

    max_total_cut: Unit = 0.5
    flags: dict[str, FlagRule] = Field(default_factory=dict)


class WeightingConfig(StrictModel):
    """Brief-driven relevance of the objective components (WeightAdjuster)."""

    floor: Unit = 0.05  # relevance of an objective neither selected nor implied
    context_implied_relevance: Unit = 0.5  # relevance when only wound context implies it
    wound_closure_min_relevance: Unit = 0.5  # wound_closure never drops below this

    @model_validator(mode="after")
    def _validate_order(self):
        if self.floor > self.context_implied_relevance:
            raise ValueError("floor must not exceed context_implied_relevance")
        return self


class ModifierRule(StrictModel):
    """One Stage-1-driven mapping: a condition on desired_functions or
    wound_context, and the components it concerns. A desired_functions rule
    marks its objective components as selected. A wound_context rule marks
    the objective components whose modifier is > 1 as implied; modifiers are
    never multiplied into weights, they only carry the mapping."""

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


# Built-in adjustments derived from the product brief.
DEFAULT_DESIRED_FUNCTIONS_RULES: list[ModifierRule] = [
    ModifierRule(
        reason="antimicrobial_objective",
        field="desired_functions",
        any_of=["antimicrobial", "antimicrobial_action", "antimicrobial action"],
        modifiers={"antimicrobial": 1.25},
    ),
    ModifierRule(
        reason="migration_objective",
        field="desired_functions",
        any_of=[
            "keratinocyte_migration",
            "fibroblast_migration",
            "cell proliferation/migration",
        ],
        modifiers={"wound_closure": 1.25},
    ),
    ModifierRule(
        reason="angiogenesis_objective",
        field="desired_functions",
        any_of=["angiogenesis"],
        modifiers={"angiogenesis": 1.25},
    ),
    ModifierRule(
        reason="immunomodulation_objective",
        field="desired_functions",
        any_of=["anti_inflammatory", "immunomodulation"],
        modifiers={"immunomodulation": 1.25},
    ),
    ModifierRule(
        reason="collagen_objective",
        field="desired_functions",
        any_of=["collagen_remodeling", "collagen_synthesis", "collagen synthesis"],
        modifiers={"collagen_ecm": 1.25},
    ),
]


# Built-in wound-context adjustments, combined with desired-function rules.
DEFAULT_WOUND_CONTEXT_RULES: list[ModifierRule] = [
    ModifierRule(
        reason="infected_wound",
        field="wound_context",
        any_of=["infected"],
        modifiers={"antimicrobial": 1.25, "immunomodulation": 1.15},
    ),
    ModifierRule(
        reason="diabetic_or_chronic_wound",
        field="wound_context",
        any_of=["diabetic", "chronic", "high_glucose"],
        modifiers={
            "wound_closure": 1.20,
            "angiogenesis": 1.20,
            "immunomodulation": 1.15,
            "stability": 1.20,
        },
    ),
    ModifierRule(
        reason="clean_surgical_wound",
        field="wound_context",
        all_of=["clean", "surgical"],
        none_of=["infected"],
        modifiers={
            "wound_closure": 1.20,
            "collagen_ecm": 1.25,
            "angiogenesis": 1.15,
            "antimicrobial": 0.75,
        },
    ),
    ModifierRule(
        reason="biofilm_positive_wound",
        field="wound_context",
        any_of=["biofilm_positive"],
        modifiers={"stability": 1.20},
    ),
    ModifierRule(
        reason="acute_wound",
        field="wound_context",
        any_of=["acute", "surgical", "traumatic"],
        modifiers={"wound_closure": 1.20, "collagen_ecm": 1.20, "safety": 1.15},
    ),
    ModifierRule(
        reason="ischemic_or_low_perfusion_wound",
        field="wound_context",
        any_of=["ischemic", "low_perfusion"],
        modifiers={"angiogenesis": 1.35},
    ),
    ModifierRule(
        reason="necrotic_wound",
        field="wound_context",
        any_of=["necrotic"],
        modifiers={"antimicrobial": 1.20, "wound_closure": 0.80},
    ),
    ModifierRule(
        reason="high_exudate_wound",
        field="wound_context",
        any_of=["high_exudate"],
        modifiers={"stability": 1.20},
    ),
    ModifierRule(
        reason="burn_wound",
        field="wound_context",
        any_of=["burn"],
        modifiers={"antimicrobial": 1.20, "immunomodulation": 1.20, "safety": 1.15},
    ),
    ModifierRule(
        reason="radiation_induced_wound",
        field="wound_context",
        any_of=["radiation_induced"],
        modifiers={"angiogenesis": 1.25, "collagen_ecm": 1.15},
    ),
]


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
    wiring, constructed in code. Validation enforces: baseline weights
    sum to 1, are non-negative, unknown score names raise."""

    baseline_weights: dict[Component, Nonnegative]
    modifier_rules: list[ModifierRule] = Field(
        default_factory=lambda: [
            *DEFAULT_DESIRED_FUNCTIONS_RULES,
            *DEFAULT_WOUND_CONTEXT_RULES,
        ]
    )
    normalizers: dict[Component, NormalizerConfig] = Field(default_factory=dict)
    input_sources: InputSources = Field(default_factory=InputSources)
    flag_penalties: FlagPenaltyConfig = Field(default_factory=FlagPenaltyConfig)
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)

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


# ----------------------------------------------------------------------
# ScoreNormalizer: converts a raw value into a [0,1] score, higher = better.
# Missing values pass through untouched.
# ----------------------------------------------------------------------


# Flag lines mirror Stage 5's DEFAULT_THRESHOLDS; limits (severity 1) are Stage 11's own.
DEFAULT_FLAG_PENALTIES: dict[str, Any] = {
    "max_total_cut": 0.5,
    "flags": {
        "net_charge": {
            "component": "safety",
            "source": "net_charge",
            "bounds": [{"line": -5.0, "limit": -10.0}, {"line": 9.0, "limit": 14.0}],
        },
        "hydrophobic_moment": {
            "component": "safety",
            "source": "hydrophobic_moment",
            "bounds": [{"line": 0.5, "limit": 1.0}],
        },
        "amphipathicity": {
            "component": "safety",
            "source": "amphipathicity",
            "bounds": [{"line": 0.8, "limit": 1.0}],
        },
        "oxidation_risk": {
            "component": "stability",
            "source": "oxidation_risk.risk_category",
            "levels": {"medium": 0.5, "high": 1.0},
        },
        "deamidation_risk": {
            "component": "stability",
            "source": "deamidation_risk.risk_category",
            "levels": {"medium": 0.5, "high": 1.0},
        },
        "instability_index": {
            "component": "stability",
            "source": "instability_index",
            "bounds": [{"line": 40.0, "limit": 100.0}],
        },
        "aggregation_tendency": {
            "component": "stability",
            "source": "aggregation_tendency.score",
            "bounds": [{"line": 0.6, "limit": 0.7}],
        },
        "solubility": {
            "component": "stability",
            "source": "solubility.score",
            "bounds": [{"line": 0.4, "limit": 0.0}],
        },
        "isoelectric_point": {
            "component": "stability",
            "source": "isoelectric_point",
            "center": 7.4,
            "bounds": [{"line": 0.5, "limit": 0.0}],
        },
        "disulfide_complexity": {
            "component": "synthesis_feasibility",
            "source": "disulfide_complexity.category",
            "levels": {"flag": 0.5, "high": 1.0},
        },
        "secondary_structure_confidence": {
            "component": "mechanistic_confidence",
            "source": "secondary_structure_consistency.mean_confidence",
            "bounds": [{"line": 0.7, "limit": 0.5}],
        },
        "secondary_structure_mechanism": {
            "component": "mechanistic_confidence",
            "source": "secondary_structure_consistency.mechanism_consistent",
            "levels": {"False": 0.5},
        },
    },
}

# Code-owned ranking policy; run configs cannot override it.
BUILTIN_RANKING_POLICY = RankingConfig.model_validate(
    {
        "baseline_weights": {
            "wound_closure": 0.22,
            "antimicrobial": 0.18,
            "immunomodulation": 0.13,
            "angiogenesis": 0.12,
            "collagen_ecm": 0.1,
            "safety": 0.12,
            "stability": 0.08,
            "synthesis_feasibility": 0.03,
            "mechanistic_confidence": 0.02,
        },
        "normalizers": {"safety": {"kind": "probability", "higher_is_better": False}},
        "input_sources": {
            "scores": {
                "antimicrobial": "amp_probability",
                "immunomodulation": "anti_inflammatory_probability",
                "angiogenesis": "angiogenic_activity.angiogenic",
                "wound_closure": "proliferation_migration.migration",
                "safety": "cytotoxicity.score",
                "stability": "cleavage_stability.score",
                "synthesis_feasibility": "synthesis_feasibility.ml_feasibility_prior.score",
                "mechanistic_confidence": "mechanism.structural_confidence",
            }
        },
        "flag_penalties": DEFAULT_FLAG_PENALTIES,
    }
)


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
    """w_i = baseline_i * relevance_i for objective components, baseline_i for
    quality components, divided by the sum. Relevance is the max of: 1.0 if
    selected in desired_functions, context_implied_relevance if implied by
    wound context (implications do not stack), else the floor;
    wound_closure is never below wound_closure_min_relevance. An empty brief
    (no functions, no context) keeps the baseline weights."""

    def __init__(self, config: RankingConfig):
        self.config = config

    def _selected(self, stage1: ProductObjective) -> set[Component]:
        return {
            component
            for rule in self.config.modifier_rules
            if rule.field == "desired_functions" and rule.matches(stage1.desired_functions)
            for component in rule.modifiers
            if component in OBJECTIVE_COMPONENTS
        }

    def _implied(self, stage1: ProductObjective) -> dict[Component, set[str]]:
        """Objective component -> wound-context terms that imply it."""
        implied: dict[Component, set[str]] = {}
        for rule in self.config.modifier_rules:
            if rule.field != "wound_context" or not rule.matches(stage1.wound_context):
                continue
            terms = set(stage1.wound_context) & {*rule.any_of, *rule.all_of}
            for component, modifier in rule.modifiers.items():
                if component in OBJECTIVE_COMPONENTS and modifier > 1:
                    implied.setdefault(component, set()).update(terms)
        return implied

    def adjust(
        self, stage1: ProductObjective
    ) -> tuple[dict[Component, float], dict[Component, WeightAudit]]:
        cfg = self.config.weighting
        baseline = self.config.baseline_weights
        empty_brief = not stage1.desired_functions and not stage1.wound_context
        selected = set() if empty_brief else self._selected(stage1)
        implied = {} if empty_brief else self._implied(stage1)

        relevance: dict[Component, float] = {}
        reason: dict[Component, str] = {}
        for component in COMPONENTS:
            if component not in OBJECTIVE_COMPONENTS:
                relevance[component], reason[component] = 1.0, "always on"
            elif empty_brief:
                relevance[component], reason[component] = 1.0, "empty brief"
            elif component in selected:
                relevance[component], reason[component] = 1.0, "selected"
            elif component in implied:
                relevance[component] = cfg.context_implied_relevance
                reason[component] = f"implied by {', '.join(sorted(implied[component]))}"
            else:
                relevance[component], reason[component] = cfg.floor, "floor"
            if (
                component == "wound_closure"
                and not empty_brief
                and relevance[component] < cfg.wound_closure_min_relevance
            ):
                relevance[component] = cfg.wound_closure_min_relevance
                reason[component] = "wound_closure minimum"

        weights = {c: baseline[c] * relevance[c] for c in COMPONENTS}
        total = math.fsum(weights.values())
        if total <= 0:
            raise ValueError("all adjusted weights collapsed to zero or below")
        normalized: dict[Component, float] = {c: weights[c] / total for c in COMPONENTS}
        audit = {
            c: WeightAudit(
                baseline_weight=baseline[c],
                relevance=relevance[c],
                reason=reason[c],
                requested=c in selected,
                final_weight=normalized[c],
            )
            for c in COMPONENTS
        }
        return normalized, audit


# ----------------------------------------------------------------------
# CandidateScorer: missing-value handling, final score. No gating -- every
# candidate is scored; only zero evidence makes a rank impossible.
# ----------------------------------------------------------------------


class ApplyFlagPenalties:
    """s'_i = s_i * prod(1 - cut_f), floored at s_i * (1 - max_total_cut), where
    cut_f = severity_f * p_f. Flags on a missing component are ignored (see
    docs/TODO.md). Nothing is rejected."""

    def __init__(self, config: FlagPenaltyConfig):
        self.config = config

    @staticmethod
    def _severity(rule: FlagRule, raw: Any, brief_limits: dict[str, float]) -> float:
        if rule.levels:
            return rule.levels.get(str(raw), 0.0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"flag value {raw!r} is not numeric")
        if not math.isfinite(raw):
            raise ValueError("flag value must be finite")
        value = abs(raw - rule.center) if rule.center is not None else raw
        shift = 0.0
        brief_line = brief_limits.get(rule.brief_line_field or "")
        if brief_line is not None:
            shift = brief_line - rule.bounds[0].line
        return max(
            min(1.0, max(0.0, (value - (b.line + shift)) / (b.limit - b.line)))
            for b in rule.bounds
        )

    def _floor(self, cuts: list[FlagCutAudit]) -> float:
        return max(
            math.prod(1.0 - c.cut for c in cuts), 1.0 - self.config.max_total_cut
        )

    def apply(
        self,
        normalized: dict[Component, float | None],
        flag_values: dict[str, Any],
        brief_limits: dict[str, float],
    ) -> tuple[
        dict[Component, float | None],
        dict[Component, ComponentPenaltyAudit],
    ]:
        """Returns penalized scores and the per-component audit."""
        by_component: dict[Component, list[FlagCutAudit]] = {c: [] for c in COMPONENTS}
        for name, rule in self.config.flags.items():
            raw = flag_values.get(name)
            if raw is None:
                continue
            severity = self._severity(rule, raw, brief_limits)
            if severity > 0:
                by_component[rule.component].append(
                    FlagCutAudit(
                        flag=name,
                        raw_value=raw,
                        severity=severity,
                        cut=severity * rule.p,
                    )
                )

        penalized: dict[Component, float | None] = {}
        audit: dict[Component, ComponentPenaltyAudit] = {}
        for component in COMPONENTS:
            score = normalized[component]
            cuts = by_component[component]
            if score is None:
                cuts = []
                penalized[component] = None
            else:
                penalized[component] = score * self._floor(cuts) if cuts else score
            audit[component] = ComponentPenaltyAudit(
                score_before_penalty=score,
                flags=cuts,
                penalized_score=penalized[component],
            )
        return penalized, audit


class CandidateScorer:
    def __init__(self, config: RankingConfig):
        self.config = config
        self.flag_penalties = ApplyFlagPenalties(config.flag_penalties)
        self.adjuster = WeightAdjuster(config)
        self.normalizers = {
            component: ScoreNormalizer(
                config.normalizers.get(component, NormalizerConfig())
            )
            for component in COMPONENTS
        }

    def score(self, candidate: RankingInput) -> RankingResult:
        adjusted_weights, weight_audit = self.adjuster.adjust(candidate.stage1)

        normalized: dict[Component, float | None] = {
            component: self.normalizers[component].normalize(
                getattr(candidate.scores, component)
            )
            for component in COMPONENTS
        }
        normalized, penalty_audit = self.flag_penalties.apply(
            normalized, candidate.flag_values, candidate.brief_limits
        )
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
            requested_components_without_data=[
                c for c in missing if weight_audit[c].requested
            ],
            weight_audit=weight_audit,
            flag_penalties=penalty_audit,
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

    def run(
        self, candidates: list[Candidate], config: StageConfig, ctx: RunContext
    ) -> list[Candidate]:
        if config.params:
            raise ValueError(
                "Stage 11 ranking policy is built into the code; parameters are not accepted"
            )
        policy = BUILTIN_RANKING_POLICY.model_copy(deep=True)
        stage1 = ProductObjective(
            wound_context=ctx.brief.wound_context if ctx.brief else [],
            desired_functions=ctx.brief.desired_functions if ctx.brief else [],
        )

        flag_rules = policy.flag_penalties.flags
        brief_limits = {
            rule.brief_line_field: value
            for rule in flag_rules.values()
            if rule.brief_line_field
            and (value := getattr(ctx.brief, rule.brief_line_field, None)) is not None
        }

        inputs = []
        for candidate in candidates:
            if not candidate.sequence:
                raise ValueError(f"Stage 11 candidate {candidate.id} has no sequence")
            inputs.append(
                RankingInput(
                    candidate_id=candidate.id,
                    sequence=candidate.sequence,
                    stage1=stage1,
                    flag_values={
                        name: _read_path(candidate.predictions, rule.source)
                        for name, rule in flag_rules.items()
                    },
                    brief_limits=brief_limits,
                    scores=ComponentScores(
                        **{
                            component: _read_path(candidate.predictions, path)
                            for component, path in policy.input_sources.scores.items()
                        }
                    ),
                )
            )

        batch = Stage11Service(policy).rank_batch(inputs)

        by_id = {candidate.id: candidate for candidate in candidates}
        ordered = batch.ranked_candidates + batch.insufficient_evidence_candidates
        for result in ordered:
            result_dict = result.model_dump(mode="json")
            result_dict.pop("candidate_id", None)
            result_dict.pop("sequence", None)
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
