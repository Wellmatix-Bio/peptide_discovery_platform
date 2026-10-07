# Stage 11: Multi-Objective Ranking; scores and orders candidates, never filters them.
from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext
from schemas.brief import Brief
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

logger = get_logger(__name__)

ModuleName = Literal[
    "wound_closure",
    "antimicrobial",
    "anti_inflammatory",
    "immunomodulation",
    "angiogenesis",
    "collagen_ecm",
    "safety",
    "stability",
    "synthesis_feasibility",
    "mechanistic_confidence",
]
MODULE_NAMES: tuple[ModuleName, ...] = (
    "wound_closure",
    "antimicrobial",
    "anti_inflammatory",
    "immunomodulation",
    "angiogenesis",
    "collagen_ecm",
    "safety",
    "stability",
    "synthesis_feasibility",
    "mechanistic_confidence",
)
# "objective": N_k comes from the brief. "always_on": N_k is the objectives' average, and it carries flags.
ModuleGroup = Literal["objective", "always_on"]

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Unit = Annotated[Finite, Field(ge=0, le=1)]
Nonnegative = Annotated[Finite, Field(ge=0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Input and result models.


class ImmuneInputs(StrictModel):
    """The pathway probabilities the immunomodulation module combines; None means missing."""

    nfkb: Finite | None = None
    cytokine: Finite | None = None


class RankingInput(StrictModel):
    """One candidate's Stage 11 input: raw module scores, immunomodulation inputs and flag values; None or missing means missing."""

    candidate_id: str = Field(min_length=1)
    sequence: str = Field(min_length=1)
    brief: Brief | None = None
    measurements: dict[ModuleName, Finite | None] = Field(default_factory=dict)
    immune_inputs: ImmuneInputs = Field(default_factory=ImmuneInputs)
    flag_values: dict[str, Any] = Field(default_factory=dict)


class FlagDeductionAudit(StrictModel):
    flag: str
    raw_value: Any
    deduction: Unit


class ModuleResult(StrictModel):
    """Everything about one module for one candidate; score = max(0, normalized_score - total_deduction)."""

    group: ModuleGroup
    activated: bool
    measurement_source: str | None
    raw_measurement: Finite | None
    measurement_detail: dict[str, float] | None = None  # parts of a computed measurement
    normalized_score: Unit | None
    flags: list[FlagDeductionAudit]
    total_deduction: Unit
    score: Unit | None
    n_k: Nonnegative  # always-on modules: the average of the objectives' N_k
    references: list[str]  # brief entries that name this module (+reference points)
    implications: list[str]  # wound-context tags that point to it (+implication points each)
    requested: bool  # the brief named it directly (desired_functions or pathogens)
    nominal_weight: Unit
    weight: Unit
    contribution: Unit | None


class RankingResult(StrictModel):
    """One candidate's full Stage 11 output, with every value needed to audit the ranking."""

    candidate_id: str
    sequence: str
    status: Literal["ranked", "insufficient_evidence"]
    final_score: Unit | None
    rank: int | None = None
    evidence_coverage: Unit  # activated modules with a score / activated modules
    incomplete_evidence: bool
    modules: dict[ModuleName, ModuleResult]
    missing_modules: list[ModuleName]  # activated modules with no score
    # Named by the brief but no score available (e.g. placeholder collagen_ecm).
    requested_modules_without_data: list[ModuleName] = Field(default_factory=list)


class BatchResult(StrictModel):
    configuration: dict[str, Any]
    total_candidates: int
    insufficient_evidence: int
    ranked_candidates: list[RankingResult]
    insufficient_evidence_candidates: list[RankingResult]


# Config: biological policy, separate from the scoring logic.


class NormalizerConfig(StrictModel):
    """A ScoreNormalizer's config: "probability" is already [0,1]; "linear" maps fixed [lower, upper] to [0,1]."""

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


def _validate_source_path(path: str) -> None:
    if not path or any(not part for part in path.split(".")):
        raise ValueError(f"source path {path!r} must be nonempty dot-separated keys")


class Measurement(StrictModel):
    """A module's score: a dot-path lookup into Candidate.predictions, normalized to [0, 1] with higher = better."""

    source: str
    normalizer: NormalizerConfig = Field(default_factory=NormalizerConfig)

    @model_validator(mode="after")
    def _validate_source(self):
        _validate_source_path(self.source)
        return self


class ImmuneAlignment(StrictModel):
    """The immunomodulation measurement from the NF-kB and cytokine pathway probabilities: match to the wound-context target times agreement; missing if either is missing."""

    kind: Literal["immune_alignment"] = "immune_alignment"
    nfkb_source: str
    cytokine_source: str
    nfkb_weight: Unit = 0.5
    context_targets: dict[str, Finite]

    @model_validator(mode="after")
    def _validate_sources(self):
        for path in (self.nfkb_source, self.cytokine_source):
            _validate_source_path(path)
        return self

    def target_direction(self, wound_context: list[str]) -> float:
        targets = [self.context_targets[tag] for tag in set(wound_context) if tag in self.context_targets]
        return min(targets) if targets else 0.0

    def evaluate(
        self, inputs: ImmuneInputs, wound_context: list[str]
    ) -> tuple[float | None, dict[str, float] | None]:
        if inputs.nfkb is None or inputs.cytokine is None:
            return None, None
        d_nfkb = 1.0 - 2.0 * inputs.nfkb
        d_cyto = 1.0 - 2.0 * inputs.cytokine
        direction = self.nfkb_weight * d_nfkb + (1.0 - self.nfkb_weight) * d_cyto
        target = self.target_direction(wound_context)
        match = 1.0 - abs(direction - target) / (1.0 + abs(target))
        agreement = 1.0 - abs(d_nfkb - d_cyto) / 2.0
        score = max(0.0, min(1.0, match * (0.5 + 0.5 * agreement)))
        return score, {
            "direction_nfkb": d_nfkb,
            "direction_cytokine": d_cyto,
            "direction": direction,
            "target_direction": target,
            "match": match,
            "agreement": agreement,
        }

    @property
    def label(self) -> str:
        return f"immune_alignment({self.nfkb_source}, {self.cytokine_source})"


class Bound(StrictModel):
    """One side of a numeric flag: the deduction starts at `line` and is maximal at `limit`."""

    line: Finite
    limit: Finite

    @model_validator(mode="after")
    def _distinct(self):
        if self.line == self.limit:
            raise ValueError("flag line and limit must differ")
        return self


class FlagRule(StrictModel):
    """One Stage 5 flag that deducts from its module's score, via numeric `bounds` or categorical `levels`."""

    source: str
    bounds: list[Bound] = Field(default_factory=list)
    levels: dict[str, Unit] = Field(default_factory=dict)
    center: Finite | None = None

    @model_validator(mode="after")
    def _validate_rule(self):
        _validate_source_path(self.source)
        if bool(self.bounds) == bool(self.levels):
            raise ValueError("a flag rule needs exactly one of bounds or levels")
        if self.levels and self.center is not None:
            raise ValueError("center only applies to numeric flags")
        return self


class ModuleSpec(StrictModel):
    """One ranking module; weights come from N_k, a None `measurement` is a placeholder, and only always-on modules carry `flags`."""

    group: ModuleGroup
    measurement: Measurement | ImmuneAlignment | None = None
    flags: dict[str, FlagRule] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_group(self):
        if self.group == "objective" and self.flags:
            raise ValueError("an objective module carries no flags")
        return self


class BriefLink(StrictModel):
    """A brief entry's link to an objective module and the N_k points it adds."""

    module: ModuleName
    points: Finite


class ObjectiveWeightingConfig(StrictModel):
    """How the brief turns into objective counts N_k (function, pathogen and wound-context links)."""

    empty_brief_n_k: Finite = 1.0
    pathogen_link: BriefLink
    function_links: dict[str, BriefLink]
    context_links: dict[str, list[BriefLink]]


def _links(points: float, *modules: ModuleName) -> list[BriefLink]:
    return [BriefLink(module=m, points=points) for m in modules]


DEFAULT_OBJECTIVE_WEIGHTING = ObjectiveWeightingConfig(
    pathogen_link=BriefLink(module="antimicrobial", points=1.0),
    function_links={
        "angiogenesis": BriefLink(module="angiogenesis", points=1.0),
        "anti_inflammatory": BriefLink(module="anti_inflammatory", points=1.0),
        "immunomodulation": BriefLink(module="immunomodulation", points=0.5),
        "antimicrobial": BriefLink(module="antimicrobial", points=1.0),
        "cell_proliferation/migration": BriefLink(module="wound_closure", points=1.0),
        "fibroblast_migration": BriefLink(module="wound_closure", points=1.0),
        "keratinocyte_migration": BriefLink(module="wound_closure", points=1.0),
        "collagen_remodeling": BriefLink(module="collagen_ecm", points=1.0),
        "collagen_synthesis": BriefLink(module="collagen_ecm", points=1.0),
    },
    # high_exudate is a formulation/dressing concern and stays unmapped.
    context_links={
        "infected": _links(0.5, "antimicrobial", "immunomodulation"),
        "biofilm_positive": _links(0.5, "antimicrobial", "immunomodulation"),
        "necrotic": _links(0.5, "antimicrobial", "wound_closure", "immunomodulation"),
        "chronic": _links(0.5, "anti_inflammatory", "immunomodulation", "wound_closure"),
        "diabetic": _links(
            0.5, "anti_inflammatory", "immunomodulation", "angiogenesis", "wound_closure"
        ),
        "high_glucose": _links(0.5, "anti_inflammatory"),
        "ischemic": _links(0.5, "angiogenesis"),
        "low_perfusion": _links(0.5, "angiogenesis"),
        "acute": _links(0.5, "wound_closure"),
        "surgical": _links(0.5, "wound_closure"),
        "traumatic": _links(0.5, "wound_closure"),
        "clean": _links(0.5, "wound_closure"),
        "burn": _links(0.5, "anti_inflammatory"),
        "radiation_induced": _links(0.5, "anti_inflammatory"),
    },
)


class RankingConfig(StrictModel):
    """The full Stage 11 policy: the modules, how the brief gives each its N_k, and the flag deduction cap."""

    modules: dict[ModuleName, ModuleSpec]
    objective_weighting: ObjectiveWeightingConfig = DEFAULT_OBJECTIVE_WEIGHTING
    max_flag_deduction: Unit = 0.4  # ceiling for one numeric flag

    @model_validator(mode="after")
    def _validate_policy(self):
        if set(self.modules) != set(MODULE_NAMES):
            raise ValueError("modules must contain exactly all ten modules")
        if not self.objective_modules or not self.always_on_modules:
            raise ValueError("modules need at least one objective and one always-on module")
        flag_names = [name for spec in self.modules.values() for name in spec.flags]
        if len(flag_names) != len(set(flag_names)):
            raise ValueError("flag names must be unique across modules")
        cfg = self.objective_weighting
        targets = {link.module for link in cfg.function_links.values()}
        targets |= {link.module for links in cfg.context_links.values() for link in links}
        targets.add(cfg.pathogen_link.module)
        if any(self.modules[m].group != "objective" for m in targets):
            raise ValueError("brief mappings may only point to objective modules")
        return self

    @property
    def objective_modules(self) -> tuple[ModuleName, ...]:
        return tuple(m for m in MODULE_NAMES if self.modules[m].group == "objective")

    @property
    def always_on_modules(self) -> tuple[ModuleName, ...]:
        return tuple(m for m in MODULE_NAMES if self.modules[m].group == "always_on")

    @property
    def flag_rules(self) -> dict[str, FlagRule]:
        return {
            name: rule for spec in self.modules.values() for name, rule in spec.flags.items()
        }


# Built-in policy; flag lines mirror Stage 5's DEFAULT_THRESHOLDS.

# Code-owned ranking policy; run configs cannot override it.
BUILTIN_RANKING_POLICY = RankingConfig(
    modules={
        # --- Objective modules: the brief weights them ---
        "wound_closure": ModuleSpec(
            group="objective",
            measurement=Measurement(source="proliferation_migration.migration"),
        ),
        "antimicrobial": ModuleSpec(
            group="objective",
            measurement=Measurement(source="amp_probability"),
        ),
        "anti_inflammatory": ModuleSpec(
            group="objective",
            measurement=Measurement(source="anti_inflammatory_probability"),
        ),
        "immunomodulation": ModuleSpec(
            group="objective",
            measurement=ImmuneAlignment(
                nfkb_source="mechanism.pathway_involvement.NF_KB.probability",
                cytokine_source="mechanism.pathway_involvement.CYTOKINE_MACROPHAGE.probability",
                # d* per tag (+ = calmer, - = some immune response is useful); the minimum across tags wins.
                context_targets={
                    "diabetic": 0.7,
                    "high_glucose": 0.7,
                    "chronic": 0.7,
                    "ischemic": 0.5,
                    "burn": 0.4,
                    "low_perfusion": 0.4,
                    "surgical": 0.3,
                    "traumatic": 0.1,
                    "acute": 0.0,
                    "biofilm_positive": 0.0,
                    "necrotic": 0.0,
                    "infected": -0.1,
                },
            ),
        ),
        "angiogenesis": ModuleSpec(
            group="objective",
            measurement=Measurement(source="angiogenic_activity.angiogenic"),
        ),
        # Placeholder: no measurement yet, so always missing.
        "collagen_ecm": ModuleSpec(group="objective"),
        # --- Always-on modules: N_k = average of the objectives' N_k, carry the flags ---
        "safety": ModuleSpec(
            group="always_on",
            measurement=Measurement(
                source="cytotoxicity.score",
                normalizer=NormalizerConfig(higher_is_better=False),  # 1 - cytotoxicity
            ),
            flags={
                "net_charge": FlagRule(
                    source="net_charge",
                    bounds=[
                        Bound(line=-5.0, limit=-10.0),
                        Bound(line=9.0, limit=14.0),
                    ],
                ),
                "hydrophobic_moment": FlagRule(
                    source="hydrophobic_moment",
                    bounds=[Bound(line=0.5, limit=1.0)],
                ),
                "amphipathicity": FlagRule(
                    source="amphipathicity",
                    bounds=[Bound(line=0.8, limit=1.0)],
                ),
            },
        ),
        "stability": ModuleSpec(
            group="always_on",
            measurement=Measurement(source="cleavage_stability.score"),
            flags={
                "oxidation_risk": FlagRule(
                    source="oxidation_risk.risk_category",
                    levels={"medium": 0.2, "high": 0.4},
                ),
                "deamidation_risk": FlagRule(
                    source="deamidation_risk.risk_category",
                    levels={"medium": 0.2, "high": 0.4},
                ),
                "instability_index": FlagRule(
                    source="instability_index",
                    bounds=[Bound(line=40.0, limit=100.0)],
                ),
                "aggregation_tendency": FlagRule(
                    source="aggregation_tendency.score",
                    bounds=[Bound(line=0.6, limit=0.7)],
                ),
                "solubility": FlagRule(
                    source="solubility.score",
                    bounds=[Bound(line=0.4, limit=0.0)],
                ),
                "isoelectric_point": FlagRule(
                    source="isoelectric_point",
                    center=7.4,
                    bounds=[Bound(line=0.5, limit=0.0)],
                ),
            },
        ),
        "synthesis_feasibility": ModuleSpec(
            group="always_on",
            measurement=Measurement(
                source="synthesis_feasibility.ml_feasibility_prior.score"
            ),
            flags={
                "disulfide_complexity": FlagRule(
                    source="disulfide_complexity.category",
                    levels={"flag": 0.2, "high": 0.4},
                ),
            },
        ),
        "mechanistic_confidence": ModuleSpec(
            group="always_on",
            measurement=Measurement(source="mechanism.structural_confidence"),
            flags={
                # Stage 5's secondary-structure check: fold does not fit the mechanism.
                "secondary_structure_mechanism": FlagRule(
                    source="secondary_structure_consistency.mechanism_consistent",
                    levels={"False": 0.2},
                ),
                # Stage 5's secondary-structure check: low mean s4pred confidence.
                "secondary_structure_confidence": FlagRule(
                    source="secondary_structure_consistency.mean_confidence",
                    bounds=[Bound(line=0.7, limit=0.5)],
                ),
            },
        ),
    },
)


# ScoreNormalizer: raw measurement to a [0,1] score; missing values pass through.


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


# ApplyFlagDeductions: a module's flags to its total deduction.


class ApplyFlagDeductions:
    """Deductions add: score = max(0, normalized - sum); nothing is rejected."""

    def __init__(self, max_flag_deduction: float):
        self.max_flag_deduction = max_flag_deduction

    def _deduction(self, rule: FlagRule, raw: Any) -> float:
        if rule.levels:
            return rule.levels.get(str(raw), 0.0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"flag value {raw!r} is not numeric")
        if not math.isfinite(raw):
            raise ValueError("flag value must be finite")
        value = abs(raw - rule.center) if rule.center is not None else raw
        return max(
            min(
                self.max_flag_deduction,
                max(0.0, (value - b.line) / (2.0 * (b.limit - b.line))),
            )
            for b in rule.bounds
        )

    def evaluate(
        self, flags: dict[str, FlagRule], flag_values: dict[str, Any]
    ) -> tuple[list[FlagDeductionAudit], float]:
        """Returns the flags that deducted and the total deduction; flags with no raw value are skipped."""
        fired: list[FlagDeductionAudit] = []
        for name, rule in flags.items():
            raw = flag_values.get(name)
            if raw is None:
                continue
            deduction = self._deduction(rule, raw)
            if deduction > 0:
                fired.append(
                    FlagDeductionAudit(flag=name, raw_value=raw, deduction=deduction)
                )
        total = min(1.0, math.fsum(f.deduction for f in fired))
        return fired, total


# WeightAllocator: brief to module weights.


class ModuleWeight(StrictModel):
    """A module's N_k and its weight before missing-evidence renormalization."""

    activated: bool
    n_k: float
    references: list[str]
    implications: list[str]
    requested: bool
    nominal_weight: float


class WeightAllocator:
    """Accumulates N_k per module from the brief; weight_k = N_k / sum(N), and N_k = 0 means weight 0."""

    def __init__(self, config: RankingConfig):
        self.config = config

    def _counts(
        self, brief: Brief | None
    ) -> tuple[
        dict[ModuleName, list[str]], dict[ModuleName, float], dict[ModuleName, list[tuple[str, float]]]
    ]:
        cfg = self.config.objective_weighting
        references: dict[ModuleName, list[str]] = {}
        reference_points: dict[ModuleName, float] = {}
        implications: dict[ModuleName, list[tuple[str, float]]] = {}
        if brief is None:
            return references, reference_points, implications
        for function in brief.desired_functions:
            link = cfg.function_links.get(function)
            if link is not None:
                references.setdefault(link.module, []).append(function)
                reference_points[link.module] = max(reference_points.get(link.module, 0.0), link.points)
        if brief.pathogens:
            link = cfg.pathogen_link
            references.setdefault(link.module, []).extend(brief.pathogens)
            reference_points[link.module] = max(reference_points.get(link.module, 0.0), link.points)
        for tag in brief.wound_context:
            for link in cfg.context_links.get(tag, []):
                implications.setdefault(link.module, []).append((tag, link.points))
        return references, reference_points, implications

    def allocate(self, brief: Brief | None) -> dict[ModuleName, ModuleWeight]:
        cfg = self.config.objective_weighting
        references, reference_points, implications = self._counts(brief)
        objectives = self.config.objective_modules
        n_k: dict[ModuleName, float] = {
            m: reference_points.get(m, 0.0) + math.fsum(p for _, p in implications.get(m, []))
            for m in objectives
        }
        if math.fsum(n_k.values()) == 0:  # empty brief: objectives count equally
            n_k = {m: cfg.empty_brief_n_k for m in objectives}
            references, implications = {}, {}
        average = math.fsum(n_k.values()) / len(objectives)
        for module in self.config.always_on_modules:
            n_k[module] = average
        total_n = math.fsum(n_k.values())

        return {
            module: ModuleWeight(
                activated=n_k[module] > 0,
                n_k=n_k[module],
                references=references.get(module, []),
                implications=[tag for tag, _ in implications.get(module, [])],
                requested=module in references,
                nominal_weight=n_k[module] / total_n,
            )
            for module in MODULE_NAMES
        }


# CandidateScorer: per-module score, missing-value handling, final score.


class CandidateScorer:
    def __init__(self, config: RankingConfig):
        self.config = config
        self.deductions = ApplyFlagDeductions(config.max_flag_deduction)
        self.allocator = WeightAllocator(config)
        self.normalizers = {
            name: ScoreNormalizer(
                spec.measurement.normalizer
                if isinstance(spec.measurement, Measurement)
                else NormalizerConfig()
            )
            for name, spec in config.modules.items()
        }

    def score(self, candidate: RankingInput) -> RankingResult:
        allocation = self.allocator.allocate(candidate.brief)

        # Per module: normalize the measurement, then deduct its flags.
        raw: dict[ModuleName, float | None] = {}
        detail: dict[ModuleName, dict[str, float] | None] = {}
        normalized: dict[ModuleName, float | None] = {}
        fired: dict[ModuleName, list[FlagDeductionAudit]] = {}
        total_deduction: dict[ModuleName, float] = {}
        score: dict[ModuleName, float | None] = {}
        for name in MODULE_NAMES:
            spec = self.config.modules[name]
            if isinstance(spec.measurement, ImmuneAlignment):
                raw[name], detail[name] = spec.measurement.evaluate(
                    candidate.immune_inputs, candidate.brief.wound_context if candidate.brief else []
                )
            else:
                raw[name], detail[name] = candidate.measurements.get(name), None
            value = self.normalizers[name].normalize(raw[name])
            normalized[name] = value
            if value is None:
                # Flags on a missing module are ignored: there is no score to deduct from.
                fired[name], total_deduction[name], score[name] = [], 0.0, None
            else:
                fired[name], total_deduction[name] = self.deductions.evaluate(
                    spec.flags, candidate.flag_values
                )
                score[name] = max(0.0, value - total_deduction[name])

        activated = [m for m in MODULE_NAMES if allocation[m].activated]
        scored = [m for m in activated if score[m] is not None]
        missing: list[ModuleName] = [m for m in activated if score[m] is None]

        # evidence_coverage: activated modules that have a score / activated modules.
        evidence_coverage = len(scored) / len(activated)
        status = "insufficient_evidence" if not scored else "ranked"

        # Missing scores are excluded and the remaining weights renormalized.
        scored_weight = math.fsum(allocation[m].nominal_weight for m in scored)
        modules: dict[ModuleName, ModuleResult] = {}
        for name in MODULE_NAMES:
            spec = self.config.modules[name]
            alloc = allocation[name]
            module_score = score[name]
            weight = (
                min(1.0, alloc.nominal_weight / scored_weight)
                if name in scored and scored_weight
                else 0.0
            )
            modules[name] = ModuleResult(
                group=spec.group,
                activated=alloc.activated,
                measurement_source=(
                    None
                    if spec.measurement is None
                    else spec.measurement.label
                    if isinstance(spec.measurement, ImmuneAlignment)
                    else spec.measurement.source
                ),
                raw_measurement=raw[name],
                measurement_detail=detail[name],
                normalized_score=normalized[name],
                flags=fired[name],
                total_deduction=total_deduction[name],
                score=module_score,
                n_k=alloc.n_k,
                references=alloc.references,
                implications=alloc.implications,
                requested=alloc.requested,
                nominal_weight=alloc.nominal_weight,
                weight=weight,
                contribution=(
                    weight * module_score
                    if name in scored and module_score is not None
                    else None
                ),
            )

        final_score = None
        if status == "ranked":
            final_score = min(
                1.0, math.fsum(m.contribution or 0.0 for m in modules.values())
            )

        return RankingResult(
            candidate_id=candidate.candidate_id,
            sequence=candidate.sequence,
            status=status,
            final_score=final_score,
            evidence_coverage=evidence_coverage,
            incomplete_evidence=bool(missing),
            modules=modules,
            missing_modules=missing,
            requested_modules_without_data=[m for m in missing if allocation[m].requested],
        )


# CandidateRanker: sort, rank, batch summary.


class CandidateRanker:
    """Sorts by final_score descending; ties break on evidence_coverage, then stability, then candidate_id."""

    @staticmethod
    def _tie_break_key(result: RankingResult) -> tuple[float, float, float, str]:
        stability = result.modules["stability"].score
        return (
            -(result.final_score or 0.0),
            -result.evidence_coverage,
            -(stability if stability is not None else -1.0),
            result.candidate_id,
        )

    def rank(self, results: list[RankingResult]) -> list[RankingResult]:
        ranked = [r for r in results if r.status == "ranked"]
        ranked.sort(key=self._tie_break_key)
        return [r.model_copy(update={"rank": i + 1}) for i, r in enumerate(ranked)]


# Stage11Service: orchestrates scoring and ranking for a batch.


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


# Pipeline adapter: Candidate.predictions to RankingInput/RankingResult.



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

        flag_rules = policy.flag_rules
        immune = next(
            (s.measurement for s in policy.modules.values() if isinstance(s.measurement, ImmuneAlignment)),
            None,
        )
        inputs = []
        for candidate in candidates:
            if not candidate.sequence:
                raise ValueError(f"Stage 11 candidate {candidate.id} has no sequence")
            inputs.append(
                RankingInput(
                    candidate_id=candidate.id,
                    sequence=candidate.sequence,
                    brief=ctx.brief,
                    measurements={
                        name: _read_path(candidate.predictions, spec.measurement.source)
                        for name, spec in policy.modules.items()
                        if isinstance(spec.measurement, Measurement)
                    },
                    immune_inputs=ImmuneInputs(
                        nfkb=_read_path(candidate.predictions, immune.nfkb_source),
                        cytokine=_read_path(candidate.predictions, immune.cytokine_source),
                    )
                    if immune
                    else ImmuneInputs(),
                    flag_values={
                        name: _read_path(candidate.predictions, rule.source)
                        for name, rule in flag_rules.items()
                    },
                )
            )

        batch = Stage11Service(policy).rank_batch(inputs)

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
