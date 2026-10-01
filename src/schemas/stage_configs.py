# Typed per-stage params, one model per stage, mirroring each stage's own
# config.params.get(key, default) reads. Deployment only: src/worker/worker.py
# resolves each stage's raw params dict through resolve_params() before
# building its StageConfig, merging over these defaults rather than reading
# raw. main.py's local runs (RunConfig.for_stage) are untouched by this --
# configs/runs/<run_id>.yaml stays the source of truth there. Unknown keys
# are ignored, not rejected: a stale/leftover key must not block a run.
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseStageParams(BaseModel):
    """Extra keys in a run config are ignored (not an error) -- this is a
    defaults-merge model, not a strict schema."""

    model_config = ConfigDict(extra="ignore")


WoundContext = Literal[
    "acute",
    "biofilm_positive",
    "burn",
    "chronic",
    "clean",
    "diabetic",
    "high_exudate",
    "high_glucose",
    "infected",
    "ischemic",
    "low_perfusion",
    "necrotic",
    "radiation_induced",
    "surgical",
    "traumatic",
]

DesiredFunction = Literal[
    "angiogenesis",
    "anti_inflammatory",
    "antimicrobial",
    "cell_proliferation/migration",
    "collagen_remodeling",
    "collagen_synthesis",
    "fibroblast_migration",
    "immunomodulation",
    "keratinocyte_migration",
]

#: MIC_SUPPORTED_ORGANISMS from model_store/mic_predictor_v1/predictor.py
#: (3 species), underscore-normalized to match what check_pathogen_vocabulary
#: actually compares against (and what every existing data/briefs/*.json
#: already uses, e.g. "Escherichia_coli") -- NOT the space-separated
#: ATCC-suffixed strings the raw model constant uses internally. Narrower
#: than MBIC_SUPPORTED_SPECIES (13 species, s06_functional_models/stage.py),
#: which recognizes several species MIC doesn't (e.g. Candida_albicans,
#: Klebsiella_pneumoniae) -- MIC is the stricter of the two models sharing
#: this field, so it's the one this restriction is scoped to. A pathogen
#: outside this list is silently skipped by both predictors
#: (out-of-vocabulary, not an error) rather than scored.
Pathogens = Literal[
    "Escherichia_coli",
    "Staphylococcus_aureus",
    "Pseudomonas_aeruginosa",
]


# Manufacturing inputs are unused by the pipeline; disabled for now.
# class ManufacturingFields(BaseModel):
#     """Every existing data/briefs/*.json uses "solid_phase_synthesis" (the
#     one exception, "ribosomal_expression", is TC-20's deliberately-invalid
#     edge case) -- hardwired to that single value rather than left as an
#     open string. max_cost_per_gram_usd remains the one real, adjustable
#     input; the 4 other manufacturing sub-keys seen in brief data
#     (protease_resistance_required, ambient_stability_required,
#     sterile_filterable, electrospinning_compatible) stay unmodeled/ignored,
#     same as today."""
#
#     method: Literal["solid_phase_synthesis"] = "solid_phase_synthesis"
#     max_cost_per_gram_usd: float | None = None
#


class BriefFields(BaseModel):
    wound_context: list[WoundContext] = Field(..., description="Context of the wound")
    desired_functions: list[DesiredFunction] = Field(
        ..., description="Desired functions of the therapeutic product"
    )
    pathogens: list[Pathogens]
    min_length: int = Field(ge=6, le=50)
    max_length: int = Field(ge=6, le=50)
    dosing_interval_hours: int = Field(ge=1, le=168)

    @model_validator(mode="after")
    def _check_length_bounds(self) -> "BriefFields":
        if self.min_length > self.max_length:
            raise ValueError("min_length must be <= max_length")
        return self


class Stage1Params(BaseStageParams):
    brief: BriefFields
    seed: int = Field(42, description="Random seed for reproducibility")


class Stage2Params(BaseStageParams):
    # No in-code fallback in Stage2.run() either.
    deficit_rules_path: str = "./configs/deficit_rules.config.yaml"


class RouteAGAParams(BaseStageParams):
    pop_size: int = 3
    n_generations: int = 5
    mutation_rate: float = 0.1
    random_seed: int = 42


class RouteAConstraintConfig(BaseStageParams):
    charge_min: float = -3.0
    charge_max: float = 8.0
    hydro_min: float = -1.0
    hydro_max: float = 1.0
    aggregation_threshold: float = 3.0
    ph: float = 7.4


class Stage4Params(BaseStageParams):
    tags: list[Literal["<AMP>", "<ANTIBIOFILM>", "<ANTIBACTERIAL>"]] = ["<AMP>"]
    n_peptides: int = 100
    max_new_tokens: int = 120
    batch_size: int = 16
    max_attempts: int = 20
    generator: str | None = None
    route_a_ga_params: RouteAGAParams = RouteAGAParams()
    route_a_constraint_config: RouteAConstraintConfig = RouteAConstraintConfig()


class Stage5Params(BaseStageParams):
    ph: float = 7.4
    solubility_solvent: str = "Ultrapure water"
    aggregation_tendency_flag_max: float = 0.6
    solubility_flag_min: float = 0.4
    amphipathicity_flag_max: float = 0.8
    isoelectric_point_flag_ph_window: float = 0.5
    ss_confidence_unstable_max: float = 0.5
    ss_confidence_ambiguous_max: float = 0.7
    risk_category_low_max: int = 0
    risk_category_medium_max: int = 2


class Stage6Thresholds(BaseStageParams):
    min_amp_probability: float = 0.7
    min_proliferation_migration: float = 0.5
    min_angiogenic_activity: float = 0.1
    min_anti_inflammatory_probability: float = 0.5
    max_mic: dict[str, float] = {
        "Escherichia_coli": 32.0,
        "Staphylococcus_aureus": 32.0,
        "Pseudomonas_aeruginosa": 32.0,
    }
    max_mbic: dict[str, float] = {
        "Pseudomonas_aeruginosa": 32.0,
        "Staphylococcus_aureus": 32.0,
        "Candida_albicans": 32.0,
    }


class Stage6Params(BaseStageParams):
    stage6_thresholds: Stage6Thresholds = Stage6Thresholds()


class Stage7Params(BaseStageParams):
    plddt_low_confidence_max: float = 0.5
    pathway_engagement_min_probability: float = 0.5
    helix_phi_min: float = -100.0
    helix_phi_max: float = -30.0
    helix_psi_min: float = -77.0
    helix_psi_max: float = -5.0
    sheet_phi_min: float = -180.0
    sheet_phi_max: float = -45.0
    sheet_psi_min: float = 90.0
    sheet_psi_max: float = 180.0


CytotoxicityCellType = Literal[
    "DBAASP_no_cytotoxicity_hit",
    "DRAMP_aggregate",
    "Human PBMC",
    "Human Primary Epidermal Keratinocytes (HEK)",
    "Human keratinocytes HaCat",
    "Human microvascular endothelial cells HMEC-1",
    "Human skin fibroblasts",
]


class Stage8Params(BaseStageParams):
    hemolysis_phc50_reject_max: float = 4.0
    solubility_solvent: str = "Ultrapure water"
    cytotoxicity_cell_type: CytotoxicityCellType = "DRAMP_aggregate"
    cytotoxicity_reject_max: float = 0.5
    aggregation_reject_max: float = 0.7
    solubility_flag_min: float = 0.3
    cleavage_stability_reject_min: float = 0.3
    cleavage_decay_rate: float = 0.5


class Stage9Params(BaseStageParams):
    rejected_difficulty_classes: list[str] = ["high"]


class Stage11Params(BaseStageParams):
    """Ranking uses a code-owned policy and accepts no run parameters."""

    model_config = ConfigDict(extra="forbid")


#: Stage name -> its typed params model, for RunConfig.for_stage to
#: validate/default a run config's raw params dict against. A stage with no
#: entry here (s03, and s10/s12-s14 once implemented) keeps params as a
#: raw passthrough dict.
STAGE_PARAMS_MODELS: dict[str, type[BaseStageParams]] = {
    "s01_therapeutic_product_brief": Stage1Params,
    "s02_wound_biology_and_targets": Stage2Params,
    "s04_candidate_generation": Stage4Params,
    "s05_physchem_screening": Stage5Params,
    "s06_functional_models": Stage6Params,
    "s07_structure_mechanism": Stage7Params,
    "s08_safety_developability": Stage8Params,
    "s09_synthesis_cmc": Stage9Params,
    "s11_ranking": Stage11Params,
}


#: Stricter than the Stage*Params they wrap: a client request rejects an
#: unrecognized key outright (e.g. a typo, or the internal {"params": {...}}
#: shape) instead of silently dropping it the way a defaults-merge does.
class Stage1Request(Stage1Params):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage4Request(Stage4Params):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage5Request(Stage5Params):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage6Request(Stage6Params):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage7Request(Stage7Params):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage8Request(Stage8Params):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage9Request(Stage9Params):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class E2ERequest(BaseModel):
    """The public e2e job request shape: one field per client-configurable
    stage, each stage's own params flattened together with its `enabled`
    flag (no {"params": {...}} wrapper). Every field is required to be
    present (a bare {} accepts that stage's defaults) so the OpenAPI schema
    and pydantic validation reflect the real per-stage shape directly,
    instead of a generic dict a stage name could otherwise be mismatched
    against. s02_wound_biology_and_targets, s03_data_integration, and
    s11_ranking are deliberately absent -- e2e_config.py always overrides
    them itself (s02/s03 force-disabled, s11 force-default), so nothing a
    client sends for them would ever be used."""

    model_config = ConfigDict(extra="forbid")

    s01_therapeutic_product_brief: Stage1Request
    s04_candidate_generation: Stage4Request
    s05_physchem_screening: Stage5Request
    s06_functional_models: Stage6Request
    s07_structure_mechanism: Stage7Request
    s08_safety_developability: Stage8Request
    s09_synthesis_cmc: Stage9Request


def resolve_params(stage_name: str, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Merge a run config's raw params for `stage_name` over that stage's
    defaults, dropping unknown keys. Stages with no registered model pass
    their raw params through unchanged."""
    model = STAGE_PARAMS_MODELS.get(stage_name)
    if model is None:
        return raw_params
    return model.model_validate(raw_params).model_dump(mode="json", exclude_none=True)
