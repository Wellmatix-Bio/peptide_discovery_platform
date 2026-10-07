# Typed per-stage params mirroring each stage's config.params reads; resolve_params() merges them over defaults and ignores unknown keys.
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseStageParams(BaseModel):
    """Extra keys in a run config are ignored; this is a defaults-merge model, not a strict schema."""

    model_config = ConfigDict(extra="ignore")


WoundContext = Literal[
    "acute",
    "biofilm_positive",
    "burn",
    "chronic",
    # "clean",
    "diabetic",
    # "high_exudate" -> affects dressing choice and peptide washout, not any current module. It may belong in a later formulation stage instead.
    "high_glucose",
    "infected",
    "ischemic",
    "low_perfusion",
    "necrotic",
    # "radiation_induced", -> the thinnest mappings. The paper says little about either.
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

#: MIC_SUPPORTED_ORGANISMS from mic_predictor_v1, underscore-normalized; pathogens outside it are skipped, not errors.
Pathogens = Literal[
    "Escherichia_coli",
    "Staphylococcus_aureus",
    "Pseudomonas_aeruginosa",
]


# Manufacturing inputs are unused by the pipeline; disabled for now.
# class ManufacturingFields(BaseModel):
#     """Hardwired to solid_phase_synthesis; max_cost_per_gram_usd is the one adjustable input."""
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


#: Stage name -> typed params model; stages without an entry keep raw passthrough params.
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


#: Stricter than the Stage*Params they wrap: unrecognized keys are rejected instead of dropped.
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
    """The public e2e job request shape: each client-configurable stage's params flattened with its `enabled` flag; s02, s03 and s11 are absent because e2e_config.py overrides them."""

    model_config = ConfigDict(extra="forbid")

    s01_therapeutic_product_brief: Stage1Request
    s04_candidate_generation: Stage4Request
    s05_physchem_screening: Stage5Request
    s06_functional_models: Stage6Request
    s07_structure_mechanism: Stage7Request
    s08_safety_developability: Stage8Request
    s09_synthesis_cmc: Stage9Request


def resolve_params(stage_name: str, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Merge a run config's raw params for `stage_name` over the stage defaults, dropping unknown keys; unregistered stages pass through."""
    
    model = STAGE_PARAMS_MODELS.get(stage_name)
    if model is None:
        return raw_params
    return model.model_validate(raw_params).model_dump(mode="json", exclude_none=True)
