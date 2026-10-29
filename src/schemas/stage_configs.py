# Typed per-stage params mirroring each stage's config.params reads; resolve_params() merges them over defaults and ignores unknown keys.
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.brief import CosmeticBrief, MetabolicBrief


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
    brief: BriefFields | None = None
    #: Local runs only (DEV_MODE): a brief JSON file in place of the inline brief.
    brief_path: str | None = None
    seed: int = Field(42, description="Random seed for reproducibility")

    @model_validator(mode="after")
    def _needs_a_brief(self):
        if self.brief is None and not self.brief_path:
            raise ValueError("Stage 1 needs a brief (or, for local runs, a brief_path)")
        return self


class Stage1MetabolicParams(BaseStageParams):
    brief: MetabolicBrief
    seed: int = Field(42, description="Random seed for reproducibility")


class Stage1CosmeticParams(BaseStageParams):
    brief: CosmeticBrief = CosmeticBrief()
    seed: int = Field(42, description="Random seed for reproducibility")


class Stage2Params(BaseStageParams):
    # No in-code fallback in Stage2.run() either.
    deficit_rules_path: str = "./configs/deficit_rules.config.yaml"


class RouteAGAParams(BaseStageParams):
    pop_size: int = 3
    n_generations: int = 5
    mutation_rate: float = 0.1


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


class Stage4MetabolicParams(BaseStageParams):
    """Metabolic Stage 4: generation settings; receptors and lengths fall back to the brief when unset here."""

    route_a_ga_params: dict[str, Any] = {}
    target_receptors: list[MetabolicReceptor] | None = None
    min_length: int | None = Field(None, ge=5, le=50)
    max_length: int | None = Field(None, ge=5, le=50)
    #: GA fitness penalty on the pEC50-vs-pKd gap: lam * max(0, gap - tol); lam 0 turns it off.
    lam: float = Field(0.5, ge=0)
    tol: float = Field(1.0, ge=0)
    #: Route B is always untagged; this is how many peptides it samples, and the rest tune its sampling.
    n_peptides: int = Field(100, ge=1, le=10000)
    max_new_tokens: int = 120
    batch_size: int = 16
    max_attempts: int = 20


class Stage4CosmeticParams(BaseStageParams):
    """Cosmetic Stage 4: GA settings; function, lengths and pH fall back to the cosmetic brief when unset here."""

    primary_function: Literal["collagen_ecm", "antioxidant", "pigmentation"] | None = None
    #: Floor of 6 residues, the same as the cosmetic brief.
    min_length: int | None = Field(None, ge=6, le=20)
    max_length: int | None = Field(None, ge=6, le=20)
    product_ph: float | None = Field(None, ge=4.0, le=7.5)
    #: The primary function counts as active at a model score at or above this.
    primary_threshold: float = Field(0.5, ge=0, le=1)
    pop_size: int = Field(7, ge=2)
    n_generations: int = Field(25, ge=1)
    tournament_size: int = Field(3, ge=2)
    crossover_prob: float = Field(0.5, ge=0, le=1)
    top_k: int = Field(3, ge=1)


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


class Stage5CosmeticParams(BaseStageParams):
    """Cosmetic Stage 5: every cutoff is a run parameter and the defaults are the specification; the pH falls back to the brief."""

    product_ph: float | None = Field(None, ge=4.0, le=7.5)
    #: Seed peptides for the percentile bounds; when unset, the set Stage 4 attached to its candidates is used.
    seed_sequences: list[str] | None = None
    # C. Rule rejects
    molecular_weight_reject_max: float = 3000.0
    net_charge_reject_min: float = -1.0
    net_charge_reject_max: float = 2.0
    hydrophobicity_percentile_low: float = Field(5.0, ge=0, le=100)
    hydrophobicity_percentile_high: float = Field(95.0, ge=0, le=100)
    moment_percentile: float = Field(90.0, ge=0, le=100)
    membrane_active_min_charge: float = 2.0
    repeat_run_length: int = Field(3, ge=2)
    # D. Rule flags
    molecular_weight_flag_min: float = 1000.0
    molecular_weight_flag_max: float = 1500.0
    #: Also flag the lower-risk motifs (deamidation Asn-Ser / Asn-His, hydrolysis Asp-Gly).
    flag_lower_risk_motifs: bool = False
    # E. Models; neither documents a decision threshold, so these are stage defaults.
    aggregation_flag_min: float = Field(0.6, ge=0, le=1)
    solubility_threshold: float = Field(0.5, ge=0, le=1)
    solubility_solvent: str = "Ultrapure water"


class Stage5MetabolicParams(BaseStageParams):
    """Defaults of the metabolism / weight-control physicochemical screen; the stage reads these as its thresholds."""

    ph: float = 7.4
    solubility_solvent: str = "Ultrapure water"
    # Hard filters (candidate removed); length is checked against the brief's min_length / max_length.
    molecular_weight_reject_max: float = 5000.0
    #: Mean Eisenberg hydrophobicity above this rejects; just above the highest of the 168 metabolic seeds (0.257).
    hydrophobicity_reject_max: float = 0.3
    # Soft flags (candidate retained).
    molecular_weight_flag_max: float = 4577.6  # HORDB 99th percentile
    net_charge_flag_min: float = -3.02
    net_charge_flag_max: float = 1.03
    isoelectric_point_flag_ph_window: float = 0.233
    hydrophobic_moment_flag_max: float = 0.189
    amphipathicity_flag_max: float = 0.245
    instability_index_flag_max: float = 70.5
    solubility_flag_min: float = 0.173
    aggregation_tendency_flag_max: float = 0.470
    ss_confidence_unstable_max: float = 0.5
    ss_confidence_ambiguous_max: float = 0.7
    oxidation_count_flag_max: int = 3
    deamidation_count_flag_max: int = 1


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


MetabolicReceptor = Literal["GLP1R", "GIPR", "GCGR"]


class Stage6MetabolicParams(BaseStageParams):
    """Metabolic Stage 6: EC50 gates are in nM; defaults are placeholders until HORDB-derived values are supplied."""

    #: Receptors the client wants activated; unselected ones are scored for evidence but never gate or enter the fitness.
    target_receptors: list[MetabolicReceptor] = ["GLP1R"]
    ec50_max_nm: dict[str, float] = {"GLP1R": 1000.0, "GIPR": 1000.0, "GCGR": 1000.0}
    #: Predicted EC50 below this is implausible for the potency model and is clamped up to it.
    ec50_floor_nm: float = 0.001
    #: Per-receptor minimum agonism probability; unset until the classifier's threshold is calibrated.
    agonism_probability_min: dict[str, float] = {}
    #: Candidates predicted anti-inflammatory below this probability are rejected; an unscorable sequence is flagged, not rejected.
    anti_inflammatory_probability_min: float = Field(0.5, ge=0, le=1)


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


class Stage8MetabolicParams(BaseStageParams):
    """Metabolic Stage 8 defaults; the stage reads these as its thresholds."""

    #: False skips hemolysis entirely: no prediction, no filter, and Stage 11 drops it from the safety score.
    hemolysis_screening: bool = True
    #: P(hemolytic) above this rejects (hemolysis_predictor_v2 classifier, HC50 < 100 uM).
    hemolysis_probability_reject_max: float | None = 0.5
    solubility_solvent: str = "Ultrapure water"
    cytotoxicity_cell_type: CytotoxicityCellType = "DRAMP_aggregate"
    cytotoxicity_reject_max: float = 0.5
    aggregation_reject_max: float = 0.7
    solubility_flag_min: float = 0.173
    #: Cleavage-stability score below this rejects; the score is exp(-cleavage_decay_rate * exposure-weighted site severity).
    cleavage_stability_reject_min: float = 0.3
    cleavage_decay_rate: float = 0.5
    #: Predicted human-plasma half-life below this rejects; provisional until the client sets it.
    plasma_half_life_min_minutes: float = 60.0


class Stage9Params(BaseStageParams):
    rejected_difficulty_classes: list[str] = ["high"]


class Stage11MetabolicParams(BaseStageParams):
    """Metabolic Stage 11: module weights, receptor selection and the scales its scores are normalized on."""

    weights: dict[str, float] = {
        "receptor_agonism": 0.35,
        "stability": 0.20,
        "anti_inflammatory": 0.10,
        "safety": 0.15,
        "developability": 0.15,
        "mechanistic_confidence": 0.05,
    }
    target_receptors: list[MetabolicReceptor] = ["GLP1R"]
    #: Same scale as the metabolic Stage 6, so the EC50 score matches what that stage gated on.
    ec50_max_nm: dict[str, float] = {"GLP1R": 1000.0, "GIPR": 1000.0, "GCGR": 1000.0}
    ec50_floor_nm: float = 0.001
    #: Plasma half-life that scores 0.5; 10x it scores 1.0. Provisional, like the Stage 8 minimum.
    half_life_target_minutes: float = 60.0
    #: Scale of the hemolysis score (0.5 at this pHC50, 0 at twice it), from the v3 regression; null leaves hemolysis out of the safety score.
    hemolysis_phc50_reject_max: float | None = 4.0
    #: pKd mapped linearly to 0..1 (clipped); a placeholder until calibrated on incretin data.
    pkd_range: tuple[float, float] = (5.0, 11.0)


class Stage6CosmeticParams(BaseStageParams):
    """Cosmetic Stage 6: the function gates and the antimicrobial reject filter; thresholds are run parameters."""

    #: Brief fallbacks; unset means the cosmetic brief decides.
    primary_function: Literal["collagen_ecm", "antioxidant", "pigmentation"] | None = None
    secondary_functions: list[Literal["collagen_ecm", "antioxidant", "pigmentation"]] | None = None
    #: A function passes at a model score at or above its threshold.
    collagen_ecm_threshold: float = Field(0.5, ge=0, le=1)
    antioxidant_threshold: float = Field(0.5, ge=0, le=1)
    pigmentation_threshold: float = Field(0.5, ge=0, le=1)
    #: Rejected at an antimicrobial probability at or above this, so peptides do not wipe out healthy skin bacteria.
    antimicrobial_reject_min: float = Field(0.6, ge=0, le=1)


class Stage11CosmeticParams(BaseStageParams):
    """Cosmetic Stage 11: component base points, the brief-driven scalings and the score scales; all provisional."""

    #: Brief fallbacks; unset means the cosmetic brief decides.
    target_depth: Literal["outer_layer", "epidermis", "dermis"] | None = None
    safety_level: Literal["standard", "strict"] | None = None
    product_ph: float | None = Field(None, ge=4.0, le=7.5)
    #: Base points per component; the secondary entry is per secondary function.
    points_primary: float = Field(4.0, ge=0)
    points_secondary_each: float = Field(1.0, ge=0)
    points_safety: float = Field(3.0, ge=0)
    points_delivery: float = Field(2.0, ge=0)
    points_stability: float = Field(2.0, ge=0)
    points_developability: float = Field(1.0, ge=0)
    #: Brief-driven scalings of the base points.
    safety_strict_multiplier: float = Field(1.5, ge=0)
    delivery_depth_multiplier: dict[str, float] = {"outer_layer": 0.0, "epidermis": 1.0, "dermis": 1.5}
    stability_depth_multiplier: dict[str, float] = {"outer_layer": 0.5, "epidermis": 1.0, "dermis": 1.5}
    #: A product pH outside this range multiplies the developability points.
    ph_comfort_range: tuple[float, float] = (4.5, 6.5)
    developability_ph_multiplier: float = Field(1.5, ge=0)
    #: Safety score is multiplied by this for a peptide with a lipid tail.
    lipid_safety_factor: float = Field(0.75, ge=0, le=1)
    #: Safety score is multiplied by this for a predicted skin sensitizer kept in the run (Stage 8 `reject_skin_sensitizers` off).
    skin_sensitizer_safety_factor: float = Field(0.5, ge=0, le=1)
    #: Delivery score is 1 at or below the first MW, falling to 1 - delivery_max_penalty at or above first + span.
    delivery_mw_full: float = 500.0
    delivery_mw_span: float = Field(1000.0, gt=0)
    delivery_max_penalty: float = Field(0.5, ge=0, le=1)
    #: Developability falls by this per flag, counting at most `developability_flag_cap` flags.
    developability_flag_penalty: float = Field(0.25, ge=0)
    developability_flag_cap: int = Field(4, ge=1)
    #: Score of a component whose check did not run.
    not_screened_score: float = Field(0.5, ge=0, le=1)


class Stage9CosmeticParams(BaseStageParams):
    """Cosmetic Stage 9: the synthesis-feasibility cutoff is a run parameter."""

    #: Rejected below this synthesis-feasibility probability. The model sees the sequence only, not the modifications.
    synthesis_feasibility_min: float = Field(0.5, ge=0, le=1)


class Stage8CosmeticParams(BaseStageParams):
    """Cosmetic Stage 8: every cutoff is a run parameter. All defaults are provisional guesses, to be tuned or replaced."""

    #: Brief fallbacks; unset means the cosmetic brief decides.
    target_depth: Literal["outer_layer", "epidermis", "dermis"] | None = None
    safety_level: Literal["standard", "strict"] | None = None
    product_ph: float | None = Field(None, ge=4.0, le=7.5)
    # A. Rule checks
    #: Irritation tiers by seed-set percentile of (hydrophobic moment x positive charge).
    irritation_percentile_medium: float = Field(75.0, ge=0, le=100)
    irritation_percentile_high: float = Field(90.0, ge=0, le=100)
    #: Epidermis penetration needs MW at or below this, or a lipid tail. A starting value, not a validated cutoff.
    penetration_epidermis_mw_max: float = 1000.0
    # B. Models
    #: Reject predicted skin sensitizers (ensemble probability >= 0.55, fixed in the model, not a parameter); false only flags them.
    reject_skin_sensitizers: bool = True
    #: Cell types to test, named as the cytotoxicity model's labels; an unlisted name maps to the closest label.
    cytotoxicity_cell_types: list[str] = ["Human keratinocytes HaCat", "Human skin fibroblasts"]
    cytotoxicity_reject_min: float = Field(0.5, ge=0, le=1)
    hemolysis_reject_min: float = Field(0.5, ge=0, le=1)
    allergenicity_reject_min: float = Field(0.5, ge=0, le=1)
    #: The allergenicity model was trained on full-length proteins (shortest 50 residues); shorter sequences are not
    #: scored and are recorded as not screened. Set to 1 to score everything anyway, accepting meaningless values.
    allergenicity_min_length: int = Field(50, ge=1)
    #: Predicted log10 half-life in seconds below this rejects; the same 60 minutes as the metabolic pipeline.
    half_life_reject_min: float = math.log10(60 * 60)
    # C. Skin-protease stability
    #: KLK5, KLK7, KLK14; MMP1 is added for dermis.
    skin_protease_accessions: list[str] = ["Q9Y337", "P49862", "Q9P0G3"]
    dermis_extra_protease_accessions: list[str] = ["P03956"]
    cleavage_stability_reject_min: float = Field(0.3, ge=0, le=1)
    cleavage_decay_rate: float = Field(0.5, ge=0)


class Stage11Params(BaseStageParams):
    """Ranking uses a code-owned policy and accepts no run parameters."""

    model_config = ConfigDict(extra="forbid")


#: Stage name -> typed params model; stages without an entry keep raw passthrough params.
STAGE_PARAMS_MODELS: dict[str, type[BaseStageParams]] = {
    "s01_therapeutic_product_brief": Stage1Params,
    "s01_metabolic_therapeutic_product_brief": Stage1MetabolicParams,
    "s01_cosmetic_therapeutic_product_brief": Stage1CosmeticParams,
    "s02_wound_biology_and_targets": Stage2Params,
    "s04_candidate_generation": Stage4Params,
    "s04_metabolic_candidate_generation": Stage4MetabolicParams,
    "s04_cosmetic_candidate_generation": Stage4CosmeticParams,
    "s05_physchem_screening": Stage5Params,
    "s05_metabolic_physchem_screening": Stage5MetabolicParams,
    "s05_cosmetic_physchem_screening": Stage5CosmeticParams,
    "s06_functional_models": Stage6Params,
    "s06_metabolic_functional_screening": Stage6MetabolicParams,
    "s06_cosmetic_functional_models": Stage6CosmeticParams,
    "s07_structure_mechanism": Stage7Params,
    "s08_safety_developability": Stage8Params,
    "s08_metabolic_safety_developability": Stage8MetabolicParams,
    "s08_cosmetic_safety_developability": Stage8CosmeticParams,
    "s09_synthesis_cmc": Stage9Params,
    "s09_cosmetic_synthesis_cmc": Stage9CosmeticParams,
    "s11_ranking": Stage11Params,
    "s11_metabolic_ranking": Stage11MetabolicParams,
    "s11_cosmetic_ranking": Stage11CosmeticParams,
}


#: Stricter than the Stage*Params they wrap: unrecognized keys are rejected instead of dropped.
class Stage1Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    brief: BriefFields
    seed: int = Field(42, description="Random seed for reproducibility")
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


class Stage1MetabolicRequest(Stage1MetabolicParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage4MetabolicRequest(Stage4MetabolicParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage5MetabolicRequest(Stage5MetabolicParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage6MetabolicRequest(Stage6MetabolicParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage8MetabolicRequest(Stage8MetabolicParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage11MetabolicRequest(Stage11MetabolicParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class MetabolicE2ERequest(BaseModel):
    """The public metabolic job request shape (`pipeline: "metabolic"`): only the brief is required, every other stage defaults.

    Stages 7 and 9 are the shared wound-pipeline stages and stay on; the metabolic stages replace their wound peers, so
    the wound stage names are not accepted here.
    """

    model_config = ConfigDict(extra="forbid")

    s01_metabolic_therapeutic_product_brief: Stage1MetabolicRequest
    s04_metabolic_candidate_generation: Stage4MetabolicRequest = Stage4MetabolicRequest()
    s05_metabolic_physchem_screening: Stage5MetabolicRequest = Stage5MetabolicRequest()
    s06_metabolic_functional_screening: Stage6MetabolicRequest = Stage6MetabolicRequest()
    s07_structure_mechanism: Stage7Request = Stage7Request()
    s08_metabolic_safety_developability: Stage8MetabolicRequest = Stage8MetabolicRequest()
    s09_synthesis_cmc: Stage9Request = Stage9Request()
    s11_metabolic_ranking: Stage11MetabolicRequest = Stage11MetabolicRequest()


class Stage1CosmeticRequest(Stage1CosmeticParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage4CosmeticRequest(Stage4CosmeticParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage5CosmeticRequest(Stage5CosmeticParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage6CosmeticRequest(Stage6CosmeticParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage8CosmeticRequest(Stage8CosmeticParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage9CosmeticRequest(Stage9CosmeticParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class Stage11CosmeticRequest(Stage11CosmeticParams):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


class CosmeticE2ERequest(BaseModel):
    """The public cosmetic job request shape (`pipeline: "cosmetic"`): only the brief is required, every other stage defaults.

    Stage 7 is the shared structure stage and stays on; the cosmetic stages replace their wound-healing peers, so
    the wound-healing stage names are not accepted here.
    """

    model_config = ConfigDict(extra="forbid")

    s01_cosmetic_therapeutic_product_brief: Stage1CosmeticRequest
    s04_cosmetic_candidate_generation: Stage4CosmeticRequest = Stage4CosmeticRequest()
    s05_cosmetic_physchem_screening: Stage5CosmeticRequest = Stage5CosmeticRequest()
    s06_cosmetic_functional_models: Stage6CosmeticRequest = Stage6CosmeticRequest()
    s07_structure_mechanism: Stage7Request = Stage7Request()
    s08_cosmetic_safety_developability: Stage8CosmeticRequest = Stage8CosmeticRequest()
    s09_cosmetic_synthesis_cmc: Stage9CosmeticRequest = Stage9CosmeticRequest()
    s11_cosmetic_ranking: Stage11CosmeticRequest = Stage11CosmeticRequest()


def resolve_params(stage_name: str, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Merge a run config's raw params for `stage_name` over the stage defaults, dropping unknown keys; unregistered stages pass through."""
    
    model = STAGE_PARAMS_MODELS.get(stage_name)
    if model is None:
        return raw_params
    return model.model_validate(raw_params).model_dump(mode="json", exclude_none=True)
