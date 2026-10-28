# Stage name -> stage class lookup.
from __future__ import annotations

from pipeline.base import CandidateStage, SetupStage
from pipeline.s01_brief.cosmetic_stage import CosmeticStage1
from pipeline.s01_brief.metabolic_stage import Stage1Metabolic
from pipeline.s01_brief.stage import Stage1
from pipeline.s02_target_definition.stage import Stage2
from pipeline.s03_data_integration.stage import Stage3
from pipeline.s04_generation.cosmetic_stage import CosmeticStage4
from pipeline.s04_generation.metabolic_stage import Stage4Metabolic
from pipeline.s04_generation.stage import Stage4
from pipeline.s05_physchem_screening.cosmetic_stage import CosmeticStage5
from pipeline.s05_physchem_screening.metabolic_stage import Stage5Metabolic
from pipeline.s05_physchem_screening.stage import Stage5
from pipeline.s06_functional_models.cosmetic_stage import CosmeticStage6
from pipeline.s06_functional_models.metabolic_stage import Stage6Metabolic
from pipeline.s06_functional_models.stage import Stage6
from pipeline.s07_structure_mechanism.stage import Stage7
from pipeline.s08_safety_developability.cosmetic_stage import CosmeticStage8
from pipeline.s08_safety_developability.metabolic_stage import Stage8Metabolic
from pipeline.s08_safety_developability.stage import Stage8
from pipeline.s09_synthesis_cmc.cosmetic_stage import CosmeticStage9
from pipeline.s09_synthesis_cmc.stage import Stage9
from pipeline.s11_ranking.cosmetic_stage import CosmeticStage11
from pipeline.s11_ranking.metabolic_stage import Stage11Metabolic
from pipeline.s11_ranking.stage import Stage11
from schemas.run_config import RunConfig

SETUP_STAGE_CLASSES = [Stage1, Stage2, Stage3]
CANDIDATE_STAGE_CLASSES: list[CandidateStage] = [
    Stage4,
    Stage5,
    Stage6,
    Stage7,
    Stage8,
    Stage9,
    Stage11,
]


#: Setup-stage alternatives, slotted after their anchor the same way; a later stage's brief replaces the anchor's.
OPT_IN_SETUP_STAGES: list[tuple[type[SetupStage], type[SetupStage]]] = [
    (Stage1, Stage1Metabolic),
    (Stage1, CosmeticStage1),
]


# Opt-in alternatives as (anchor, stage): built only when the run config names them (a stage absent from the
# config defaults to enabled), and slotted right after the anchor. They are not in CANDIDATE_STAGE_CLASSES
# because the per-stage worker derives each stage's predecessor from that list's order.
OPT_IN_CANDIDATE_STAGES: list[tuple[type[CandidateStage], type[CandidateStage]]] = [
    (Stage4, Stage4Metabolic),
    (Stage4, CosmeticStage4),
    (Stage5, Stage5Metabolic),
    (Stage5, CosmeticStage5),
    (Stage6, Stage6Metabolic),
    (Stage6, CosmeticStage6),
    (Stage8, Stage8Metabolic),
    (Stage8, CosmeticStage8),
    (Stage9, CosmeticStage9),
    (Stage11, Stage11Metabolic),
    (Stage11, CosmeticStage11),
]


def build_stages(config: RunConfig) -> tuple[list[SetupStage], list[CandidateStage]]:
    """Resolve enabled stages from the run config, in declared order."""
    setup_stages = []
    for cls in SETUP_STAGE_CLASSES:
        if config.for_stage(cls.name).enabled:
            setup_stages.append(cls())
        for anchor, extra in OPT_IN_SETUP_STAGES:
            if (
                cls is anchor
                and extra.name in config.stages
                and config.stages[extra.name].enabled
            ):
                setup_stages.append(extra())
    candidate_stages = []
    for cls in CANDIDATE_STAGE_CLASSES:
        if config.for_stage(cls.name).enabled:
            candidate_stages.append(cls())
        for anchor, extra in OPT_IN_CANDIDATE_STAGES:
            if (
                cls is anchor
                and extra.name in config.stages
                and config.stages[extra.name].enabled
            ):
                candidate_stages.append(extra())
    return setup_stages, candidate_stages
