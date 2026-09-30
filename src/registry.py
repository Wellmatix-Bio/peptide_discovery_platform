# Stage name -> stage class lookup.
from __future__ import annotations

from pipeline.base import CandidateStage, SetupStage
from pipeline.s01_brief.stage import Stage1
from pipeline.s02_target_definition.stage import Stage2
from pipeline.s03_data_integration.stage import Stage3
from pipeline.s04_generation.stage import Stage4
from pipeline.s05_physchem_screening.stage import Stage5
from pipeline.s06_functional_models.stage import Stage6
from pipeline.s07_structure_mechanism.stage import Stage7
from pipeline.s08_safety_developability.stage import Stage8
from pipeline.s09_synthesis_cmc.stage import Stage9
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


def build_stages(config: RunConfig) -> tuple[list[SetupStage], list[CandidateStage]]:
    """Resolve enabled stages from the run config, in declared order."""
    setup_stages = [
        cls() for cls in SETUP_STAGE_CLASSES if config.for_stage(cls.name).enabled
    ]
    candidate_stages = [
        cls() for cls in CANDIDATE_STAGE_CLASSES if config.for_stage(cls.name).enabled
    ]
    return setup_stages, candidate_stages
