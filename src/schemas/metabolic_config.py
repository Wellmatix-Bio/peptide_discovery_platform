"""Job-config validation for the metabolic worker: the metabolic stages replace their wound-healing peers."""

from copy import deepcopy

from pydantic import BaseModel

from schemas.run_config import RunConfig
from schemas.stage_configs import STAGE_PARAMS_MODELS, resolve_params

BRIEF_STAGE = "s01_metabolic_therapeutic_product_brief"

#: Metabolic stages, in pipeline order; each is forced on.
METABOLIC_STAGES = (
    BRIEF_STAGE,
    "s04_metabolic_candidate_generation",
    "s05_metabolic_physchem_screening",
    "s06_metabolic_functional_screening",
    "s08_metabolic_safety_developability",
    "s11_metabolic_ranking",
)

#: Wound-healing peers the metabolic stages replace; each is forced off.
REPLACED_STAGES = (
    "s01_therapeutic_product_brief",
    "s02_wound_biology_and_targets",
    "s03_data_integration",
    "s04_candidate_generation",
    "s05_physchem_screening",
    "s06_functional_models",
    "s08_safety_developability",
    "s11_ranking",
)


def validate_metabolic_job_config(payload: dict) -> RunConfig:
    """Normalize a metabolic job request into a RunConfig with only the metabolic peers of s01/s04/s05/s06/s08/s11 on."""
    if not isinstance(payload, dict):
        raise ValueError("job config must be a mapping")
    payload = deepcopy(payload)
    stages = payload.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("stages must be a mapping")
    for stage_name, stage in list(stages.items()):
        if isinstance(stage, BaseModel):
            stage = stage.model_dump(mode="json")
        if not isinstance(stage, dict):
            raise ValueError(f"{stage_name} must be a mapping")
        if "params" in stage:
            extra = set(stage) - {"enabled", "params"}
            if extra:
                raise ValueError(
                    f"{stage_name} mixes params with flat parameters: {sorted(extra)}"
                )
            stage.setdefault("enabled", True)
            stages[stage_name] = stage
        else:
            # Public requests place each stage's parameters directly under its name.
            enabled = stage.pop("enabled", True)
            stages[stage_name] = {"enabled": enabled, "params": stage}

    for name in REPLACED_STAGES:
        stages[name] = {"enabled": False, "params": {}}
    for name in METABOLIC_STAGES:
        stage = stages.setdefault(name, {"enabled": True, "params": {}})
        stage["enabled"] = True

    if not stages[BRIEF_STAGE]["params"].get("brief"):
        raise ValueError("the metabolic worker requires an inline Stage 1 brief")

    config = RunConfig.model_validate(payload)
    for name in METABOLIC_STAGES:
        stage = config.for_stage(name)
        stage.params = resolve_params(name, stage.params)
        config.stages[name] = stage
    if "seed" not in payload:
        config.seed = config.for_stage(BRIEF_STAGE).params.get("seed", 42)
    return config
