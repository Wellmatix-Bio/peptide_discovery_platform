"""Lightweight config validation shared by the API and worker."""

from copy import deepcopy
from pydantic import BaseModel
from schemas.run_config import RunConfig
from schemas.stage_configs import STAGE_PARAMS_MODELS, resolve_params


def validate_job_config(payload: dict) -> RunConfig:
    if not isinstance(payload, dict):
        raise ValueError("job config must be a mapping")
    payload = deepcopy(payload)
    stages = payload.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("stages must be a mapping")
    for stage_name, stage in stages.items():
        if isinstance(stage, BaseModel):
            stage = stage.model_dump(mode="json")
            stages[stage_name] = stage
        if not isinstance(stage, dict):
            raise ValueError(f"{stage_name} must be a mapping")
        if "params" in stage:
            # The runner's serialized config already uses this internal shape.
            extra = set(stage) - {"enabled", "params"}
            if extra:
                raise ValueError(
                    f"{stage_name} mixes params with flat parameters: {sorted(extra)}"
                )
            stage.setdefault("enabled", True)
        else:
            # Public requests place each stage's parameters directly under its name.
            enabled = stage.pop("enabled", True)
            stages[stage_name] = {"enabled": enabled, "params": stage}

    # This is done on purpose as the current version makes no use of them
    for name in ("s02_wound_biology_and_targets", "s03_data_integration"):
        stages.setdefault(name, {"enabled": False})

    stages["s11_ranking"] = {"enabled": True, "params": {}}

    config = RunConfig.model_validate(payload)
    for name in STAGE_PARAMS_MODELS:
        stage = config.for_stage(name)
        if stage.enabled:
            stage.params = resolve_params(name, stage.params)
            config.stages[name] = stage
    if "seed" not in payload:
        config.seed = config.for_stage("s01_therapeutic_product_brief").params.get("seed", 42)
    if not config.for_stage("s01_therapeutic_product_brief").enabled:
        raise ValueError("the e2e worker requires Stage 1")
    if not config.for_stage("s01_therapeutic_product_brief").params.get("brief"):
        raise ValueError("the e2e worker requires an inline Stage 1 brief")
    return config
