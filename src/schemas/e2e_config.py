"""Lightweight config validation shared by the API and worker."""

from copy import deepcopy
from pydantic import BaseModel
from common.env import DEV_MODE
from schemas.cosmetic_config import COSMETIC_REPLACED_STAGES, COSMETIC_STAGES
from schemas.metabolic_config import METABOLIC_STAGES, REPLACED_STAGES
from schemas.run_config import RunConfig
from schemas.stage_configs import STAGE_PARAMS_MODELS, resolve_params


def validate_job_config(payload: dict, pipeline: str = "wound_healing") -> RunConfig:
    """Normalize a job request into a RunConfig for `pipeline` ("wound_healing", "metabolic" or "cosmetic")."""
    if pipeline not in ("wound_healing", "metabolic", "cosmetic"):
        raise ValueError(f"unknown pipeline {pipeline!r}")
    metabolic = pipeline == "metabolic"
    cosmetic = pipeline == "cosmetic"
    brief_stage = {
        "metabolic": "s01_metabolic_therapeutic_product_brief",
        "cosmetic": "s01_cosmetic_therapeutic_product_brief",
    }.get(pipeline, "s01_therapeutic_product_brief")
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

    # An absent stage defaults to enabled, so every other pipeline's opt-in stages are switched off explicitly.
    own = METABOLIC_STAGES if metabolic else COSMETIC_STAGES if cosmetic else ()
    for name in (*METABOLIC_STAGES, *COSMETIC_STAGES):
        if name not in own:
            stages.setdefault(name, {"enabled": False, "params": {}})

    if metabolic:
        # The metabolic stages replace their wound peers: those are forced off, these forced on.
        for name in REPLACED_STAGES:
            stages[name] = {"enabled": False, "params": {}}
        for name in METABOLIC_STAGES:
            stages.setdefault(name, {"enabled": True, "params": {}})["enabled"] = True
        if not stages[brief_stage]["params"].get("brief"):
            raise ValueError("the metabolic worker requires an inline Stage 1 brief")
    elif cosmetic:
        # The cosmetic stages replace their wound peers (Stage 7 is shared and stays on).
        for name in COSMETIC_REPLACED_STAGES:
            stages[name] = {"enabled": False, "params": {}}
        for name in COSMETIC_STAGES:
            stages.setdefault(name, {"enabled": True, "params": {}})["enabled"] = True
        if stages[brief_stage]["params"].get("brief") is None:
            raise ValueError("the cosmetic worker requires an inline Stage 1 brief (an empty one is allowed)")
    else:
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
        config.seed = config.for_stage(brief_stage).params.get("seed", 42)
    if not metabolic and not cosmetic:
        if not config.for_stage(brief_stage).enabled:
            raise ValueError("the e2e worker requires Stage 1")
        params = config.for_stage(brief_stage).params
        if not params.get("brief") and not (DEV_MODE and params.get("brief_path")):
            raise ValueError("the e2e worker requires an inline Stage 1 brief")
    return config
