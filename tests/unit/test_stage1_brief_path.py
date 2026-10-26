"""A worker config can carry `brief_path` for local runs (DEV_MODE); the deployed path still needs the brief inline."""

import json

import pytest

from schemas import e2e_config
from schemas.e2e_config import validate_job_config

BRIEF = {
    "wound_context": ["chronic"],
    "desired_functions": ["anti_inflammatory"],
    "pathogens": [],
    "min_length": 6,
    "max_length": 40,
    "dosing_interval_hours": 24,
}


def payload(stage1_params):
    return {
        "run_id": "t",
        "artifacts_dir": "./artifacts",
        "model_store": "./model_store",
        "seed_candidates_path": "seeds.fasta",
        "stages": {"s01_therapeutic_product_brief": {"enabled": True, "params": stage1_params}},
    }


def test_dev_mode_config_keeps_brief_path(monkeypatch, tmp_path):
    monkeypatch.setattr(e2e_config, "DEV_MODE", True)
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(BRIEF))
    config = validate_job_config(payload({"brief_path": str(path)}))
    assert config.for_stage("s01_therapeutic_product_brief").params["brief_path"] == str(path)


def test_deployed_config_still_needs_the_brief_inline(monkeypatch, tmp_path):
    monkeypatch.setattr(e2e_config, "DEV_MODE", False)
    with pytest.raises(ValueError, match="inline Stage 1 brief"):
        validate_job_config(payload({"brief_path": str(tmp_path / "brief.json")}))


def test_a_config_with_neither_is_refused(monkeypatch):
    monkeypatch.setattr(e2e_config, "DEV_MODE", True)
    with pytest.raises(ValueError, match="brief"):
        validate_job_config(payload({}))
