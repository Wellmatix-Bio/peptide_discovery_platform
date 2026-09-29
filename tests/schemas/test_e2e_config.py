from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from schemas.e2e_config import validate_job_config
from schemas.stage_configs import E2ERequest


def payload():
    root = Path(__file__).resolve().parents[2]
    request = json.loads((root / "src/backend/api_e2e/example_request.json").read_text())
    return {"run_id": "123", "artifacts_dir": "./artifacts", "model_store": "./model_store",
            "seed_candidates_path": "seeds.fasta", "stages": request["stages"]}


def test_defaults_seed_and_no_input_mutation():
    data = payload()
    data["stages"]["s01_therapeutic_product_brief"]["seed"] = 7
    data["stages"]["s05_physchem_screening"] = {"ph": 6.5, "unknown": "ignored"}
    del data["stages"]["s06_functional_models"]
    before = deepcopy(data)
    config = validate_job_config(data)
    assert data == before
    assert config.seed == 7
    params = config.for_stage("s05_physchem_screening").params
    assert params["ph"] == 6.5
    assert params["aggregation_tendency_flag_max"] == 0.6
    assert "unknown" not in params
    assert config.for_stage("s06_functional_models").params["stage6_thresholds"]["min_amp_probability"] == 0.70
    assert validate_job_config(config.model_dump()).model_dump() == config.model_dump()


@pytest.mark.parametrize("field,value", [("max_length", 51), ("desired_functions", ["antibiofilm"]),
                                         ("pathogens", ["unsupported"])])
def test_brief_schema_enforced(field, value):
    data = payload()
    data["stages"]["s01_therapeutic_product_brief"]["brief"][field] = value
    with pytest.raises(ValidationError):
        validate_job_config(data)


def test_correct_schema_selected_for_stage8():
    data = payload()
    data["stages"]["s08_safety_developability"]["cytotoxicity_cell_type"] = "invalid"
    with pytest.raises(ValidationError):
        validate_job_config(data)


def test_disabled_stage_skips_parameter_validation():
    data = payload()
    data["stages"]["s08_safety_developability"] = {"enabled": False, "cytotoxicity_cell_type": "invalid"}
    assert not validate_job_config(data).for_stage("s08_safety_developability").enabled


def test_e2e_request_requires_every_client_configurable_stage():
    data = payload()
    del data["stages"]["s06_functional_models"]
    with pytest.raises(ValidationError, match="s06_functional_models"):
        E2ERequest.model_validate(data["stages"])


def test_e2e_request_accepts_empty_stage_as_defaults():
    data = payload()
    data["stages"]["s06_functional_models"] = {}
    request = E2ERequest.model_validate(data["stages"])
    assert request.s06_functional_models.stage6_thresholds.min_amp_probability == 0.70


def test_e2e_request_rejects_unknown_stage():
    data = payload()
    data["stages"]["s99_made_up"] = {}
    with pytest.raises(ValidationError, match="s99_made_up"):
        E2ERequest.model_validate(data["stages"])


def test_e2e_request_rejects_s02_s03_s11():
    """s02/s03/s11 are server-forced regardless of client input (see
    e2e_config.py) -- E2ERequest excludes them rather than accepting and
    silently discarding whatever a client sends for them."""
    data = payload()
    data["stages"]["s11_ranking"] = {}
    with pytest.raises(ValidationError, match="s11_ranking"):
        E2ERequest.model_validate(data["stages"])
