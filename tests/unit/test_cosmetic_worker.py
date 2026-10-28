# Unit tests for the cosmetic worker's config: stage selection, forced-on/off stages and the startup stage contracts.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.worker_cosmetic import worker
from registry import build_stages
from runner import PipelineRunner
from schemas.candidate import Candidate
from schemas.cosmetic_config import COSMETIC_REPLACED_STAGES, COSMETIC_STAGES
from schemas.e2e_config import validate_job_config

EXAMPLE = Path(__file__).resolve().parents[2] / "src" / "backend" / "worker_cosmetic" / "example_run.json"


def payload(**overrides) -> dict:
    p = json.loads(EXAMPLE.read_text())
    p.update(overrides)
    return p


def test_example_run_validates() -> None:
    config = validate_job_config(payload(), pipeline="cosmetic")
    for name in COSMETIC_STAGES:
        assert config.for_stage(name).enabled
    for name in COSMETIC_REPLACED_STAGES:
        assert not config.for_stage(name).enabled
    assert config.for_stage("s07_structure_mechanism").enabled


def test_stages_are_built_in_pipeline_order() -> None:
    config = validate_job_config(payload(), pipeline="cosmetic")
    setup, candidate = build_stages(config)
    assert [s.name for s in setup] == ["s01_cosmetic_therapeutic_product_brief"]
    assert [s.name for s in candidate] == [
        "s04_cosmetic_candidate_generation",
        "s05_cosmetic_physchem_screening",
        "s06_cosmetic_functional_models",
        "s07_structure_mechanism",
        "s08_cosmetic_safety_developability",
        "s09_cosmetic_synthesis_cmc",
        "s11_cosmetic_ranking",
    ]


def test_stage_contracts_are_satisfiable_from_seed_candidates() -> None:
    config = validate_job_config(payload(), pipeline="cosmetic")
    _, candidate_stages = build_stages(config)
    runner = SimpleNamespace(candidate_stages=candidate_stages)
    PipelineRunner._validate_contracts(runner, [Candidate(id="s", sequence="GHK")])


def test_empty_brief_is_allowed_but_a_missing_one_is_not() -> None:
    p = payload()
    p["stages"]["s01_cosmetic_therapeutic_product_brief"]["params"]["brief"] = {}
    config = validate_job_config(p, pipeline="cosmetic")
    assert config.for_stage("s01_cosmetic_therapeutic_product_brief").params["brief"]["functions"]["primary"] == "collagen_ecm"
    del p["stages"]["s01_cosmetic_therapeutic_product_brief"]["params"]["brief"]
    with pytest.raises(ValueError, match="Stage 1 brief"):
        validate_job_config(p, pipeline="cosmetic")


def test_stage_params_are_resolved_over_defaults() -> None:
    p = payload()
    p["stages"]["s11_cosmetic_ranking"] = {"points_primary": 6}
    config = validate_job_config(p, pipeline="cosmetic")
    assert config.for_stage("s11_cosmetic_ranking").params["points_primary"] == 6
    assert config.for_stage("s08_cosmetic_safety_developability").params["allergenicity_min_length"] == 50


def test_other_pipelines_do_not_build_cosmetic_stages() -> None:
    wound = json.loads((EXAMPLE.parent.parent / "worker_e2e" / "example_run.json").read_text())
    for pipeline, example in (("wound_healing", wound), ("metabolic", json.loads((EXAMPLE.parent.parent / "worker_metabolic" / "example_run.json").read_text()))):
        _, candidate = build_stages(validate_job_config(example, pipeline=pipeline))
        assert not any("cosmetic" in s.name for s in candidate), pipeline


def test_worker_loads_the_cosmetic_pipeline(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text(EXAMPLE.read_text())
    config = worker.load_job_config(str(path))
    assert config.for_stage("s11_cosmetic_ranking").enabled


def test_unknown_pipeline_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown pipeline"):
        validate_job_config(payload(), pipeline="nope")
