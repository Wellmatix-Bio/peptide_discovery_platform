"""Running food stages with the platform runner: the stage-builder hook, the empty start, and keeping food fields on reload."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from common.io import BoundaryWriter  # noqa: E402
from food.shared.io import load_food_candidates_jsonl  # noqa: E402
from food.pipeline.runner import FoodPipelineRunner  # noqa: E402
from registry import build_stages  # noqa: E402
from runner import PipelineRunner  # noqa: E402
from food.shared.candidate import FoodCandidate, ParentOrigin  # noqa: E402
from schemas.run_config import RunConfig  # noqa: E402


def config(tmp_path, seed="") -> RunConfig:
    return RunConfig(seed_candidates_path=seed, artifacts_dir=str(tmp_path), model_store=str(tmp_path), stages={}, run_id="t1")


def food_candidate(**kw) -> FoodCandidate:
    base = {
        "id": "PEP-1",
        "sequence": "AAK",
        "origin_type": "DIGESTION_DERIVED",
        "generation_method": "in_silico_digestion",
        "parents": [ParentOrigin(protein_id="P1:1-8", allergen_source=True, enzymes=["trypsin"], ph=7.5, start_in_product=1, end_in_product=3)],
        "mono_mass": 300.1,
    }
    return FoodCandidate(**{**base, **kw})


def test_the_default_stage_builder_is_the_one_the_other_pipelines_use():
    assert inspect.signature(PipelineRunner.__init__).parameters["stage_builder"].default is build_stages


def test_a_runner_uses_the_stage_builder_it_is_given(tmp_path):
    runner = PipelineRunner(config(tmp_path), stage_builder=lambda cfg: (["setup"], ["candidate"]))
    assert (runner.setup_stages, runner.candidate_stages) == (["setup"], ["candidate"])


def test_a_food_run_starts_with_no_candidates_and_needs_no_seed_file(tmp_path):
    assert FoodPipelineRunner(config(tmp_path))._load_entry_candidates() == []


def test_a_food_run_can_enter_from_a_stage_file(tmp_path):
    c = food_candidate()
    BoundaryWriter(str(tmp_path)).write("s03_food_digestion", [c], run_id="t1")
    seed = tmp_path / "candidates" / "s03_food_digestion.jsonl"
    (entry,) = FoodPipelineRunner(config(tmp_path, str(seed)))._load_entry_candidates()
    assert entry == c


def test_every_food_field_survives_a_save_and_reload(tmp_path):
    c = food_candidate(predictions={"x": 1}, duplicate_family_id="PEP-1", parent_clusters=["c1"], avg_mass=300.5, chemistry_notes="linear")
    BoundaryWriter(str(tmp_path)).write("s03", [c], run_id="t1")
    (back,) = load_food_candidates_jsonl(str(tmp_path / "candidates" / "s03.jsonl"))
    assert back == c
    assert back.any_parent_allergen_source is True


def test_a_missing_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        load_food_candidates_jsonl(str(tmp_path / "nope.jsonl"))
