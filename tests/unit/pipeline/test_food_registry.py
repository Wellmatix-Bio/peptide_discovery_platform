"""The food stage registry: same contract as `registry.build_stages`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.base import CandidateStage, SetupStage  # noqa: E402
from pipeline.food import registry  # noqa: E402
from schemas.run_config import RunConfig, StageConfig  # noqa: E402


def config(**stages) -> RunConfig:
    return RunConfig(seed_candidates_path="x", artifacts_dir="x", model_store="x", stages={k: StageConfig(**v) for k, v in stages.items()})


class FakeSetup(SetupStage):
    name = "fake_setup"

    def run(self, config, ctx):
        return {}


class FakeCandidate(CandidateStage):
    name = "fake_candidate"

    def run(self, candidates, config, ctx):
        return candidates


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(registry, "FOOD_SETUP_STAGE_CLASSES", [FakeSetup])
    monkeypatch.setattr(registry, "FOOD_CANDIDATE_STAGE_CLASSES", [FakeCandidate])


def test_the_registered_food_stages_are_built_in_order():
    setup, candidate = registry.build_food_stages(config())
    assert setup == []
    assert [s.name for s in candidate] == ["s03_food_digestion", "s04_food_generation", "s05_food_screening", "s06_food_gut_stability", "s07_food_ranking_decision"]


def test_a_stage_missing_from_the_run_config_is_enabled(fakes):
    setup, candidate = registry.build_food_stages(config())
    assert [type(s) for s in setup] == [FakeSetup]
    assert [type(s) for s in candidate] == [FakeCandidate]


def test_a_disabled_stage_is_left_out(fakes):
    setup, candidate = registry.build_food_stages(config(fake_setup={"enabled": False}, fake_candidate={"enabled": False}))
    assert (setup, candidate) == ([], [])
