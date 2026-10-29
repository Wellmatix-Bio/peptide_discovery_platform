# Unit tests for the cosmetic Stage 9: synthesis-feasibility gate and the never-rejecting synthesis flags.
#
# The synthesis model is mocked so no real model or ESM is loaded.

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pipeline.base import StageError
from pipeline.s09_synthesis_cmc import cosmetic_stage
from pipeline.s09_synthesis_cmc.cosmetic_stage import CosmeticStage9, synthesis_flags
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage9CosmeticParams


class FakeExtractor:
    def __init__(self) -> None:
        self.warmed: list[list[str]] = []

    def get_esm2_embedding_batch(self, sequences):
        self.warmed.append(list(sequences))


def install(monkeypatch, probabilities=None):
    probabilities = probabilities or {}
    model = SimpleNamespace(predict_proba=lambda s, fe: probabilities.get(s, 0.9))
    monkeypatch.setattr(cosmetic_stage, "build_model", lambda: model)


def run(sequences, params=None, fe=None):
    candidates = [Candidate(id=s, sequence=s) for s in sequences]
    ctx = SimpleNamespace(feature_extractor=fe or FakeExtractor())
    return CosmeticStage9().run(candidates, StageConfig(enabled=True, params=params or {}), ctx)


def test_default_threshold_is_half() -> None:
    assert Stage9CosmeticParams().synthesis_feasibility_min == 0.5


def test_unknown_param_and_out_of_range_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Stage9CosmeticParams(synthesis_feasibility_min=1.5)


def test_rejects_below_threshold_and_keeps_at_threshold(monkeypatch) -> None:
    install(monkeypatch, {"GHKAS": 0.49, "GHKASS": 0.5})
    out = run(["GHKAS", "GHKASS", "KTTKS"])
    assert [c.sequence for c in out] == ["GHKASS", "KTTKS"]


def test_threshold_is_configurable(monkeypatch) -> None:
    install(monkeypatch, {"GHKAS": 0.7})
    assert run(["GHKAS"], {"synthesis_feasibility_min": 0.8}) == []
    assert len(run(["GHKAS"], {"synthesis_feasibility_min": 0.6})) == 1


def test_records_probability_and_threshold(monkeypatch) -> None:
    install(monkeypatch, {"GHKAS": 0.75})
    (c,) = run(["GHKAS"])
    record = c.predictions["synthesis_feasibility"]
    assert record == {"probability": 0.75, "min": 0.5, "pass": True, "flags": {}}


def test_flags_never_reject(monkeypatch) -> None:
    install(monkeypatch)
    (c,) = run(["MDGWK"])
    assert c.predictions["synthesis_feasibility"]["flags"] == {
        "aspartimide_risk": [1],
        "met_trp_cleavage_oxidation": [0, 3],
    }


def test_flag_rules() -> None:
    assert synthesis_flags("GHKAS") == {}
    assert synthesis_flags("ADGA") == {"aspartimide_risk": [1]}
    assert synthesis_flags("ANGA") == {}  # only Asp-Gly, not Asn-Gly
    assert synthesis_flags("GDA") == {}
    assert synthesis_flags("AWA") == {"met_trp_cleavage_oxidation": [1]}


def test_embedding_cache_is_warmed_once(monkeypatch) -> None:
    install(monkeypatch)
    fe = FakeExtractor()
    run(["GHKAS", "KTTKS"], fe=fe)
    assert fe.warmed == [["GHKAS", "KTTKS"]]


def test_empty_input_and_missing_sequence(monkeypatch) -> None:
    install(monkeypatch)
    assert run([]) == []
    with pytest.raises(StageError):
        CosmeticStage9().run(
            [Candidate(id="x", sequence="")],
            StageConfig(enabled=True, params={}),
            SimpleNamespace(feature_extractor=FakeExtractor()),
        )
