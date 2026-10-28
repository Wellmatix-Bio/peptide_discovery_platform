# Unit tests for the cosmetic Stage 6: primary gate, antimicrobial reject filter, secondary scoring and configurable thresholds.
#
# The function models and the AMP classifier are mocked so no real model is loaded.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.base import StageError
from pipeline.s06_functional_models import cosmetic_stage
from pipeline.s06_functional_models.cosmetic_stage import CosmeticStage6
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage6CosmeticParams


def brief(primary="antioxidant", secondary=()):
    return SimpleNamespace(functions=SimpleNamespace(primary=primary, secondary=list(secondary)))


def ctx(primary="antioxidant", secondary=()):
    return SimpleNamespace(brief=brief(primary, secondary), feature_extractor=object())


class Mocks:
    """Scores per function and AMP probabilities, keyed by sequence; records every call."""

    def __init__(self, function_scores=None, amp=None) -> None:
        self.function_scores = function_scores or {}
        self.amp = amp or {}
        self.scored: list[tuple[str, list[str]]] = []
        self.amp_seen: list[list[str]] = []

    def install(self, monkeypatch) -> None:
        def score(function, sequences, fe):
            self.scored.append((function, list(sequences)))
            return [self.function_scores.get(function, {}).get(s, 0.9) for s in sequences]

        def amp(sequences, fe):
            self.amp_seen.append(list(sequences))
            return [self.amp.get(s, 0.1) for s in sequences]

        monkeypatch.setattr(cosmetic_stage, "score_function", score)
        monkeypatch.setattr(
            cosmetic_stage, "_get_amp_model", lambda: SimpleNamespace(predict_proba_batch=amp)
        )


def cands(*sequences):
    return [Candidate(id=s, sequence=s) for s in sequences]


def run(candidates, params=None, context=None):
    return CosmeticStage6().run(candidates, StageConfig(enabled=True, params=params or {}), context or ctx())


class TestDefaults:
    def test_thresholds_default_to_half_and_amp_to_point_six(self) -> None:
        p = Stage6CosmeticParams()
        assert (p.collagen_ecm_threshold, p.antioxidant_threshold, p.pigmentation_threshold) == (0.5, 0.5, 0.5)
        assert p.antimicrobial_reject_min == 0.6


class TestPrimaryGate:
    def test_pass_at_or_above_the_threshold(self, monkeypatch) -> None:
        Mocks({"antioxidant": {"AAA": 0.5, "BBB": 0.49}}).install(monkeypatch)
        out = run(cands("AAA", "BBB"))
        assert [c.sequence for c in out] == ["AAA"]

    @pytest.mark.parametrize("function", ["collagen_ecm", "antioxidant", "pigmentation"])
    def test_each_function_uses_its_own_model(self, monkeypatch, function: str) -> None:
        mocks = Mocks()
        mocks.install(monkeypatch)
        run(cands("AAA"), context=ctx(function))
        assert mocks.scored[0][0] == function

    def test_gate_result_is_recorded_on_the_candidate(self, monkeypatch) -> None:
        Mocks({"antioxidant": {"AAA": 0.7}}).install(monkeypatch)
        (out,) = run(cands("AAA"))
        assert out.predictions["functional_scores"]["antioxidant"] == {"score": 0.7, "threshold": 0.5, "pass": True}
        assert out.predictions["primary_function_pass"] is True
        assert out.predictions["functional_verdict"]["overall"] == "proceed"

    def test_threshold_is_configurable_per_function(self, monkeypatch) -> None:
        Mocks({"antioxidant": {"AAA": 0.7}, "collagen_ecm": {"AAA": 0.7}}).install(monkeypatch)
        assert run(cands("AAA"), {"antioxidant_threshold": 0.8}) == []
        assert len(run(cands("AAA"), {"antioxidant_threshold": 0.7})) == 1
        # Another function's threshold does not move the primary gate.
        assert len(run(cands("AAA"), {"collagen_ecm_threshold": 0.99})) == 1
        assert run(cands("AAA"), {"collagen_ecm_threshold": 0.99}, ctx("collagen_ecm")) == []


class TestAntimicrobialFilter:
    def test_rejects_at_or_above_point_six(self, monkeypatch) -> None:
        Mocks(amp={"AAA": 0.6, "BBB": 0.59}).install(monkeypatch)
        assert [c.sequence for c in run(cands("AAA", "BBB"))] == ["BBB"]

    def test_threshold_is_configurable(self, monkeypatch) -> None:
        Mocks(amp={"AAA": 0.7}).install(monkeypatch)
        assert run(cands("AAA")) == []
        assert len(run(cands("AAA"), {"antimicrobial_reject_min": 0.8})) == 1
        assert run(cands("AAA"), {"antimicrobial_reject_min": 0.5}) == []

    def test_only_candidates_that_passed_the_primary_gate_are_checked(self, monkeypatch) -> None:
        mocks = Mocks({"antioxidant": {"BBB": 0.1}})
        mocks.install(monkeypatch)
        run(cands("AAA", "BBB"))
        assert mocks.amp_seen == [["AAA"]]

    def test_probability_is_recorded(self, monkeypatch) -> None:
        Mocks(amp={"AAA": 0.2}).install(monkeypatch)
        (out,) = run(cands("AAA"))
        assert out.predictions["antimicrobial"] == {"probability": 0.2, "reject_min": 0.6}

    def test_no_survivors_of_the_gate_means_no_amp_call(self, monkeypatch) -> None:
        mocks = Mocks({"antioxidant": {"AAA": 0.0}})
        mocks.install(monkeypatch)
        assert run(cands("AAA")) == []
        assert mocks.amp_seen == []


class TestSecondaryFunctions:
    def test_scored_and_recorded_but_never_reject(self, monkeypatch) -> None:
        Mocks({"pigmentation": {"AAA": 0.0}}).install(monkeypatch)
        (out,) = run(cands("AAA"), context=ctx("antioxidant", ["pigmentation"]))
        assert out.predictions["functional_scores"]["pigmentation"]["pass"] is False
        assert out.predictions["functional_verdict"]["secondary_functions"] == ["pigmentation"]

    def test_only_functions_listed_in_the_brief_are_scored(self, monkeypatch) -> None:
        mocks = Mocks()
        mocks.install(monkeypatch)
        run(cands("AAA"), context=ctx("antioxidant", ["pigmentation"]))
        assert {f for f, _ in mocks.scored} == {"antioxidant", "pigmentation"}

    def test_stage_params_override_the_brief(self, monkeypatch) -> None:
        mocks = Mocks()
        mocks.install(monkeypatch)
        run(cands("AAA"), {"primary_function": "collagen_ecm", "secondary_functions": []}, ctx("antioxidant", ["pigmentation"]))
        assert {f for f, _ in mocks.scored} == {"collagen_ecm"}


class TestErrors:
    def test_no_primary_function_anywhere_is_an_error(self, monkeypatch) -> None:
        Mocks().install(monkeypatch)
        with pytest.raises(StageError):
            run(cands("AAA"), context=SimpleNamespace(brief=None, feature_extractor=object()))

    def test_candidate_without_a_sequence_is_an_error(self, monkeypatch) -> None:
        Mocks().install(monkeypatch)
        with pytest.raises(StageError):
            run([Candidate(id="x")])
