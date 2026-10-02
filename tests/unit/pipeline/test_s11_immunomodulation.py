# Unit tests for the s11_ranking immunomodulation module (immune alignment score).
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.s11_ranking.stage import (
    BUILTIN_RANKING_POLICY,
    CandidateScorer,
    ImmuneAlignment,
    ImmuneInputs,
    RankingInput,
    Stage11,
)
from schemas.brief import Brief
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

SPEC = BUILTIN_RANKING_POLICY.modules["immunomodulation"].measurement


def evaluate(nfkb, cytokine, context=()):
    return SPEC.evaluate(ImmuneInputs(nfkb=nfkb, cytokine=cytokine), list(context))


def test_builtin_module_uses_immune_alignment_on_the_pathway_probabilities():
    assert isinstance(SPEC, ImmuneAlignment)
    assert SPEC.nfkb_source == "mechanism.pathway_involvement.NF_KB.probability"
    assert SPEC.cytokine_source == "mechanism.pathway_involvement.CYTOKINE_MACROPHAGE.probability"


def test_each_probability_becomes_a_direction():
    _, detail = evaluate(0.0, 1.0)
    assert detail["direction_nfkb"] == pytest.approx(1.0)  # surely an inhibitor
    assert detail["direction_cytokine"] == pytest.approx(-1.0)  # surely an activator
    assert evaluate(0.5, 0.5)[1]["direction"] == pytest.approx(0.0)


def test_directions_are_blended_equally():
    assert evaluate(0.1, 0.5)[1]["direction"] == pytest.approx(0.5 * 0.8 + 0.5 * 0.0)


def test_blend_weight_is_configurable():
    spec = SPEC.model_copy(update={"nfkb_weight": 0.8})
    _, detail = spec.evaluate(ImmuneInputs(nfkb=0.1, cytokine=0.5), [])
    assert detail["direction"] == pytest.approx(0.8 * 0.8 + 0.2 * 0.0)


@pytest.mark.parametrize(
    "context, expected",
    [
        ([], 0.0),
        (["chronic"], 0.7),
        (["diabetic"], 0.7),
        (["surgical"], 0.3),
        (["burn"], 0.4),
        (["infected"], -0.1),
        (["diabetic", "infected"], -0.1),  # the minimum wins
        (["chronic", "diabetic", "high_glucose"], 0.7),  # not additive
        (["surgical", "traumatic"], 0.1),
        (["acute"], 0.0),
        (["clean"], 0.0),  # not in the table
        (["clean", "surgical"], 0.3),  # unknown tags are ignored
        (["chronic", "chronic"], 0.7),
    ],
)
def test_target_direction_is_the_minimum_over_the_tags(context, expected):
    assert SPEC.target_direction(context) == pytest.approx(expected)


def test_score_formula_matches_the_spec():
    # d_n = 0.6, d_c = 0.0, d = 0.3, d* = 0.7 (chronic)
    score, detail = evaluate(0.2, 0.5, ["chronic"])
    assert detail["direction"] == pytest.approx(0.3)
    assert detail["target_direction"] == pytest.approx(0.7)
    assert detail["match"] == pytest.approx(1 - 0.4 / 1.7)
    assert detail["agreement"] == pytest.approx(1 - 0.6 / 2)
    assert score == pytest.approx((1 - 0.4 / 1.7) * (0.5 + 0.5 * 0.7))


def test_a_perfect_match_with_full_agreement_scores_one():
    # p = 0.15 on both: d = 0.7 = d* (chronic)
    score, _ = evaluate(0.15, 0.15, ["chronic"])
    assert score == pytest.approx(1.0)


def test_disagreement_discounts_the_score():
    agree, _ = evaluate(0.5, 0.5, [])  # d = 0
    split, detail = evaluate(0.0, 1.0, [])  # d = 0 too, but the pathways disagree fully
    assert detail["agreement"] == 0.0
    assert split == pytest.approx(agree * 0.5)


def test_worst_possible_miss_scores_zero():
    # d* = 0.7 and d = -1: |d - d*| = 1.7 = 1 + |d*|
    score, detail = evaluate(1.0, 1.0, ["chronic"])
    assert detail["match"] == pytest.approx(0.0)
    assert score == 0.0


def test_infected_wound_penalizes_strong_suppression():
    strong_inhibitor, _ = evaluate(0.0, 0.0, ["infected"])  # d = 1, d* = -0.1
    balanced, _ = evaluate(0.55, 0.55, ["infected"])  # d = -0.1
    assert balanced > strong_inhibitor


def test_score_never_leaves_zero_to_one():
    for p in (0.0, 0.2, 0.5, 1.0):
        for q in (0.0, 0.5, 1.0):
            for context in ([], ["chronic", "diabetic", "burn"], ["infected", "necrotic"]):
                score, _ = evaluate(p, q, context)
                assert 0.0 <= score <= 1.0


def test_a_missing_pathway_probability_makes_the_score_missing():
    assert evaluate(None, 0.5) == (None, None)
    assert evaluate(0.5, None) == (None, None)


def test_scorer_uses_the_computed_score_and_reports_its_parts():
    scorer = CandidateScorer(BUILTIN_RANKING_POLICY)
    result = scorer.score(RankingInput(
        candidate_id="c1", sequence="KLLK",
        brief=Brief(min_length=5, max_length=30, desired_functions=["immunomodulation"], wound_context=["chronic"]),
        immune_inputs=ImmuneInputs(nfkb=0.2, cytokine=0.5),
    ))
    module = result.modules["immunomodulation"]
    assert module.raw_measurement == pytest.approx((1 - 0.4 / 1.7) * 0.85)
    assert module.score == pytest.approx(module.raw_measurement)
    assert module.measurement_detail["target_direction"] == pytest.approx(0.7)
    assert module.measurement_source.startswith("immune_alignment(")
    assert module.activated is True


def test_scorer_marks_missing_pathways_as_missing_evidence():
    result = CandidateScorer(BUILTIN_RANKING_POLICY).score(RankingInput(
        candidate_id="c1", sequence="KLLK",
        brief=Brief(min_length=5, max_length=30, desired_functions=["immunomodulation"]),
    ))
    assert "immunomodulation" in result.missing_modules
    assert "immunomodulation" in result.requested_modules_without_data
    assert result.modules["immunomodulation"].measurement_detail is None


def test_run_reads_the_pathway_sources_from_predictions():
    candidate = Candidate(id="c1", sequence="KLLK", predictions={
        "anti_inflammatory_probability": 0.8,
        "mechanism": {"pathway_involvement": {
            "NF_KB": {"probability": 0.2}, "CYTOKINE_MACROPHAGE": {"probability": 0.5},
        }},
    })
    ctx = SimpleNamespace(
        run_id="r", brief=Brief(
            min_length=5, max_length=30, wound_context=["chronic"], desired_functions=["immunomodulation"],
        ),
    )
    ranking = Stage11().run([candidate], StageConfig(params={}), ctx)[0].predictions["ranking"]
    module = ranking["modules"]["immunomodulation"]
    assert module["measurement_detail"]["direction"] == pytest.approx(0.3)
    assert module["score"] == pytest.approx((1 - 0.4 / 1.7) * 0.85)
    anti = ranking["modules"]["anti_inflammatory"]
    assert anti["raw_measurement"] == 0.8
    assert anti["n_k"] == 0.5  # chronic implies it; only immunomodulation was referenced
