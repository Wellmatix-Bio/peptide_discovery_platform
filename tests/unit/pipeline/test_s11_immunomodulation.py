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


def evaluate(nfkb, cytokine, anti, context=()):
    return SPEC.evaluate(
        ImmuneInputs(nfkb=nfkb, cytokine=cytokine, anti_inflammatory=anti), list(context)
    )


def test_builtin_module_uses_immune_alignment_on_the_pathway_probabilities():
    assert isinstance(SPEC, ImmuneAlignment)
    assert SPEC.nfkb_source == "mechanism.pathway_involvement.NF_KB.probability"
    assert SPEC.cytokine_source == "mechanism.pathway_involvement.CYTOKINE_MACROPHAGE.probability"
    assert SPEC.direction_source == "anti_inflammatory_probability"


def test_relevance_is_high_if_either_pathway_is_likely():
    r = lambda a, b: evaluate(a, b, 0.5)[1]["relevance"]
    assert r(0.8, 0.0) == pytest.approx(0.8)
    assert r(0.0, 0.8) == pytest.approx(0.8)
    assert r(0.8, 0.5) == pytest.approx(1 - 0.2 * 0.5)
    assert r(0.0, 0.0) == 0.0


def test_low_relevance_gives_a_low_score_whatever_the_direction():
    score, _ = evaluate(0.05, 0.05, 1.0, ["chronic"])
    assert score < 0.1


def test_direction_is_two_p_minus_one():
    assert evaluate(0.5, 0.5, 1.0)[1]["direction"] == pytest.approx(1.0)
    assert evaluate(0.5, 0.5, 0.5)[1]["direction"] == pytest.approx(0.0)
    assert evaluate(0.5, 0.5, 0.0)[1]["direction"] == pytest.approx(-1.0)


def test_unknown_direction_is_zero_and_earns_no_reward_or_penalty():
    score, detail = evaluate(0.6, 0.6, None)
    assert detail["direction"] == 0.0
    assert detail["pro_inflammatory_penalty"] == 0.0
    assert score == pytest.approx(detail["relevance"])  # no context: d* = 0


@pytest.mark.parametrize(
    "context, expected",
    [
        ([], 0.0),
        (["chronic"], 0.40),
        (["diabetic"], 0.30),
        (["chronic", "diabetic"], 0.70),  # additive
        (["chronic", "diabetic", "high_glucose"], 0.90),
        (["infected"], -0.20),
        (["infected", "chronic"], 0.20),
        (["acute"], 0.0),
        (["clean"], 0.0),  # not in the offset table
        (["chronic", "chronic"], 0.40),  # a repeated tag counts once
    ],
)
def test_target_direction_adds_the_context_offsets(context, expected):
    assert SPEC.target_direction(context) == pytest.approx(expected)


def test_target_direction_is_clamped_to_plus_minus_one():
    heavy = ["chronic", "diabetic", "high_glucose", "burn", "surgical", "ischemic"]
    assert SPEC.target_direction(heavy) == 1.0
    assert SPEC.target_direction(["infected", "necrotic", "biofilm_positive"]) == pytest.approx(-0.55)


def test_score_formula_matches_the_spec():
    # R = 1 - 0.3 * 0.5 = 0.85, d = 0.6, d* = 0.4 (chronic)
    score, detail = evaluate(0.7, 0.5, 0.8, ["chronic"])
    assert detail["relevance"] == pytest.approx(0.85)
    assert detail["direction"] == pytest.approx(0.6)
    assert detail["target_direction"] == pytest.approx(0.4)
    assert score == pytest.approx(0.85 * (1 - abs(0.6 - 0.4) / 2))


def test_matching_the_target_direction_scores_the_relevance():
    # d* = 0.4 (chronic); P(anti-inflammatory) = 0.7 gives d = 0.4
    score, detail = evaluate(0.6, 0.6, 0.7, ["chronic"])
    assert score == pytest.approx(detail["relevance"])


def test_pro_inflammatory_peptide_is_penalized():
    score, detail = evaluate(0.8, 0.8, 0.2, [])  # d = -0.6, d* = 0, R = 0.96
    assert detail["pro_inflammatory_penalty"] == pytest.approx(0.96 * 0.6)
    assert score == pytest.approx(max(0.0, 0.96 * (1 - 0.6 / 2) - 0.96 * 0.6))
    calm, _ = evaluate(0.8, 0.8, 0.8, [])
    assert score < calm


def test_score_never_leaves_zero_to_one():
    for p in (0.0, 0.2, 0.5, 1.0):
        for context in ([], ["chronic", "diabetic", "burn"], ["infected", "necrotic"]):
            score, _ = evaluate(1.0, 1.0, p, context)
            assert 0.0 <= score <= 1.0


def test_a_missing_pathway_probability_makes_the_score_missing():
    assert evaluate(None, 0.5, 0.8) == (None, None)
    assert evaluate(0.5, None, 0.8) == (None, None)


def test_infected_context_prefers_a_less_anti_inflammatory_peptide():
    strongly_anti, _ = evaluate(0.7, 0.7, 0.95, ["infected"])
    neutral, _ = evaluate(0.7, 0.7, 0.4, ["infected"])
    assert neutral > strongly_anti


def test_scorer_uses_the_computed_score_and_reports_its_parts():
    scorer = CandidateScorer(BUILTIN_RANKING_POLICY)
    result = scorer.score(RankingInput(
        candidate_id="c1", sequence="KLLK",
        brief=Brief(min_length=5, max_length=30, desired_functions=["immunomodulation"], wound_context=["chronic"]),
        immune_inputs=ImmuneInputs(nfkb=0.7, cytokine=0.5, anti_inflammatory=0.8),
    ))
    module = result.modules["immunomodulation"]
    assert module.raw_measurement == pytest.approx(0.85 * 0.9)
    assert module.score == pytest.approx(module.raw_measurement)
    assert module.measurement_detail["target_direction"] == pytest.approx(0.4)
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


def test_run_reads_the_pathway_and_direction_sources_from_predictions():
    candidate = Candidate(id="c1", sequence="KLLK", predictions={
        "anti_inflammatory_probability": 0.8,
        "mechanism": {"pathway_involvement": {
            "NF_KB": {"probability": 0.7}, "CYTOKINE_MACROPHAGE": {"probability": 0.5},
        }},
    })
    ctx = SimpleNamespace(
        run_id="r", brief=Brief(
            min_length=5, max_length=30, wound_context=["chronic"], desired_functions=["immunomodulation"],
        ),
    )
    ranking = Stage11().run([candidate], StageConfig(params={}), ctx)[0].predictions["ranking"]
    module = ranking["modules"]["immunomodulation"]
    assert module["measurement_detail"]["relevance"] == pytest.approx(0.85)
    assert module["score"] == pytest.approx(0.85 * 0.9)
    anti = ranking["modules"]["anti_inflammatory"]
    assert anti["raw_measurement"] == 0.8
    assert anti["n_k"] == 0.5  # chronic implies it; only immunomodulation was referenced
