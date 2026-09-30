# Unit tests for s11_ranking.
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pipeline.s11_ranking.stage import (
    COMPONENTS,
    CandidateRanker,
    CandidateScorer,
    ComponentScores,
    InputSources,
    ModifierRule,
    NormalizerConfig,
    ProductObjective,
    RankingConfig,
    RankingInput,
    Stage11,
    Stage11Service,
)
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

BASELINE_WEIGHTS = {
    "wound_closure": 0.22,
    "antimicrobial": 0.18,
    "immunomodulation": 0.13,
    "angiogenesis": 0.12,
    "collagen_ecm": 0.10,
    "safety": 0.12,
    "stability": 0.08,
    "synthesis_feasibility": 0.03,
    "mechanistic_confidence": 0.02,
}

FULL_SCORES = {
    "wound_closure": 0.82, "antimicrobial": 0.91, "immunomodulation": 0.74,
    "angiogenesis": 0.68, "collagen_ecm": 0.59, "safety": 0.88,
    "stability": 0.71, "synthesis_feasibility": 0.94, "mechanistic_confidence": 0.76,
}


def make_config(**overrides) -> RankingConfig:
    kwargs = {
        "baseline_weights": dict(BASELINE_WEIGHTS),
        "modifier_rules": [],
        "normalizers": {},
        "input_sources": InputSources(),
    }
    kwargs.update(overrides)
    return RankingConfig(**kwargs)


def make_input(candidate_id="c1", scores=None, stage1=None) -> RankingInput:
    return RankingInput(
        candidate_id=candidate_id,
        sequence="KLLKLLKK",
        stage1=stage1 or ProductObjective(),
        scores=ComponentScores(**(scores if scores is not None else FULL_SCORES)),
    )


# ------------------------------------------------------------------
# Baseline score calculation
# ------------------------------------------------------------------


def test_baseline_score_calculation():
    config = make_config()
    result = CandidateScorer(config).score(make_input())
    assert result.status == "ranked"
    expected = sum(BASELINE_WEIGHTS[c] * FULL_SCORES[c] for c in COMPONENTS)
    assert result.final_score == pytest.approx(expected, abs=1e-9)


# ------------------------------------------------------------------
# Weights sum to 1
# ------------------------------------------------------------------


def test_baseline_weights_must_sum_to_one():
    bad_weights = dict(BASELINE_WEIGHTS)
    bad_weights["safety"] += 0.1
    with pytest.raises(ValidationError, match="sum to 1.0"):
        make_config(baseline_weights=bad_weights)


# ------------------------------------------------------------------
# Infected-wound weight adjustment
# ------------------------------------------------------------------


def test_infected_wound_increases_antimicrobial_and_immunomodulation():
    config = make_config(modifier_rules=[
        ModifierRule(
            reason="infected_wound", field="wound_context", any_of=["infected"],
            modifiers={"antimicrobial": 1.25, "immunomodulation": 1.15},
        ),
    ])
    stage1 = ProductObjective(wound_context=["infected"])
    result = CandidateScorer(config).score(make_input(stage1=stage1))
    assert result.adjusted_weights["antimicrobial"] > result.baseline_weights["antimicrobial"]
    assert result.adjusted_weights["immunomodulation"] > result.baseline_weights["immunomodulation"]
    reasons = {a.reason for a in result.weight_adjustments}
    assert reasons == {"infected_wound"}


# ------------------------------------------------------------------
# Biological-objective weight adjustment
# ------------------------------------------------------------------


def test_desired_function_increases_matching_component():
    config = make_config(modifier_rules=[
        ModifierRule(
            reason="angiogenesis_objective", field="desired_functions", any_of=["angiogenesis"],
            modifiers={"angiogenesis": 1.25},
        ),
    ])
    stage1 = ProductObjective(desired_functions=["angiogenesis"])
    result = CandidateScorer(config).score(make_input(stage1=stage1))
    assert result.adjusted_weights["angiogenesis"] > result.baseline_weights["angiogenesis"]


# ------------------------------------------------------------------
# Weight renormalization
# ------------------------------------------------------------------


def test_adjusted_weights_still_sum_to_one():
    config = make_config(modifier_rules=[
        ModifierRule(
            reason="infected_wound", field="wound_context", any_of=["infected"],
            modifiers={"antimicrobial": 1.25, "immunomodulation": 1.15},
        ),
        ModifierRule(
            reason="diabetic_or_chronic_wound", field="wound_context",
            any_of=["diabetic", "chronic"],
            modifiers={"wound_closure": 1.20, "angiogenesis": 1.20, "stability": 1.20},
        ),
    ])
    stage1 = ProductObjective(wound_context=["infected", "chronic"])
    result = CandidateScorer(config).score(make_input(stage1=stage1))
    assert math.fsum(result.adjusted_weights.values()) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------
# Missing score handling -- NOT interpreted as zero
# ------------------------------------------------------------------


def test_missing_score_excluded_and_renormalized_not_zeroed():
    scores = dict(FULL_SCORES)
    del scores["collagen_ecm"]
    config = make_config()
    result = CandidateScorer(config).score(make_input(scores=scores))

    assert result.normalized_scores.collagen_ecm is None
    assert "collagen_ecm" in result.missing_components
    assert result.incomplete_evidence is True

    remaining_weight = math.fsum(
        BASELINE_WEIGHTS[c] for c in COMPONENTS if c != "collagen_ecm"
    )
    assert result.evidence_coverage == pytest.approx(remaining_weight, abs=1e-9)

    expected = math.fsum(
        BASELINE_WEIGHTS[c] * FULL_SCORES[c] for c in COMPONENTS if c != "collagen_ecm"
    ) / remaining_weight
    assert result.final_score == pytest.approx(expected, abs=1e-6)


# ------------------------------------------------------------------
# Deterministic ranking
# ------------------------------------------------------------------


def test_ranking_is_deterministic_given_same_inputs_and_config():
    config = make_config()
    inputs = [
        make_input("a", scores={**FULL_SCORES, "antimicrobial": 0.9}),
        make_input("b", scores={**FULL_SCORES, "antimicrobial": 0.5}),
        make_input("c", scores={**FULL_SCORES, "antimicrobial": 0.7}),
    ]
    batch1 = Stage11Service(config).rank_batch(inputs)
    batch2 = Stage11Service(config).rank_batch(inputs)
    order1 = [r.candidate_id for r in batch1.ranked_candidates]
    order2 = [r.candidate_id for r in batch2.ranked_candidates]
    assert order1 == order2 == ["a", "c", "b"]
    assert [r.final_score for r in batch1.ranked_candidates] == [
        r.final_score for r in batch2.ranked_candidates
    ]


# ------------------------------------------------------------------
# Tie-breaking
# ------------------------------------------------------------------


def test_tie_break_prefers_higher_safety_then_coverage_then_stability_then_id():
    config = make_config()
    scores_a = dict(FULL_SCORES)
    scores_b = dict(FULL_SCORES)
    scores_a["safety"] = 0.95
    scores_b["safety"] = 0.95
    inputs = [make_input("zzz", scores=scores_a), make_input("aaa", scores=scores_b)]
    batch = Stage11Service(config).rank_batch(inputs)
    assert [r.candidate_id for r in batch.ranked_candidates] == ["aaa", "zzz"]


def test_tie_break_prefers_higher_safety_on_equal_final_score():
    config = make_config(baseline_weights={**BASELINE_WEIGHTS, "safety": 0.0, "wound_closure": 0.34})
    scores_high_safety = {**FULL_SCORES, "safety": 0.99}
    scores_low_safety = {**FULL_SCORES, "safety": 0.10}
    inputs = [
        make_input("low", scores=scores_low_safety),
        make_input("high", scores=scores_high_safety),
    ]
    batch = Stage11Service(config).rank_batch(inputs)
    ids = [r.candidate_id for r in batch.ranked_candidates]
    assert ids[0] == "high"


# ------------------------------------------------------------------
# Evidence coverage calculation
# ------------------------------------------------------------------


def test_evidence_coverage_reflects_fraction_of_weight_available():
    scores = dict(FULL_SCORES)
    del scores["synthesis_feasibility"]  # weight 0.03
    del scores["mechanistic_confidence"]  # weight 0.02
    config = make_config()
    result = CandidateScorer(config).score(make_input(scores=scores))
    assert result.evidence_coverage == pytest.approx(0.95, abs=1e-9)


# ------------------------------------------------------------------
# Contribution values sum to final score
# ------------------------------------------------------------------


def test_contributions_sum_to_final_score():
    scores = dict(FULL_SCORES)
    del scores["collagen_ecm"]
    config = make_config()
    result = CandidateScorer(config).score(make_input(scores=scores))
    total_contribution = math.fsum(
        c.contribution for c in result.component_contributions.values() if c.contribution is not None
    )
    assert total_contribution == pytest.approx(result.final_score, abs=1e-9)


# ------------------------------------------------------------------
# Malformed configuration
# ------------------------------------------------------------------


def test_malformed_config_missing_component_raises():
    incomplete_weights = dict(BASELINE_WEIGHTS)
    del incomplete_weights["safety"]
    incomplete_weights["wound_closure"] += 0.12  # keep sum at 1.0
    with pytest.raises(ValidationError, match="exactly all nine components"):
        make_config(baseline_weights=incomplete_weights)


def test_malformed_config_negative_weight_raises():
    bad_weights = dict(BASELINE_WEIGHTS)
    bad_weights["safety"] = -0.5
    bad_weights["wound_closure"] += 0.62
    with pytest.raises(ValidationError):
        make_config(baseline_weights=bad_weights)


def test_malformed_config_duplicate_modifier_reason_raises():
    rule = ModifierRule(
        reason="dup", field="wound_context", any_of=["infected"], modifiers={"safety": 1.1},
    )
    with pytest.raises(ValidationError, match="unique"):
        make_config(modifier_rules=[rule, rule])


def test_malformed_config_linear_normalizer_without_bounds_raises():
    with pytest.raises(ValidationError, match="lower < upper"):
        NormalizerConfig(kind="linear")


def test_malformed_config_rejects_unknown_top_level_key():
    payload = {
        "baseline_weights": dict(BASELINE_WEIGHTS),
        "not_a_real_field": True,
    }
    with pytest.raises(ValidationError):
        RankingConfig(**payload)


def test_malformed_input_sources_bad_path_raises():
    with pytest.raises(ValidationError, match="nonempty dot-separated keys"):
        InputSources(scores={"antimicrobial": ""})


# ------------------------------------------------------------------
# All component scores missing -> insufficient_evidence, not 0
# ------------------------------------------------------------------


def test_all_scores_missing_yields_insufficient_evidence_not_zero_score():
    config = make_config()
    result = CandidateScorer(config).score(make_input(scores={}))
    assert result.status == "insufficient_evidence"
    assert result.final_score is None
    assert result.evidence_coverage == 0.0
    assert set(result.missing_components) == set(COMPONENTS)
    assert result.rank is None


# ------------------------------------------------------------------
# Batch summary, ranker only ranks "ranked" status, normalizer inversion.
# ------------------------------------------------------------------


def test_batch_summary_counts():
    config = make_config()
    ranked_input = make_input("ranked1")
    insufficient_input = make_input("insufficient1", scores={})
    batch = Stage11Service(config).rank_batch([ranked_input, insufficient_input])
    assert batch.total_candidates == 2
    assert batch.insufficient_evidence == 1
    assert len(batch.ranked_candidates) == 1
    assert batch.ranked_candidates[0].rank == 1


def test_normalizer_inverts_when_higher_is_better_false():
    config = make_config(normalizers={
        "safety": NormalizerConfig(kind="probability", higher_is_better=False),
    })
    scores = dict(FULL_SCORES)
    scores["safety"] = 0.2  # raw P(cytotoxic)-style value; inverted -> 0.8
    result = CandidateScorer(config).score(make_input(scores=scores))
    assert result.normalized_scores.safety == pytest.approx(0.8, abs=1e-9)


def test_linear_normalizer_maps_range_to_unit_interval():
    config = make_config(normalizers={
        "stability": NormalizerConfig(kind="linear", lower=0.0, upper=200.0, higher_is_better=True),
    })
    scores = dict(FULL_SCORES)
    scores["stability"] = 100.0  # e.g. a half-life in hours, not [0,1]
    result = CandidateScorer(config).score(make_input(scores=scores))
    assert result.normalized_scores.stability == pytest.approx(0.5, abs=1e-9)


def test_ranker_never_ranks_insufficient_evidence_candidates():
    result = CandidateRanker().rank([
        CandidateScorer(make_config()).score(make_input("has_data")),
        CandidateScorer(make_config()).score(make_input("no_data", scores={})),
    ])
    assert [r.candidate_id for r in result] == ["has_data"]


# ------------------------------------------------------------------
# Pipeline adapter (Stage11.run/execute): config sourcing, missing config,
# input_sources dot-path mapping from Candidate.predictions, brief wiring.
# ------------------------------------------------------------------


class _FakeAuditWriter:
    def __init__(self):
        self.calls = []

    def record_stage(self, result, *, run_id):
        self.calls.append(("record_stage", result, run_id))

    def record_setup(self, result, *, run_id):
        self.calls.append(("record_setup", result, run_id))

    def record_failure(self, stage, exc, *, run_id):
        self.calls.append(("record_failure", stage, exc, run_id))


class _FakeBoundaryWriter:
    def __init__(self):
        self.calls = []

    def write(self, stage_name, candidates, *, run_id):
        self.calls.append((stage_name, candidates, run_id))


def _make_ctx(*, brief=None):
    return SimpleNamespace(
        run_id="test-run", brief=brief, audit=_FakeAuditWriter(), boundary=_FakeBoundaryWriter(),
    )


def _config_dict() -> dict:
    return {
        "baseline_weights": dict(BASELINE_WEIGHTS),
        "modifier_rules": [
            {
                "reason": "infected_wound", "field": "wound_context", "any_of": ["infected"],
                "modifiers": {"antimicrobial": 1.25},
            },
        ],
        "normalizers": {},
        "input_sources": {
            "scores": {"antimicrobial": "amp_probability"},
        },
    }


def test_run_uses_builtin_policy_without_params():
    candidate = Candidate(id="c1", sequence="KLLK", predictions={"amp_probability": 0.8})
    result = Stage11().run([candidate], StageConfig(params={}), _make_ctx())
    assert result[0].predictions["ranking"]["original_component_scores"]["antimicrobial"] == 0.8


@pytest.mark.parametrize("params", [{"ranking_config": {}}, {"ranking_config_path": "unused.yaml"}, {"bogus": True}])
def test_run_rejects_policy_overrides(params):
    with pytest.raises(ValueError, match="built into the code"):
        Stage11().run([], StageConfig(params=params), _make_ctx())


def test_run_maps_predictions_via_input_sources_and_uses_brief():
    stage = Stage11()
    candidate = Candidate(
        id="c1", sequence="KLLKLLKK",
        predictions={"amp_probability": 0.9},
    )
    ctx = _make_ctx(brief=SimpleNamespace(wound_context=["infected"], desired_functions=[]))
    survivors = stage.run([candidate], StageConfig(params={}), ctx)

    assert len(survivors) == 1
    ranking = survivors[0].predictions["ranking"]
    assert ranking["status"] == "ranked"
    assert ranking["original_component_scores"]["antimicrobial"] == 0.9
    assert ranking["stage1"]["wound_context"] == ["infected"]
    reasons = {a["reason"] for a in ranking["weight_adjustments"]}
    assert "infected_wound" in reasons


def test_run_with_no_brief_uses_empty_stage1():
    stage = Stage11()
    candidate = Candidate(id="c1", sequence="KLLK", predictions={"amp_probability": 0.5})
    ctx = _make_ctx(brief=None)
    survivors = stage.run([candidate], StageConfig(params={}), ctx)
    ranking = survivors[0].predictions["ranking"]
    assert ranking["stage1"]["wound_context"] == []
    assert ranking["stage1"]["desired_functions"] == []


def test_run_candidate_without_sequence_raises():
    stage = Stage11()
    candidate = Candidate(id="c1", sequence=None, predictions={})
    with pytest.raises(ValueError, match="no sequence"):
        stage.run([candidate], StageConfig(params={}), _make_ctx())


def test_run_orders_candidates_ranked_then_insufficient():
    stage = Stage11()
    ranked_candidate = Candidate(
        id="ranked1", sequence="KLLK", predictions={"amp_probability": 0.9},
    )
    insufficient_candidate = Candidate(id="insufficient1", sequence="KLLK", predictions={})
    ctx = _make_ctx()
    survivors = stage.run(
        [insufficient_candidate, ranked_candidate],
        StageConfig(params={}),
        ctx,
    )
    assert [c.id for c in survivors] == ["ranked1", "insufficient1"]


