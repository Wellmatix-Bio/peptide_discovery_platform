# Unit tests for s11_ranking: scoring, missing evidence, ordering, adapter.
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pipeline.s11_ranking.stage import (
    BUILTIN_RANKING_POLICY,
    MODULE_NAMES,
    CandidateRanker,
    CandidateScorer,
    FlagRule,
    Measurement,
    ModuleSpec,
    NormalizerConfig,
    RankingConfig,
    RankingInput,
    Stage11,
    Stage11Service,
)
from schemas.brief import Brief
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

ALWAYS_ON = ("safety", "stability", "synthesis_feasibility", "mechanistic_confidence")
OBJECTIVES = (
    "wound_closure", "antimicrobial", "anti_inflammatory", "immunomodulation", "angiogenesis", "collagen_ecm",
)
SCORES = {
    "wound_closure": 0.8, "antimicrobial": 0.9, "anti_inflammatory": 0.7, "immunomodulation": 0.5,
    "angiogenesis": 0.6,
    "safety": 0.9, "stability": 0.8, "synthesis_feasibility": 0.7, "mechanistic_confidence": 0.6,
}
ALL_FUNCTIONS = ["antimicrobial", "anti_inflammatory", "angiogenesis", "cell_proliferation/migration"]


def make_config(**overrides) -> RankingConfig:
    modules = {m: ModuleSpec(group="objective", measurement=Measurement(source=m)) for m in OBJECTIVES}
    modules.update({
        m: ModuleSpec(group="always_on", measurement=Measurement(source=m)) for m in ALWAYS_ON
    })
    modules["collagen_ecm"] = ModuleSpec(group="objective")
    kwargs = {"modules": modules}
    kwargs.update(overrides)
    return RankingConfig(**kwargs)


def make_input(candidate_id="c1", scores=None, brief=None, flag_values=None) -> RankingInput:
    return RankingInput(
        candidate_id=candidate_id,
        sequence="KLLKLLKK",
        brief=brief or Brief(min_length=5, max_length=30, desired_functions=ALL_FUNCTIONS),
        measurements=SCORES if scores is None else scores,
        flag_values=flag_values or {},
    )


def score(**kwargs):
    return CandidateScorer(make_config()).score(make_input(**kwargs))


# ------------------------------------------------------------------
# Final score = weighted sum of module scores, used as they are
# ------------------------------------------------------------------


def test_final_score_is_weighted_sum_of_module_scores():
    result = score()
    # four objectives requested (N_k = 1 each); immunomodulation and collagen are
    # not, so the six objectives average 4/6 and each always-on module gets that
    requested = ("wound_closure", "antimicrobial", "anti_inflammatory", "angiogenesis")
    average = 4 / 6
    expected = sum(1.0 * SCORES[m] for m in requested)
    expected += sum(average * SCORES[m] for m in ALWAYS_ON)
    expected /= 4 + 4 * average
    assert result.status == "ranked"
    assert result.final_score == pytest.approx(expected, abs=1e-9)


def test_scores_are_used_as_is_and_do_not_depend_on_the_batch():
    config = make_config()
    alone = Stage11Service(config).rank_batch([make_input("a")]).ranked_candidates[0]
    batch = Stage11Service(config).rank_batch([
        make_input("a"),
        make_input("b", scores={**SCORES, "antimicrobial": 0.1, "wound_closure": 0.2}),
    ])
    in_batch = next(r for r in batch.ranked_candidates if r.candidate_id == "a")
    assert in_batch.final_score == pytest.approx(alone.final_score, abs=1e-12)
    assert in_batch.modules["wound_closure"].normalized_score == SCORES["wound_closure"]


def test_contributions_sum_to_final_score():
    result = score(scores={**SCORES, "angiogenesis": None})
    total = math.fsum(m.contribution for m in result.modules.values() if m.contribution is not None)
    assert total == pytest.approx(result.final_score, abs=1e-9)
    assert math.fsum(m.weight for m in result.modules.values()) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------
# Missing evidence: excluded, never zero; coverage = scored / activated
# ------------------------------------------------------------------


def test_missing_score_is_excluded_not_zeroed():
    with_all = score()
    missing = score(scores={k: v for k, v in SCORES.items() if k != "angiogenesis"})
    assert "angiogenesis" in missing.missing_modules
    assert missing.incomplete_evidence is True
    assert missing.modules["angiogenesis"].contribution is None
    assert missing.modules["angiogenesis"].weight == 0.0
    # renormalized over what remains: not the same as scoring angiogenesis as 0
    zeroed = score(scores={**SCORES, "angiogenesis": 0.0})
    assert missing.final_score > zeroed.final_score
    assert missing.final_score != with_all.final_score


def test_evidence_coverage_is_scored_over_activated():
    # activated: 4 requested objectives + 4 always-on = 8; drop two scores
    result = score(scores={k: v for k, v in SCORES.items() if k not in ("angiogenesis", "stability")})
    assert result.evidence_coverage == pytest.approx(6 / 8)
    assert set(result.missing_modules) == {"angiogenesis", "stability"}


def test_unrequested_placeholder_does_not_count_against_coverage():
    result = score()
    assert result.modules["collagen_ecm"].activated is False
    assert result.evidence_coverage == 1.0
    assert result.missing_modules == []
    assert result.incomplete_evidence is False


def test_requested_placeholder_is_missing_and_reported():
    brief = Brief(min_length=5, max_length=30, desired_functions=["anti_inflammatory", "collagen_synthesis"])
    result = score(brief=brief)
    assert result.modules["collagen_ecm"].activated is True
    assert result.modules["collagen_ecm"].score is None
    assert "collagen_ecm" in result.missing_modules
    assert result.requested_modules_without_data == ["collagen_ecm"]
    assert result.evidence_coverage == pytest.approx(5 / 6)


def test_all_scores_missing_is_insufficient_evidence_not_zero():
    result = score(scores={})
    assert result.status == "insufficient_evidence"
    assert result.final_score is None
    assert result.evidence_coverage == 0.0
    assert result.rank is None


# ------------------------------------------------------------------
# Ordering: score, then coverage, then penalized stability, then id
# ------------------------------------------------------------------


def _ranked_ids(inputs, config=None):
    batch = Stage11Service(config or make_config()).rank_batch(inputs)
    return [r.candidate_id for r in batch.ranked_candidates]


def test_ordering_is_by_final_score_descending():
    inputs = [
        make_input("a", scores={**SCORES, "antimicrobial": 0.5}),
        make_input("b", scores={**SCORES, "antimicrobial": 0.9}),
        make_input("c", scores={**SCORES, "antimicrobial": 0.7}),
    ]
    assert _ranked_ids(inputs) == ["b", "c", "a"]


def _tied(candidate_id, **changes):
    """A scored candidate with its final score pinned, so only tie-breaks decide."""
    base = CandidateScorer(make_config()).score(make_input(candidate_id))
    return base.model_copy(update={"final_score": 0.5, **changes})


def test_tie_breaks_on_candidate_id_last():
    ranked = CandidateRanker().rank([_tied("zzz"), _tied("aaa")])
    assert [r.candidate_id for r in ranked] == ["aaa", "zzz"]


def test_tie_breaks_on_higher_coverage_before_stability_and_id():
    low_coverage = _tied("aaa", evidence_coverage=0.5)
    full_coverage = _tied("zzz", evidence_coverage=1.0)
    ranked = CandidateRanker().rank([low_coverage, full_coverage])
    assert [r.candidate_id for r in ranked] == ["zzz", "aaa"]


def test_tie_breaks_on_higher_penalized_stability_before_id():
    flagged = CandidateScorer(make_config()).score(make_input("aaa"))
    unflagged = CandidateScorer(make_config()).score(make_input("zzz"))
    # equal final score and coverage; the penalized stability must decide
    stability = unflagged.modules["stability"]
    weaker = flagged.model_copy(update={
        "final_score": 0.5,
        "modules": {**flagged.modules, "stability": stability.model_copy(update={"score": 0.3})},
    })
    stronger = unflagged.model_copy(update={"final_score": 0.5})
    ranked = CandidateRanker().rank([weaker, stronger])
    assert [r.candidate_id for r in ranked] == ["zzz", "aaa"]


def test_safety_is_no_longer_a_tie_break():
    low_safety = _tied("zzz")
    high_safety = _tied("aaa")
    high_safety = high_safety.model_copy(update={"modules": {
        **high_safety.modules,
        "safety": high_safety.modules["safety"].model_copy(update={"score": 0.1}),
    }})
    ranked = CandidateRanker().rank([low_safety, high_safety])
    assert [r.candidate_id for r in ranked] == ["aaa", "zzz"]  # id decides, not safety


def test_ranker_never_ranks_insufficient_evidence():
    scorer = CandidateScorer(make_config())
    ranked = CandidateRanker().rank([
        scorer.score(make_input("has_data")),
        scorer.score(make_input("no_data", scores={})),
    ])
    assert [r.candidate_id for r in ranked] == ["has_data"]


def test_batch_summary_counts():
    batch = Stage11Service(make_config()).rank_batch([make_input("r"), make_input("i", scores={})])
    assert (batch.total_candidates, batch.insufficient_evidence) == (2, 1)
    assert batch.ranked_candidates[0].rank == 1


def test_ranking_is_deterministic():
    inputs = [make_input(c, scores={**SCORES, "antimicrobial": v}) for c, v in (("a", 0.9), ("b", 0.5))]
    assert _ranked_ids(inputs) == _ranked_ids(inputs)


# ------------------------------------------------------------------
# Normalizer
# ------------------------------------------------------------------


def test_normalizer_inverts_when_higher_is_better_false():
    modules = make_config().modules
    modules["safety"] = ModuleSpec(
        group="always_on",
        measurement=Measurement(source="safety", normalizer=NormalizerConfig(higher_is_better=False)),
    )
    result = CandidateScorer(RankingConfig(modules=modules)).score(
        make_input(scores={**SCORES, "safety": 0.2})
    )
    assert result.modules["safety"].normalized_score == pytest.approx(0.8)


def test_linear_normalizer_maps_fixed_range():
    modules = make_config().modules
    modules["stability"] = ModuleSpec(
        group="always_on",
        measurement=Measurement(
            source="stability", normalizer=NormalizerConfig(kind="linear", lower=0.0, upper=200.0)
        ),
    )
    result = CandidateScorer(RankingConfig(modules=modules)).score(
        make_input(scores={**SCORES, "stability": 100.0})
    )
    assert result.modules["stability"].normalized_score == pytest.approx(0.5)


def test_out_of_range_score_raises():
    with pytest.raises(ValueError, match="outside configured normalization bounds"):
        score(scores={**SCORES, "antimicrobial": 1.5})


def test_linear_normalizer_without_bounds_raises():
    with pytest.raises(ValidationError, match="lower < upper"):
        NormalizerConfig(kind="linear")


# ------------------------------------------------------------------
# Config validation
# ------------------------------------------------------------------


def test_policy_needs_exactly_all_ten_modules():
    modules = make_config().modules
    del modules["safety"]
    with pytest.raises(ValidationError, match="exactly all ten modules"):
        RankingConfig(modules=modules)


def test_modules_need_both_groups():
    modules = {m: ModuleSpec(group="objective") for m in make_config().modules}
    with pytest.raises(ValidationError, match="at least one objective and one always-on"):
        RankingConfig(modules=modules)


def test_module_spec_takes_no_weight():
    with pytest.raises(ValidationError):
        ModuleSpec(group="always_on", weight=0.1)


def test_objective_modules_carry_no_flags():
    flag = FlagRule(source="x", levels={"a": 0.1})
    with pytest.raises(ValidationError, match="carries no flags"):
        ModuleSpec(group="objective", flags={"f": flag})


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValidationError):
        RankingConfig(modules=make_config().modules, bogus=True)


def test_bad_measurement_path_raises():
    with pytest.raises(ValidationError, match="dot-separated"):
        Measurement(source="a..b")


def test_builtin_policy_lists_the_ten_modules_with_expected_sources():
    policy = BUILTIN_RANKING_POLICY
    assert set(policy.modules) == set(MODULE_NAMES)
    assert policy.modules["collagen_ecm"].measurement is None
    assert policy.modules["wound_closure"].measurement.source == "proliferation_migration.migration"
    assert policy.modules["safety"].measurement.normalizer.higher_is_better is False


# ------------------------------------------------------------------
# Pipeline adapter: Stage11.run
# ------------------------------------------------------------------


class _FakeAuditWriter:
    def record_stage(self, result, *, run_id): ...
    def record_setup(self, result, *, run_id): ...
    def record_failure(self, stage, exc, *, run_id): ...


class _FakeBoundaryWriter:
    def write(self, stage_name, candidates, *, run_id): ...


def _ctx(brief=None):
    return SimpleNamespace(
        run_id="test-run", brief=brief, audit=_FakeAuditWriter(), boundary=_FakeBoundaryWriter()
    )


def _brief(**kw):
    base = {"min_length": 5, "max_length": 30, "desired_functions": ["antimicrobial"]}
    return Brief(**{**base, **kw})


def test_run_rejects_policy_overrides():
    with pytest.raises(ValueError, match="built into the code"):
        Stage11().run([], StageConfig(params={"bogus": True}), _ctx())


def test_run_maps_predictions_via_measurement_sources_and_uses_brief():
    candidate = Candidate(id="c1", sequence="KLLKLLKK", predictions={"amp_probability": 0.9})
    survivors = Stage11().run(
        [candidate], StageConfig(params={}), _ctx(_brief(wound_context=["infected"]))
    )
    ranking = survivors[0].predictions["ranking"]
    antimicrobial = ranking["modules"]["antimicrobial"]
    assert ranking["status"] == "ranked"
    assert antimicrobial["raw_measurement"] == 0.9
    assert antimicrobial["n_k"] == 1.5  # referenced (1.0) + infected (0.5)
    assert antimicrobial["measurement_source"] == "amp_probability"
    assert "brief" not in ranking


def test_run_with_no_brief_weights_objectives_equally():
    candidate = Candidate(id="c1", sequence="KLLK", predictions={"amp_probability": 0.5})
    survivors = Stage11().run([candidate], StageConfig(params={}), _ctx(None))
    modules = survivors[0].predictions["ranking"]["modules"]
    assert all(modules[m]["nominal_weight"] == pytest.approx(1 / 10) for m in modules)


def test_run_reads_pathogens_from_the_brief():
    candidate = Candidate(id="c1", sequence="KLLK", predictions={"amp_probability": 0.5})
    brief = _brief(desired_functions=["anti_inflammatory"], pathogens=["Escherichia_coli"])
    ranking = Stage11().run([candidate], StageConfig(params={}), _ctx(brief))[0].predictions["ranking"]
    assert ranking["modules"]["antimicrobial"]["n_k"] == 1.0
    assert ranking["modules"]["antimicrobial"]["requested"] is True


def test_run_candidate_without_sequence_raises():
    with pytest.raises(ValueError, match="no sequence"):
        Stage11().run(
            [Candidate(id="c1", sequence=None, predictions={})], StageConfig(params={}), _ctx()
        )


def test_run_orders_ranked_then_insufficient():
    ranked = Candidate(id="ranked1", sequence="KLLK", predictions={"amp_probability": 0.9})
    empty = Candidate(id="insufficient1", sequence="KLLK", predictions={})
    survivors = Stage11().run([empty, ranked], StageConfig(params={}), _ctx(_brief()))
    assert [c.id for c in survivors] == ["ranked1", "insufficient1"]
