# Unit tests for s11_ranking brief-driven (dynamic) objective weights.
from __future__ import annotations

import math

import pytest

from pipeline.s11_ranking.stage import (
    BUILTIN_RANKING_POLICY,
    COMPONENTS,
    CandidateScorer,
    ComponentScores,
    ModifierRule,
    ProductObjective,
    RankingConfig,
    RankingInput,
    WeightAdjuster,
    WeightingConfig,
)

BASELINE = {
    "wound_closure": 0.22, "antimicrobial": 0.18, "immunomodulation": 0.13,
    "angiogenesis": 0.12, "collagen_ecm": 0.10, "safety": 0.12, "stability": 0.08,
    "synthesis_feasibility": 0.03, "mechanistic_confidence": 0.02,
}
SCORES = {c: 0.7 for c in COMPONENTS}

RULES = [
    ModifierRule(reason="antimicrobial_objective", field="desired_functions",
                 any_of=["antimicrobial"], modifiers={"antimicrobial": 1.25}),
    ModifierRule(reason="angiogenesis_objective", field="desired_functions",
                 any_of=["angiogenesis"], modifiers={"angiogenesis": 1.25}),
    ModifierRule(reason="collagen_objective", field="desired_functions",
                 any_of=["collagen_synthesis"], modifiers={"collagen_ecm": 1.25}),
    ModifierRule(reason="infected_wound", field="wound_context", any_of=["infected"],
                 modifiers={"antimicrobial": 1.25, "immunomodulation": 1.15}),
    ModifierRule(reason="ischemic", field="wound_context", any_of=["ischemic"],
                 modifiers={"angiogenesis": 1.35}),
    ModifierRule(reason="low_perfusion", field="wound_context", any_of=["low_perfusion"],
                 modifiers={"angiogenesis": 1.35}),
    ModifierRule(reason="clean_surgical", field="wound_context",
                 all_of=["clean", "surgical"], modifiers={"antimicrobial": 0.75}),
    ModifierRule(reason="chronic", field="wound_context", any_of=["chronic"],
                 modifiers={"stability": 1.2, "safety": 1.15}),
]


def make_config(**overrides) -> RankingConfig:
    kwargs = {"baseline_weights": dict(BASELINE), "modifier_rules": RULES}
    kwargs.update(overrides)
    return RankingConfig(**kwargs)


def adjust(functions=(), context=(), config=None):
    return WeightAdjuster(config or make_config()).adjust(
        ProductObjective(desired_functions=list(functions), wound_context=list(context))
    )


def test_empty_brief_uses_baseline_weights():
    weights, audit = adjust()
    for c in COMPONENTS:
        assert weights[c] == pytest.approx(BASELINE[c])
        assert audit[c].baseline_weight == BASELINE[c]
    assert audit["angiogenesis"].reason == "empty brief"
    assert audit["safety"].reason == "always on"


def test_unselected_angiogenesis_drops_near_floor_and_below_baseline():
    weights, audit = adjust(functions=["antimicrobial"])
    assert audit["angiogenesis"].relevance == 0.05
    assert audit["angiogenesis"].reason == "floor"
    assert weights["angiogenesis"] < BASELINE["angiogenesis"]
    assert weights["angiogenesis"] < 0.05


def test_selected_weighs_more_than_implied():
    selected, _ = adjust(functions=["angiogenesis"])
    implied, audit = adjust(context=["ischemic"])
    assert audit["angiogenesis"].relevance == 0.5
    assert selected["angiogenesis"] > implied["angiogenesis"]


def test_infected_without_antimicrobial_selected_is_half_relevant():
    _, audit = adjust(context=["infected"])
    assert audit["antimicrobial"].relevance == 0.5
    assert audit["antimicrobial"].reason == "implied by infected"
    assert audit["antimicrobial"].requested is False


def test_two_contexts_implying_same_component_do_not_stack():
    _, audit = adjust(context=["ischemic", "low_perfusion"])
    assert audit["angiogenesis"].relevance == 0.5
    assert audit["angiogenesis"].reason == "implied by ischemic, low_perfusion"


def test_decreasing_modifier_does_not_imply_component():
    _, audit = adjust(context=["clean", "surgical"])
    assert audit["antimicrobial"].reason == "floor"


def test_wound_closure_has_minimum_relevance_and_quality_is_untouched():
    weights, audit = adjust(functions=["antimicrobial"])
    assert audit["wound_closure"].relevance == 0.5
    assert audit["wound_closure"].reason == "wound_closure minimum"
    # "chronic" carries safety/stability modifiers, which no longer act on weights.
    _, chronic = adjust(functions=["antimicrobial"], context=["chronic"])
    for c in ("safety", "stability", "synthesis_feasibility", "mechanistic_confidence"):
        assert chronic[c].relevance == 1.0
        assert chronic[c].baseline_weight == BASELINE[c]


@pytest.mark.parametrize(
    "functions,context",
    [([], []), (["antimicrobial"], []), ([], ["infected"]), (["angiogenesis"], ["infected", "ischemic"])],
)
def test_weights_always_sum_to_one(functions, context):
    weights, audit = adjust(functions, context)
    assert math.fsum(weights.values()) == pytest.approx(1.0)
    assert math.fsum(a.final_weight for a in audit.values()) == pytest.approx(1.0)


def scored(stage1, scores=None):
    return CandidateScorer(make_config()).score(
        RankingInput(
            candidate_id="c", sequence="KLLK", stage1=stage1,
            scores=ComponentScores(**(scores or SCORES)),
        )
    )


def test_unmapped_collagen_not_requested_barely_affects_coverage():
    scores = {**SCORES, "collagen_ecm": None}
    result = scored(ProductObjective(desired_functions=["antimicrobial"]), scores)
    assert result.missing_components == ["collagen_ecm"]
    assert result.requested_components_without_data == []
    assert result.evidence_coverage > 0.99


def test_unmapped_collagen_requested_is_excluded_and_reported():
    scores = {**SCORES, "collagen_ecm": None}
    result = scored(ProductObjective(desired_functions=["collagen_synthesis"]), scores)
    assert result.component_contributions["collagen_ecm"].contribution is None
    assert result.requested_components_without_data == ["collagen_ecm"]
    assert result.evidence_coverage < 0.9


def test_raw_scores_of_unrequested_components_are_still_reported():
    result = scored(ProductObjective(desired_functions=["antimicrobial"]))
    assert result.original_component_scores.angiogenesis == 0.7
    assert result.normalized_scores.angiogenesis == 0.7


def test_builtin_policy_maps_brief_to_relevance():
    adjuster = WeightAdjuster(BUILTIN_RANKING_POLICY)
    _, audit = adjuster.adjust(
        ProductObjective(
            wound_context=["chronic"], desired_functions=["anti_inflammatory", "immunomodulation"]
        )
    )
    assert audit["immunomodulation"].reason == "selected"
    assert audit["angiogenesis"].reason == "implied by chronic"
    assert audit["antimicrobial"].reason == "floor"
    assert audit["collagen_ecm"].reason == "floor"


def test_weighting_config_rejects_floor_above_implied():
    with pytest.raises(ValueError, match="floor"):
        WeightingConfig(floor=0.6, context_implied_relevance=0.5)
