# Unit tests for s11_ranking flag deductions (Stage 5 flags -> module score).
from __future__ import annotations

import pytest

from pipeline.s11_ranking.stage import (
    BUILTIN_RANKING_POLICY,
    Bound,
    CandidateRanker,
    CandidateScorer,
    FlagRule,
    Measurement,
    ModuleSpec,
    RankingConfig,
    RankingInput,
)
from schemas.brief import Brief

OBJECTIVES = ("wound_closure", "antimicrobial", "anti_inflammatory", "immunomodulation", "angiogenesis")
SCORES = {
    "wound_closure": 0.8, "antimicrobial": 0.8, "anti_inflammatory": 0.8, "immunomodulation": 0.8,
    "angiogenesis": 0.8,
    "safety": 0.8, "stability": 0.8, "synthesis_feasibility": 0.8, "mechanistic_confidence": 0.8,
}

SAFETY_FLAGS = {
    "net_charge": FlagRule(
        source="net_charge",
        bounds=[Bound(line=-5.0, limit=-10.0), Bound(line=9.0, limit=14.0)],
    ),
}
STABILITY_FLAGS = {
    "instability_index": FlagRule(source="instability_index", bounds=[Bound(line=40.0, limit=100.0)]),
    "solubility": FlagRule(source="solubility.score", bounds=[Bound(line=0.4, limit=0.0)]),
    "oxidation_risk": FlagRule(
        source="oxidation_risk.risk_category", levels={"medium": 0.2, "high": 0.4}
    ),
    "isoelectric_point": FlagRule(
        source="isoelectric_point", center=7.4, bounds=[Bound(line=0.5, limit=0.0)]
    ),
}


def make_config(**overrides) -> RankingConfig:
    modules = {m: ModuleSpec(group="objective", measurement=Measurement(source=m)) for m in OBJECTIVES}
    modules["collagen_ecm"] = ModuleSpec(group="objective")
    modules["safety"] = ModuleSpec(
        group="always_on", measurement=Measurement(source="safety"), flags=SAFETY_FLAGS
    )
    modules["stability"] = ModuleSpec(
        group="always_on", measurement=Measurement(source="stability"), flags=STABILITY_FLAGS
    )
    modules["synthesis_feasibility"] = ModuleSpec(
        group="always_on", measurement=Measurement(source="synthesis_feasibility"),
        flags={"disulfide_complexity": FlagRule(
            source="disulfide_complexity.category", levels={"flag": 0.2, "high": 0.4}
        )},
    )
    modules["mechanistic_confidence"] = ModuleSpec(
        group="always_on", measurement=Measurement(source="mechanistic_confidence"),
    )
    kwargs = {"modules": modules}
    kwargs.update(overrides)
    return RankingConfig(**kwargs)


def make_input(candidate_id="c1", scores=None, flag_values=None):
    return RankingInput(
        candidate_id=candidate_id,
        sequence="KLLKLLKK",
        brief=Brief(min_length=5, max_length=30, desired_functions=["antimicrobial", "anti_inflammatory"]),
        measurements=SCORES if scores is None else scores,
        flag_values=flag_values or {},
    )


def score(flag_values=None, scores=None, config=None):
    return CandidateScorer(config or make_config()).score(
        make_input(flag_values=flag_values, scores=scores)
    )


def test_no_flags_means_no_deduction():
    result = score()
    assert all(m.total_deduction == 0.0 and m.flags == [] for m in result.modules.values())
    assert all(m.score == m.normalized_score for m in result.modules.values() if m.score is not None)


def test_value_at_or_below_the_line_deducts_nothing():
    result = score({"instability_index": 40.0, "solubility": 0.4, "net_charge": 3.0})
    assert result.modules["stability"].score == pytest.approx(0.8)
    assert result.modules["safety"].score == pytest.approx(0.8)


def test_numeric_deduction_is_half_the_distance_to_the_limit():
    # (70 - 40) / (2 * (100 - 40)) = 0.25
    result = score({"instability_index": 70.0})
    flag = result.modules["stability"].flags[0]
    assert flag.deduction == pytest.approx(0.25)
    assert result.modules["stability"].score == pytest.approx(0.8 - 0.25)


def test_numeric_deduction_is_capped_at_point_four():
    # at the limit the formula gives 0.5; the cap holds it to 0.4
    for value in (100.0, 500.0):
        result = score({"instability_index": value})
        assert result.modules["stability"].flags[0].deduction == pytest.approx(0.4)
        assert result.modules["stability"].score == pytest.approx(0.8 - 0.4)


def test_barely_flagged_deducts_very_little():
    result = score({"instability_index": 40.6})
    assert result.modules["stability"].total_deduction == pytest.approx(0.6 / 120)


def test_lower_is_worse_direction_for_solubility():
    # (0.2 - 0.4) / (2 * (0 - 0.4)) = 0.25
    assert score({"solubility": 0.2}).modules["stability"].flags[0].deduction == pytest.approx(0.25)
    assert score({"solubility": 0.0}).modules["stability"].flags[0].deduction == pytest.approx(0.4)
    assert score({"solubility": 0.9}).modules["stability"].flags == []


def test_two_sided_flag_uses_the_nearer_boundary():
    low = score({"net_charge": -7.5}).modules["safety"].flags[0].deduction
    high = score({"net_charge": 11.5}).modules["safety"].flags[0].deduction
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)
    assert score({"net_charge": 0.0}).modules["safety"].flags == []


def test_isoelectric_point_is_measured_from_the_center():
    near = score({"isoelectric_point": 7.4}).modules["stability"].flags[0].deduction
    assert near == pytest.approx(0.4)  # distance 0 -> maximal, capped
    assert score({"isoelectric_point": 9.0}).modules["stability"].flags == []


def test_categorical_levels():
    assert score({"oxidation_risk": "medium"}).modules["stability"].flags[0].deduction == 0.2
    assert score({"oxidation_risk": "high"}).modules["stability"].flags[0].deduction == 0.4
    assert score({"oxidation_risk": "low"}).modules["stability"].flags == []
    flag = score({"disulfide_complexity": "flag"}).modules["synthesis_feasibility"].flags[0]
    assert flag.deduction == 0.2
    assert score({"disulfide_complexity": "high"}).modules["synthesis_feasibility"].flags[0].deduction == 0.4
    assert score({"disulfide_complexity": "medium"}).modules["synthesis_feasibility"].flags == []


def test_deductions_add():
    result = score({"instability_index": 70.0, "oxidation_risk": "medium"})
    assert result.modules["stability"].total_deduction == pytest.approx(0.25 + 0.2)
    assert result.modules["stability"].score == pytest.approx(0.8 - 0.45)


def test_score_never_goes_below_zero():
    result = score({"instability_index": 100.0, "oxidation_risk": "high", "solubility": 0.0},
                   scores={**SCORES, "stability": 0.3})
    assert result.modules["stability"].score == 0.0


def test_deduction_hits_the_score_not_the_weight():
    clean, flagged = score(), score({"instability_index": 70.0})
    assert flagged.modules["stability"].weight == clean.modules["stability"].weight
    assert flagged.modules["stability"].nominal_weight == clean.modules["stability"].nominal_weight
    # a fixed weight means the deduction reaches the final score in full
    expected = clean.final_score - clean.modules["stability"].weight * 0.25
    assert flagged.final_score == pytest.approx(expected)


def test_flagged_peptide_ranks_below_identical_unflagged():
    scorer = CandidateScorer(make_config())
    clean = scorer.score(make_input("clean"))
    flagged = scorer.score(make_input("flagged", flag_values={"instability_index": 80.0}))
    ranked = CandidateRanker().rank([flagged, clean])
    assert [r.candidate_id for r in ranked] == ["clean", "flagged"]


def test_flag_on_a_missing_module_is_ignored():
    scores = {**SCORES, "synthesis_feasibility": None}
    unflagged = score(scores=scores)
    flagged = score({"disulfide_complexity": "high"}, scores=scores)
    module = flagged.modules["synthesis_feasibility"]
    assert module.score is None and module.flags == []
    assert flagged.final_score == pytest.approx(unflagged.final_score)


def test_missing_flag_value_is_skipped():
    assert score({"instability_index": None}).modules["stability"].score == pytest.approx(0.8)


def test_non_numeric_value_for_numeric_flag_raises():
    with pytest.raises(ValueError, match="not numeric"):
        score({"instability_index": "high"})


def test_only_always_on_modules_carry_flags():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="carries no flags"):
        ModuleSpec(group="objective", flags=STABILITY_FLAGS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "x"},
        {"source": "x", "bounds": [Bound(line=1, limit=2)], "levels": {"a": 0.1}},
        {"source": "x", "levels": {"a": 1.5}},
        {"source": "x", "levels": {"a": 0.1}, "center": 7.4},
        {"source": "a..b", "levels": {"a": 0.1}},
    ],
)
def test_malformed_flag_rule_raises(kwargs):
    with pytest.raises(ValueError):
        FlagRule(**kwargs)


def test_equal_line_and_limit_raises():
    with pytest.raises(ValueError, match="must differ"):
        Bound(line=1.0, limit=1.0)


def test_builtin_policy_flags_sit_on_the_expected_modules():
    policy = BUILTIN_RANKING_POLICY
    flags_by_module = {m: set(spec.flags) for m, spec in policy.modules.items() if spec.flags}
    assert flags_by_module["safety"] == {"net_charge", "hydrophobic_moment", "amphipathicity"}
    assert flags_by_module["synthesis_feasibility"] == {"disulfide_complexity"}
    assert flags_by_module["mechanistic_confidence"] == {
        "secondary_structure_mechanism", "secondary_structure_confidence",
    }
    assert {"oxidation_risk", "deamidation_risk", "instability_index", "aggregation_tendency",
            "solubility", "isoelectric_point"} == flags_by_module["stability"]
    assert set(flags_by_module) == {"safety", "stability", "synthesis_feasibility", "mechanistic_confidence"}
    mismatch = policy.modules["mechanistic_confidence"].flags["secondary_structure_mechanism"]
    assert mismatch.levels == {"False": 0.2}
