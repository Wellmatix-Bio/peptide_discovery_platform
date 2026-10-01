from __future__ import annotations

import pytest

from pipeline.s11_ranking.stage import (
    COMPONENTS,
    DEFAULT_FLAG_PENALTIES,
    CandidateRanker,
    CandidateScorer,
    ComponentScores,
    FlagPenaltyConfig,
    InputSources,
    RankingConfig,
    RankingInput,
)

BASELINE_WEIGHTS = {
    "wound_closure": 0.22, "antimicrobial": 0.18, "immunomodulation": 0.13,
    "angiogenesis": 0.12, "collagen_ecm": 0.10, "safety": 0.12,
    "stability": 0.08, "synthesis_feasibility": 0.03, "mechanistic_confidence": 0.02,
}
SCORES = {c: 0.8 for c in COMPONENTS}

FLAGS = {
    "max_total_cut": 0.5,
    "flags": {
        "instability_index": {
            "component": "stability", "source": "instability_index",
            "bounds": [{"line": 40.0, "limit": 100.0}],
        },
        "solubility": {
            "component": "stability", "source": "solubility.score",
            "bounds": [{"line": 0.4, "limit": 0.0}],
        },
        "oxidation_risk": {
            "component": "stability", "source": "oxidation_risk.risk_category",
            "levels": {"high": 1.0},
        },
        "net_charge": {
            "component": "safety", "source": "net_charge",
            "bounds": [{"line": -5.0, "limit": -10.0}, {"line": 9.0, "limit": 14.0}],
        },
        "disulfide_complexity": {
            "component": "synthesis_feasibility", "source": "disulfide_complexity.category",
            "levels": {"flag": 0.5, "high": 1.0},
        },
    },
}


def make_config(flags=FLAGS, **overrides) -> RankingConfig:
    kwargs = {
        "baseline_weights": dict(BASELINE_WEIGHTS),
        "modifier_rules": [],
        "normalizers": {},
        "input_sources": InputSources(),
        "flag_penalties": FlagPenaltyConfig.model_validate(flags),
    }
    kwargs.update(overrides)
    return RankingConfig(**kwargs)


def make_input(candidate_id="c1", scores=None, flag_values=None, brief_limits=None):
    return RankingInput(
        candidate_id=candidate_id,
        sequence="KLLKLLKK",
        scores=ComponentScores(**(scores if scores is not None else SCORES)),
        flag_values=flag_values or {},
        brief_limits=brief_limits or {},
    )


def score(flag_values=None, scores=None, config=None, **kwargs):
    return CandidateScorer(config or make_config()).score(
        make_input(flag_values=flag_values, scores=scores, **kwargs)
    )


def test_no_flags_matches_pipeline_without_penalty_step():
    flagged_config = score()
    no_penalty_config = score(config=make_config(flags={}))
    assert flagged_config.final_score == pytest.approx(no_penalty_config.final_score, abs=1e-12)
    assert flagged_config.normalized_scores == no_penalty_config.normalized_scores
    assert all(a.flags == [] for a in flagged_config.flag_penalties.values())


def test_flag_below_line_is_not_penalized():
    result = score({"instability_index": 40.0, "solubility": 0.4, "net_charge": 3.0})
    assert result.normalized_scores.stability == pytest.approx(0.8)
    assert result.normalized_scores.safety == pytest.approx(0.8)


def test_barely_flagged_gives_tiny_change():
    result = score({"instability_index": 40.6})
    # severity 0.01, cut 0.003
    assert result.normalized_scores.stability == pytest.approx(0.8 * (1 - 0.003))


def test_at_limit_cut_equals_p():
    result = score({"instability_index": 100.0})
    audit = result.flag_penalties["stability"]
    assert audit.flags[0].severity == pytest.approx(1.0)
    assert audit.flags[0].cut == pytest.approx(0.3)
    assert result.normalized_scores.stability == pytest.approx(0.8 * 0.7)


def test_beyond_limit_is_clamped():
    result = score({"instability_index": 500.0})
    assert result.normalized_scores.stability == pytest.approx(0.8 * 0.7)


def test_lower_is_worse_direction_for_solubility():
    mid = score({"solubility": 0.2})
    assert mid.flag_penalties["stability"].flags[0].severity == pytest.approx(0.5)
    assert score({"solubility": 0.0}).normalized_scores.stability == pytest.approx(0.8 * 0.7)
    assert score({"solubility": 0.9}).normalized_scores.stability == pytest.approx(0.8)


def test_two_sided_flag_uses_nearer_boundary():
    low = score({"net_charge": -7.5}).flag_penalties["safety"].flags[0]
    high = score({"net_charge": 11.5}).flag_penalties["safety"].flags[0]
    assert low.severity == pytest.approx(0.5)
    assert high.severity == pytest.approx(0.5)
    assert score({"net_charge": 0.0}).flag_penalties["safety"].flags == []


def test_categorical_levels():
    flag = score({"disulfide_complexity": "flag"}).flag_penalties["synthesis_feasibility"].flags[0]
    high = score({"disulfide_complexity": "high"}).flag_penalties["synthesis_feasibility"].flags[0]
    assert flag.severity == 0.5 and high.severity == 1.0
    assert score({"disulfide_complexity": "low"}).flag_penalties["synthesis_feasibility"].flags == []


def test_several_flags_multiply():
    result = score({"instability_index": 70.0, "oxidation_risk": "high"})
    assert result.normalized_scores.stability == pytest.approx(0.8 * 0.85 * 0.7)


def test_two_maxed_flags_hit_the_cap_not_the_product():
    result = score({"instability_index": 100.0, "oxidation_risk": "high"})
    # 0.7 * 0.7 = 0.49 falls below the 0.5 floor
    assert result.normalized_scores.stability == pytest.approx(0.8 * 0.5)


def test_several_flags_never_below_cap_or_negative():
    flags = {
        "max_total_cut": 0.5,
        "flags": {
            f"f{i}": {
                "component": "stability", "source": "x", "p": 1.0,
                "bounds": [{"line": 0.0, "limit": 1.0}],
            }
            for i in range(5)
        },
    }
    result = score({f"f{i}": 1.0 for i in range(5)}, config=make_config(flags=flags))
    assert result.normalized_scores.stability == pytest.approx(0.8 * 0.5)
    assert result.normalized_scores.stability >= 0


def test_flag_on_missing_component_is_ignored():
    scores = {**SCORES, "synthesis_feasibility": None}
    unflagged = score(scores=scores)
    flagged = score({"disulfide_complexity": "high"}, scores=scores)
    audit = flagged.flag_penalties["synthesis_feasibility"]
    assert audit.penalized_score is None
    assert audit.flags == []
    assert flagged.final_score == pytest.approx(unflagged.final_score)
    assert flagged.missing_components == unflagged.missing_components


def test_missing_flag_value_is_skipped():
    result = score({"instability_index": None})
    assert result.normalized_scores.stability == pytest.approx(0.8)


def test_flagged_peptide_ranks_below_identical_unflagged():
    scorer = CandidateScorer(make_config())
    clean = scorer.score(make_input("clean"))
    flagged = scorer.score(make_input("flagged", flag_values={"instability_index": 80.0}))
    ranked = CandidateRanker().rank([flagged, clean])
    assert [r.candidate_id for r in ranked] == ["clean", "flagged"]
    assert flagged.final_score < clean.final_score


def test_tie_break_uses_penalized_values():
    # Equal final scores would tie on safety first; the penalized safety must decide.
    scorer = CandidateScorer(make_config())
    a = scorer.score(make_input("a", flag_values={"net_charge": 14.0}))
    b = scorer.score(make_input("b"))
    assert a.normalized_scores.safety < b.normalized_scores.safety


def test_audit_reports_raw_penalized_and_flag_details():
    result = score({"instability_index": 70.0})
    audit = result.flag_penalties["stability"]
    assert audit.score_before_penalty == pytest.approx(0.8)
    assert audit.penalized_score == pytest.approx(result.normalized_scores.stability)
    entry = audit.flags[0]
    assert (entry.flag, entry.raw_value) == ("instability_index", 70.0)
    assert entry.severity == pytest.approx(0.5)
    assert entry.cut == pytest.approx(0.15)


def test_contributions_sum_to_final_score_with_flags():
    result = score({"instability_index": 80.0, "net_charge": 12.0})
    total = sum(c.contribution or 0.0 for c in result.component_contributions.values())
    assert total == pytest.approx(result.final_score)


def test_brief_line_shifts_bounds_keeping_span():
    flags = {
        "flags": {
            "length": {
                "component": "synthesis_feasibility", "source": "length",
                "brief_line_field": "max_length",
                "bounds": [{"line": 40.0, "limit": 50.0}],
            }
        }
    }
    config = make_config(flags=flags)
    at_default = score({"length": 45}, config=config)
    shifted = score({"length": 45}, config=config, brief_limits={"max_length": 30.0})
    assert at_default.flag_penalties["synthesis_feasibility"].flags[0].severity == pytest.approx(0.5)
    assert shifted.flag_penalties["synthesis_feasibility"].flags[0].severity == pytest.approx(1.0)


def test_non_numeric_value_for_numeric_flag_raises():
    with pytest.raises(ValueError, match="not numeric"):
        score({"instability_index": "high"})


@pytest.mark.parametrize(
    "rule",
    [
        {"component": "stability", "source": "x"},
        {"component": "stability", "source": "x", "bounds": [{"line": 1, "limit": 1}]},
        {"component": "stability", "source": "x", "levels": {"a": 1.0}, "bounds": [{"line": 0, "limit": 1}]},
        {"component": "stability", "source": "x", "levels": {"a": 1.5}},
    ],
)
def test_malformed_flag_rule_raises(rule):
    with pytest.raises(ValueError):
        FlagPenaltyConfig.model_validate({"flags": {"f": rule}})


def test_default_policy_config_is_valid_and_maps_expected_components():
    config = FlagPenaltyConfig.model_validate(DEFAULT_FLAG_PENALTIES)
    by_component: dict[str, set[str]] = {}
    for name, rule in config.flags.items():
        by_component.setdefault(rule.component, set()).add(name)
    assert by_component["safety"] == {"net_charge", "hydrophobic_moment", "amphipathicity"}
    assert by_component["synthesis_feasibility"] == {"disulfide_complexity"}
    assert {"instability_index", "solubility", "isoelectric_point"} <= by_component["stability"]
