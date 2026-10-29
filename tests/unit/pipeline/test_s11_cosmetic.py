# Unit tests for the cosmetic Stage 11: brief-scaled weights, component scores, neutral scoring of skipped checks and Pareto fronts.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.base import StageError
from pipeline.s11_ranking.cosmetic_stage import CosmeticStage11, component_weights, percentile_ranks
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage11CosmeticParams

CUTOFFS = {"n_seeds": 10, "medium_min": 2.0, "high_min": 4.0}


def ctx(depth="epidermis", level="standard", ph=5.5, primary="antioxidant", secondary=()):
    return SimpleNamespace(
        brief=SimpleNamespace(
            target_depth=depth,
            safety_level=level,
            product=SimpleNamespace(ph=ph),
            functions=SimpleNamespace(primary=primary, secondary=list(secondary)),
        )
    )


def cand(cid, primary_score=0.9, irritation=1.0, mw=400.0, stability=0.8, flags=0, secondary=None, lipid=False):
    scores = {"antioxidant": {"score": primary_score, "pass": True}}
    scores.update(secondary or {})
    return Candidate(
        id=cid,
        sequence=cid,
        predictions={
            "functional_scores": scores,
            "irritation": {"score": irritation, "tier": "low", "cutoffs": CUTOFFS, "status": "ok"},
            "modifications": {"lipid_tail": {"position": 0}} if lipid else {},
            "molecular_weight": mw,
            "cleavage_stability": {"score": stability},
            "physchem_flag_count": flags,
        },
    )


def run(candidates, params=None, context=None):
    return CosmeticStage11().run(candidates, StageConfig(enabled=True, params=params or {}), context or ctx())


def comp(candidate, name):
    return candidate.predictions["ranking"]["components"][name]


class TestWeights:
    def test_base_points_normalize_to_one(self) -> None:
        w = component_weights(Stage11CosmeticParams(), 2, "epidermis", "standard", 5.5)
        assert sum(w.values()) == pytest.approx(1.0)
        assert w["primary_function"] == pytest.approx(4 / 14)
        assert w["secondary_functions"] == pytest.approx(2 / 14)

    def test_no_secondaries_has_zero_weight(self) -> None:
        w = component_weights(Stage11CosmeticParams(), 0, "epidermis", "standard", 5.5)
        assert w["secondary_functions"] == 0

    def test_brief_scalings(self) -> None:
        p = Stage11CosmeticParams()
        outer = component_weights(p, 0, "outer_layer", "strict", 7.0)
        assert outer["delivery_fit"] == 0
        # points: 4 + 0 + 4.5 + 0 + 1 + 1.5 = 11
        assert outer["safety"] == pytest.approx(4.5 / 11)
        assert outer["stability"] == pytest.approx(1 / 11)
        assert outer["developability"] == pytest.approx(1.5 / 11)
        dermis = component_weights(p, 0, "dermis", "standard", 5.5)
        assert dermis["delivery_fit"] == pytest.approx(3 / 14)
        assert dermis["stability"] == pytest.approx(3 / 14)

    def test_ph_bounds_are_inclusive(self) -> None:
        p = Stage11CosmeticParams()
        for ph in (4.5, 6.5):
            assert component_weights(p, 0, "epidermis", "standard", ph)["developability"] == pytest.approx(1 / 12)


class TestPercentileRanks:
    def test_best_one_worst_zero(self) -> None:
        assert percentile_ranks([0.1, 0.9, 0.5]) == [0.0, 1.0, 0.5]

    def test_ties_average(self) -> None:
        assert percentile_ranks([0.5, 0.5, 0.9]) == [0.25, 0.25, 1.0]

    def test_fewer_than_two_is_half(self) -> None:
        assert percentile_ranks([0.7]) == [0.5]


class TestComponents:
    def test_perfect_candidate_scores_100(self) -> None:
        (a,) = run([cand("AAA", irritation=0.0, mw=400)])
        # a lone survivor has primary rank 0.5, so everything else perfect gives 100 - 4/10 * 50
        assert a.predictions["ranking"]["final_score"] == pytest.approx(100 * (0.4 * 0.5 + 0.6))

    def test_safety_margin_uses_high_cutoff_standard_and_medium_strict(self) -> None:
        (a,) = run([cand("AAA", irritation=2.0)])
        assert comp(a, "safety")["score"] == pytest.approx(0.5)
        (b,) = run([cand("AAA", irritation=2.0)], context=ctx(level="strict"))
        assert comp(b, "safety")["score"] == pytest.approx(0.0)

    def test_lipid_tail_scales_safety(self) -> None:
        (a,) = run([cand("AAA", irritation=0.0, lipid=True)])
        assert comp(a, "safety")["score"] == pytest.approx(0.75)

    def test_delivery_fit_by_molecular_weight(self) -> None:
        low, mid, high = run([cand("A", mw=500), cand("B", mw=1000), cand("C", mw=2000)])
        scores = {c.id: comp(c, "delivery_fit")["score"] for c in (low, mid, high)}
        assert scores == {"A": 1.0, "B": 0.75, "C": 0.5}

    def test_stability_is_the_cleavage_score(self) -> None:
        (a,) = run([cand("AAA", stability=0.37)])
        assert comp(a, "stability")["score"] == 0.37

    def test_developability_counts_physchem_and_synthesis_flags_capped(self) -> None:
        c = cand("AAA", flags=2)
        c.predictions["synthesis_feasibility"] = {"flags": {"aspartimide_risk": [1]}}
        d = cand("BBB", flags=9)
        run([c, d])
        assert comp(c, "developability")["score"] == pytest.approx(0.25)
        assert comp(d, "developability")["score"] == 0.0

    def test_secondary_is_share_passed(self) -> None:
        secondary = {"pigmentation": {"score": 0.7, "pass": True}, "collagen_ecm": {"score": 0.1, "pass": False}}
        c = cand("AAA", secondary=secondary)
        c.predictions["functional_verdict"] = {
            "primary_function": "antioxidant",
            "secondary_functions": ["pigmentation", "collagen_ecm"],
        }
        run([c])
        assert comp(c, "secondary_functions")["score"] == 0.5
        assert comp(c, "secondary_functions")["weight"] == pytest.approx(2 / 14)


class TestNotScreened:
    def test_missing_cleavage_is_neutral_and_lowers_coverage(self) -> None:
        c = cand("AAA", stability=None)
        run([c])
        assert comp(c, "stability")["score"] == 0.5
        assert c.predictions["ranking"]["not_screened"] == ["stability"]
        assert c.predictions["ranking"]["coverage"] == pytest.approx(1 - 2 / 12)

    def test_no_seed_reference_makes_safety_neutral(self) -> None:
        c = cand("AAA")
        c.predictions["irritation"] = {"score": 1.0, "tier": None, "status": "not_screened"}
        run([c])
        assert comp(c, "safety")["score"] == 0.5
        assert "safety" in c.predictions["ranking"]["not_screened"]

    def test_neutral_score_is_configurable(self) -> None:
        c = cand("AAA", stability=None)
        run([c], {"not_screened_score": 0.2})
        assert comp(c, "stability")["score"] == 0.2


class TestRanking:
    def test_ordered_by_final_score_with_ranks(self) -> None:
        good = cand("GOOD", primary_score=0.9, irritation=0.0)
        bad = cand("BAD", primary_score=0.2, irritation=3.9, mw=1800, stability=0.1, flags=4)
        out = run([bad, good])
        assert [c.id for c in out] == ["GOOD", "BAD"]
        assert [c.predictions["ranking"]["final_score_rank"] for c in out] == [1, 2]

    def test_pareto_fronts(self) -> None:
        a = cand("AAA", primary_score=0.9, stability=0.2)
        b = cand("BBB", primary_score=0.5, stability=0.9)
        c = cand("CCC", primary_score=0.1, stability=0.1)
        run([a, b, c])
        fronts = {x.id: x.predictions["ranking"]["pareto_rank"] for x in (a, b, c)}
        assert fronts == {"AAA": 1, "BBB": 1, "CCC": 2}

    def test_never_filters(self) -> None:
        assert len(run([cand("A"), cand("B"), cand("C")])) == 3

    def test_missing_brief_values_raise(self) -> None:
        with pytest.raises(StageError):
            run([cand("A")], context=SimpleNamespace(brief=SimpleNamespace()))

    def test_empty_input(self) -> None:
        assert run([]) == []


class TestSkinSensitizer:
    def test_sensitizer_safety_is_multiplied_by_the_factor(self) -> None:
        c = cand("AAA", irritation=0.0)
        c.predictions["skin_sensitization"] = {"sensitizer": True}
        run([c])
        assert comp(c, "safety")["score"] == pytest.approx(0.5)
        assert comp(c, "safety")["detail"]["skin_sensitizer_factor"] == 0.5

    def test_factor_is_configurable_and_stacks_with_the_lipid_factor(self) -> None:
        c = cand("AAA", irritation=0.0, lipid=True)
        c.predictions["skin_sensitization"] = {"sensitizer": True}
        run([c], {"skin_sensitizer_safety_factor": 0.2})
        assert comp(c, "safety")["score"] == pytest.approx(0.75 * 0.2)

    def test_non_sensitizer_is_unchanged(self) -> None:
        c = cand("AAA", irritation=0.0)
        c.predictions["skin_sensitization"] = {"sensitizer": False}
        run([c])
        assert comp(c, "safety")["score"] == 1.0

    def test_sensitizer_with_unscreened_irritation_is_penalized_but_stays_unscreened(self) -> None:
        c = cand("AAA")
        c.predictions["irritation"] = {"score": 1.0, "tier": None, "status": "not_screened"}
        c.predictions["skin_sensitization"] = {"sensitizer": True}
        run([c])
        assert comp(c, "safety")["score"] == pytest.approx(0.25) and not comp(c, "safety")["screened"]
