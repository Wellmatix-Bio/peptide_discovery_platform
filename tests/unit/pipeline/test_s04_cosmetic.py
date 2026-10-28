# Unit tests for the cosmetic Stage 4: validity check, reference/penalty maths, GA ordering and the stage wrapper.
#
# The primary-function classifier is mocked so no real model is loaded.

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from pipeline.base import StageError
from pipeline.s04_generation import cosmetic_stage
from pipeline.s04_generation.cosmetic_stage import CosmeticStage4, build_primary_predictor
from pipeline.s04_generation.cosmetic_validity import (
    validate_sequence_and_modifications as check,
    validity_filter,
)
from pipeline.s04_generation.routeA_cosmetic import (
    Individual,
    build_reference,
    compute_properties,
    mutate_once,
    nearest_distance,
    physicochemical_penalty,
    run_cosmetic_ga,
)
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

SEEDS = ["GHK", "GQPR", "KTTKS", "GPQGPQ"]


class TestValidity:
    def test_plain_sequence_in_range_is_valid(self) -> None:
        assert check("GHKA", None, 3, 12) == []

    @pytest.mark.parametrize("sequence", ["GHKX", "ghk", "GH-K", "GH K"])
    def test_non_standard_alphabet_is_rejected(self, sequence: str) -> None:
        assert "non_standard_alphabet" in check(sequence, None, 2, 12)

    @pytest.mark.parametrize("sequence", ["GH", "G" * 13])
    def test_length_outside_bounds_is_rejected(self, sequence: str) -> None:
        assert "length_out_of_range" in check(sequence, None, 3, 12)

    def test_length_bounds_are_inclusive(self) -> None:
        assert check("GHK", None, 3, 3) == []

    def test_empty_sequence_is_rejected(self) -> None:
        assert check("", None, 3, 12) == ["empty_sequence"]

    def test_n_acetyl_needs_a_free_n_terminus(self) -> None:
        assert check("GHKA", {"n_acetyl": True}, 3, 12) == []
        mods = {"n_acetyl": True, "lipid_tail": {"position": "n_terminus"}}
        assert "n_acetyl_n_terminus_not_free" in check("GHKA", mods, 3, 12)

    def test_n_acetyl_off_the_n_terminus_is_rejected(self) -> None:
        assert "n_acetyl_not_at_n_terminus" in check(
            "GHKA", {"n_acetyl": {"position": 2}}, 3, 12
        )

    def test_c_amide_only_at_the_c_terminus(self) -> None:
        assert check("GHKA", {"c_amide": True}, 3, 12) == []
        assert check("GHKA", {"c_amide": {"position": 3}}, 3, 12) == []
        assert "c_amide_not_at_c_terminus" in check(
            "GHKA", {"c_amide": {"position": 1}}, 3, 12
        )

    def test_lipid_tail_on_n_terminus_or_lys_only(self) -> None:
        assert check("GHKA", {"lipid_tail": {"position": "n_terminus"}}, 3, 12) == []
        assert check("GHKA", {"lipid_tail": {"position": 2}}, 3, 12) == []
        assert "lipid_tail_not_on_lys" in check(
            "GHKA", {"lipid_tail": {"position": 1}}, 3, 12
        )

    def test_lipid_tail_position_must_exist(self) -> None:
        assert "lipid_tail_position_invalid" in check(
            "GHKA", {"lipid_tail": {"position": 9}}, 3, 12
        )

    def test_d_amino_acid_allowed_on_any_residue(self) -> None:
        assert check("GHKA", {"d_amino_acid": [0, 1, 2, 3]}, 3, 12) == []
        assert "d_amino_acid_position_out_of_range" in check(
            "GHKA", {"d_amino_acid": [4]}, 3, 12
        )

    def test_unrelated_modification_keys_are_ignored(self) -> None:
        assert check("GHKA", {"peg_chain": True, "cyclic": True}, 3, 12) == []

    def test_filter_keeps_valid_and_counts_reasons(self) -> None:
        candidates = [
            Candidate(id="ok", sequence="GHKA"),
            Candidate(id="bad", sequence="GHKX"),
            Candidate(id="short", sequence="G"),
        ]
        kept, rejected = validity_filter(candidates, 3, 12)
        assert [c.id for c in kept] == ["ok"]
        assert rejected["non_standard_alphabet"] == 1
        assert rejected["length_out_of_range"] == 1


class TestPenalty:
    def test_reference_band_and_zero_inside(self) -> None:
        ref = build_reference(SEEDS, 5.5)
        inside = {n: (ref[n]["q10"] + ref[n]["q90"]) / 2 for n in ref}
        total, parts = physicochemical_penalty(inside, ref)
        assert total == 0 and set(parts.values()) == {0.0}

    def test_penalty_is_distance_outside_band_in_scale_units(self) -> None:
        ref = build_reference(SEEDS, 5.5)
        props = {n: (ref[n]["q10"] + ref[n]["q90"]) / 2 for n in ref}
        props["gravy"] = ref["gravy"]["q90"] + 2 * ref["gravy"]["scale"]
        props["charge"] = ref["charge"]["q10"] - ref["charge"]["scale"]
        total, parts = physicochemical_penalty(props, ref)
        assert parts["gravy"] == pytest.approx(2.0)
        assert parts["charge"] == pytest.approx(1.0)
        assert total == pytest.approx(3.0)

    def test_zero_iqr_falls_back_to_full_range_then_to_unnormalizable(self) -> None:
        ref = build_reference(["GHK"], 5.5)  # one seed: IQR and range are both zero
        assert all(r["scale"] is None for r in ref.values())
        total, parts = physicochemical_penalty(compute_properties("GHK", 5.5), ref)
        assert total == 0 and parts == {}


class TestDiversity:
    def test_nearest_normalised_edit_distance(self) -> None:
        assert nearest_distance("AAAA", ["AAAB", "BBBB"]) == pytest.approx(0.25)

    def test_no_other_sequence_gives_zero(self) -> None:
        assert nearest_distance("AAAA", ["AAAA"]) == 0.0


class TestMutation:
    def test_exactly_one_edit_of_each_kind(self) -> None:
        rng = random.Random(0)
        kinds = set()
        for _ in range(200):
            out, step = mutate_once("ACDEFG", rng)
            kinds.add(step.split("@")[0])
            assert abs(len(out) - 6) <= 1 and out != "ACDEFG"
        assert kinds == {"substitution", "insertion", "deletion"}


def activity_by_lysine(sequences: list[str]) -> list[bool]:
    """Mock classifier: active when the peptide carries a Lys."""
    return ["K" in s for s in sequences]


class TestRanking:
    def test_order_is_active_then_penalty_then_diversity(self) -> None:
        def ind(active: bool, p: float, d: float) -> Individual:
            return Individual("A", 0, active=active, p_phys=p, diversity=d)

        ranked = sorted(
            [ind(False, 0.0, 0.9), ind(True, 5.0, 0.1), ind(True, 1.0, 0.1), ind(True, 1.0, 0.8)],
            key=Individual.key,
            reverse=True,
        )
        assert [(i.active, i.p_phys, i.diversity) for i in ranked] == [
            (True, 1.0, 0.8),
            (True, 1.0, 0.1),
            (True, 5.0, 0.1),
            (False, 0.0, 0.9),
        ]


class TestGA:
    def run(self, seed_sequence="GHKAV", modifications=None, **kw):
        return run_cosmetic_ga(
            seed_sequence,
            modifications or {},
            build_reference(SEEDS, 5.5),
            activity_by_lysine,
            3,
            12,
            5.5,
            random.Random(1),
            **kw,
        )

    def test_population_is_unique_valid_and_bounded(self) -> None:
        population = self.run(n_generations=5)
        sequences = [i.sequence for i in population]
        assert len(sequences) == len(set(sequences)) <= 7
        assert all(3 <= len(s) <= 12 and set(s) <= set("ACDEFGHIKLMNPQRSTVWY") for s in sequences)

    def test_is_deterministic_for_a_seeded_rng(self) -> None:
        a = [i.sequence for i in self.run(n_generations=5)]
        b = [i.sequence for i in self.run(n_generations=5)]
        assert a == b

    def test_best_individual_never_gets_worse(self) -> None:
        early = max(self.run(n_generations=1), key=Individual.key).key()[:2]
        late = max(self.run(n_generations=15), key=Individual.key).key()[:2]
        assert late >= early

    def test_history_records_every_edit(self) -> None:
        population = self.run(n_generations=4)
        assert all(i.history[0] == "seed" for i in population)
        assert any(len(i.history) > 1 for i in population)

    def test_lipid_on_lys_is_kept_valid_through_indels(self) -> None:
        mods = {"lipid_tail": {"position": 2}}  # the Lys of GHKAV
        population = self.run(modifications=mods, n_generations=8)
        assert all(i.sequence[2] == "K" for i in population)


class FakeCtx:
    def __init__(self, brief=None) -> None:
        self.brief = brief
        self.seed = 7
        self.feature_extractor = None


def cosmetic_brief(function="antioxidant", min_length=6, max_length=12, ph=5.5):
    return SimpleNamespace(
        functions=SimpleNamespace(primary=function),
        peptide=SimpleNamespace(min_length=min_length, max_length=max_length),
        product=SimpleNamespace(ph=ph),
    )


class TestStage:
    def stage_run(self, monkeypatch, candidates, params=None, brief=None):
        monkeypatch.setattr(
            cosmetic_stage, "build_primary_predictor", lambda *a, **k: activity_by_lysine
        )
        config = StageConfig(enabled=True, params={"n_generations": 4, **(params or {})})
        return CosmeticStage4().run(candidates, config, FakeCtx(brief or cosmetic_brief()))

    def seeds(self):
        return [Candidate(id=f"s{i}", sequence=s) for i, s in enumerate(SEEDS)]

    def test_returns_valid_active_unique_non_seed_candidates(self, monkeypatch) -> None:
        out = self.stage_run(monkeypatch, self.seeds())
        assert out
        sequences = [c.sequence for c in out]
        assert len(sequences) == len(set(sequences))
        assert not set(sequences) & set(SEEDS)
        for c in out:
            assert c.predictions["primary_active"] == 1 and "K" in c.sequence
            assert 3 <= len(c.sequence) <= 12
            assert c.predictions["parent_seed_id"].startswith("s")
            assert {"mutation_history", "p_phys", "diversity", "rank"} <= set(c.predictions)

    def test_at_most_top_k_per_seed(self, monkeypatch) -> None:
        out = self.stage_run(monkeypatch, self.seeds()[:1], {"top_k": 2})
        assert len(out) <= 2

    def test_invalid_seeds_are_dropped_before_generation(self, monkeypatch) -> None:
        out = self.stage_run(monkeypatch, [Candidate(id="x", sequence="GHKX")])
        assert out == []

    def test_params_override_the_brief(self, monkeypatch) -> None:
        out = self.stage_run(
            monkeypatch, self.seeds(), {"min_length": 4, "max_length": 6}
        )
        assert all(4 <= len(c.sequence) <= 6 for c in out)

    def test_missing_brief_and_params_is_an_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            cosmetic_stage, "build_primary_predictor", lambda *a, **k: activity_by_lysine
        )
        with pytest.raises(StageError):
            CosmeticStage4().run(
                self.seeds(), StageConfig(enabled=True, params={}), FakeCtx(None)
            )


class TestPrimaryPredictor:
    def test_active_is_a_score_at_or_above_the_threshold(self, monkeypatch) -> None:
        monkeypatch.setattr(cosmetic_stage, "score_function", lambda f, seqs, fe: [0.49, 0.5, 0.9])
        decide = build_primary_predictor("collagen_ecm", 0.5, FakeCtx())
        assert decide(["A", "B", "C"]) == [False, True, True]

    def test_threshold_comes_from_the_stage_config_with_a_default_of_half(self, monkeypatch) -> None:
        from schemas.stage_configs import Stage4CosmeticParams

        assert Stage4CosmeticParams().primary_threshold == 0.5
        assert Stage4CosmeticParams(primary_threshold=0.8).primary_threshold == 0.8

    def test_unknown_function_is_rejected(self) -> None:
        with pytest.raises(StageError):
            build_primary_predictor("hydration_barrier", 0.5, FakeCtx())
