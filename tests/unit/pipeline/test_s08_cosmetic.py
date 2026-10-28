# Unit tests for the cosmetic Stage 8: irritation and penetration rules, the safety models, skin-protease stability and the decision.
#
# Every model is mocked so nothing heavy is loaded.

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from pipeline.base import StageError
from pipeline.s08_safety_developability import cosmetic_stage
from pipeline.s08_safety_developability.cosmetic_safety import (
    irritation_cutoffs,
    irritation_score,
    irritation_tier,
    penetration,
    resolve_cell_type,
    skin_proteases,
)
from pipeline.s08_safety_developability.cosmetic_stage import CosmeticStage8
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage8CosmeticParams

SEEDS = [{"sequence": s, "modifications": {}} for s in ["GHKA", "GQPR", "KTTKS", "GPQGPQ", "AGHKS", "KKWK"]]
KLK = ["Q9Y337", "P49862", "Q9P0G3"]
LIPID = {"lipid_tail": {"position": "n_terminus"}}


class TestIrritationRules:
    def test_score_is_moment_times_positive_charge(self) -> None:
        assert irritation_score(0.5, 2.0) == 1.0
        assert irritation_score(0.5, -3.0) == 0.0

    def test_tiers_follow_the_seed_percentiles(self) -> None:
        cutoffs = {"n_seeds": 5, "medium_min": 1.0, "high_min": 2.0}
        assert irritation_tier(0.99, cutoffs, False) == "low"
        assert irritation_tier(1.0, cutoffs, False) == "medium"
        assert irritation_tier(2.0, cutoffs, False) == "medium"
        assert irritation_tier(2.01, cutoffs, False) == "high"

    def test_lipid_tail_moves_up_one_tier_and_stops_at_high(self) -> None:
        cutoffs = {"n_seeds": 5, "medium_min": 1.0, "high_min": 2.0}
        assert irritation_tier(0.5, cutoffs, True) == "medium"
        assert irritation_tier(1.5, cutoffs, True) == "high"
        assert irritation_tier(3.0, cutoffs, True) == "high"

    def test_zero_score_is_low_even_when_the_seed_percentiles_are_zero(self) -> None:
        assert irritation_tier(0.0, {"n_seeds": 5, "medium_min": 0.0, "high_min": 0.0}, False) == "low"

    def test_cutoffs_need_two_seeds_and_follow_the_configured_percentiles(self) -> None:
        assert irritation_cutoffs(SEEDS[:1], 5.5, 75, 90) is None
        a = irritation_cutoffs(SEEDS, 5.5, 75, 90)
        b = irritation_cutoffs(SEEDS, 5.5, 10, 20)
        assert a["medium_min"] <= a["high_min"] and b["high_min"] <= a["high_min"]


class TestPenetration:
    def test_outer_layer_has_no_rule(self) -> None:
        assert penetration("outer_layer", 5000.0, None, 1000.0)["passed"] is True

    @pytest.mark.parametrize("mw,mods,ok", [(1000.0, None, True), (1000.1, None, False), (2000.0, LIPID, True)])
    def test_epidermis_needs_small_mass_or_a_lipid_tail(self, mw, mods, ok) -> None:
        assert penetration("epidermis", mw, mods, 1000.0)["passed"] is ok

    @pytest.mark.parametrize("mods,ok", [(None, False), (LIPID, True)])
    def test_dermis_needs_a_lipid_tail(self, mods, ok) -> None:
        assert penetration("dermis", 300.0, mods, 1000.0)["passed"] is ok

    def test_unknown_depth_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            penetration("hypodermis", 300.0, None, 1000.0)


class TestHelpers:
    VOCAB = ["Human PBMC", "Human keratinocytes HaCat", "Human skin fibroblasts", "HEK"]

    def test_exact_label_is_used_as_is(self) -> None:
        assert resolve_cell_type("human skin fibroblasts", self.VOCAB) == {
            "requested": "human skin fibroblasts", "used": "Human skin fibroblasts", "exact": True,
        }

    def test_missing_label_maps_to_the_closest_and_is_recorded(self) -> None:
        r = resolve_cell_type("keratinocyte", self.VOCAB)
        assert r["used"] == "Human keratinocytes HaCat" and r["exact"] is False

    def test_dermis_adds_mmp1_to_the_kallikreins(self) -> None:
        assert skin_proteases("epidermis", KLK, ["P03956"]) == KLK
        assert skin_proteases("outer_layer", KLK, ["P03956"]) == KLK
        assert skin_proteases("dermis", KLK, ["P03956"]) == [*KLK, "P03956"]


class Fakes:
    """Mocked models; probabilities and half-lives are keyed by sequence and default to harmless values."""

    def __init__(self, hemolysis=None, half_life=None, cytotoxicity=None, allergen=None,
                 site_probability=None, panel=None, vocab=None, skin=None) -> None:
        self.skin = skin or {}
        self.hemolysis = hemolysis or {}
        self.half_life = half_life or {}
        self.cytotoxicity = cytotoxicity or {}
        self.allergen = allergen
        self.site_probability = site_probability or {}
        self.panel = panel if panel is not None else {*KLK, "P03956"}
        self.vocab = vocab or ["Human keratinocytes HaCat", "Human skin fibroblasts"]
        self.calls: dict[str, list] = {
            "hemolysis": [], "half_life": [], "cytotoxicity": [], "cleavage": [], "allergenicity": [], "skin": [],
        }

    def install(self, monkeypatch) -> None:
        f = self
        monkeypatch.setattr(cosmetic_stage, "_get_hemolysis_v2_model", lambda: SimpleNamespace(
            predict_hemolytic_proba_batch=lambda seqs, fe: (f.calls["hemolysis"].append(list(seqs)), [f.hemolysis.get(s, 0.1) for s in seqs])[1]))
        monkeypatch.setattr(cosmetic_stage, "_get_half_life_model", lambda: SimpleNamespace(
            predict_log10_half_life_batch=lambda seqs, fe, ctx: (f.calls["half_life"].append(list(seqs)), [f.half_life.get(s, 4.5) for s in seqs])[1]))

        def cytotox(seq, fe, cell_type):
            f.calls["cytotoxicity"].append((seq, cell_type))
            return f.cytotoxicity.get((seq, cell_type), f.cytotoxicity.get(seq, 0.1))

        monkeypatch.setattr(cosmetic_stage, "_get_cytotoxicity_model", lambda: SimpleNamespace(
            _load=lambda: None, cell_type_to_id={v: i for i, v in enumerate(f.vocab)}, predict_cytotoxicity=cytotox))

        def predict_batch(seqs, accession, matrices):
            f.calls["cleavage"].append(accession)
            out = []
            for s in seqs:
                p = f.site_probability.get(s)
                out.append({
                    "enzyme_name": accession,
                    "cleavage_site_positions": [] if p is None else [0],
                    "per_residue_probability": [p or 0.0] * len(s),
                })
            return out

        monkeypatch.setattr(cosmetic_stage, "_get_cleavage_model", lambda: SimpleNamespace(
            available_enzymes=lambda: [{"enzyme_uniprot": a} for a in f.panel], predict_batch=predict_batch))
        def skin(seqs, mods):
            f.calls["skin"].append((list(seqs), list(mods)))
            return [
                {"p_sensitizer": (p := f.skin.get(s, 0.1)), "members": {"rf": p, "lda": p, "svm": p, "mlp": p}}
                for s in seqs
            ]

        monkeypatch.setattr(cosmetic_stage, "_get_skin_sensitization_model", lambda: SimpleNamespace(
            version="test", predict_batch=skin))

        def allergen(seqs, fe):
            f.calls["allergenicity"].append(list(seqs))
            return [
                {"p_allergen": (p := (f.allergen or {}).get(s, 0.1)), "members": {"svm": p, "lda": p, "mlp": p}}
                for s in seqs
            ]

        monkeypatch.setattr(cosmetic_stage, "_get_allergenicity_model", lambda: SimpleNamespace(
            version="test", predict_batch=allergen))


def candidate(sequence="GHKASS", moment=0.0, charge=1.0, mw=700.0, mods=None):
    return Candidate(
        id=sequence,
        sequence=sequence,
        predictions={
            "hydrophobic_moment": moment, "net_charge": charge, "molecular_weight": mw,
            "modifications": mods or {}, "seed_set": SEEDS, "physchem_flag_count": 2,
            "structure": {"ca_distance_matrix": [[0.0] * len(sequence)] * len(sequence),
                          "exposure_by_position": [1.0] * len(sequence)},
        },
    )


def brief(depth="outer_layer", level="standard", ph=5.5):
    return SimpleNamespace(target_depth=depth, safety_level=level, product=SimpleNamespace(ph=ph))


def run(candidates, params=None, depth="outer_layer", level="standard"):
    ctx = SimpleNamespace(brief=brief(depth, level), feature_extractor=object())
    return CosmeticStage8().run(candidates, StageConfig(enabled=True, params=params or {}), ctx)


def ids(out):
    return [c.id for c in out]


class TestDefaults:
    def test_half_life_default_is_the_metabolic_sixty_minutes_in_log10_seconds(self) -> None:
        assert Stage8CosmeticParams().half_life_reject_min == pytest.approx(math.log10(3600))

    def test_every_model_cutoff_defaults_to_half(self) -> None:
        p = Stage8CosmeticParams()
        assert (p.cytotoxicity_reject_min, p.hemolysis_reject_min, p.allergenicity_reject_min) == (0.5, 0.5, 0.5)
        assert (p.cleavage_stability_reject_min, p.cleavage_decay_rate) == (0.3, 0.5)

    def test_stage_requires_stage_7_structure_and_stage_5_properties(self) -> None:
        assert {"structure", "hydrophobic_moment", "net_charge", "molecular_weight"} <= CosmeticStage8.requires


class TestRuleChecks:
    @pytest.fixture(autouse=True)
    def pinned_cutoffs(self, monkeypatch) -> None:
        """Seed percentiles pinned so the tiers are exact: low < 1.0, medium 1.0-2.0, high > 2.0 (charge 1 makes score = moment)."""
        monkeypatch.setattr(
            cosmetic_stage, "irritation_cutoffs",
            lambda seed_set, ph, medium, high: {"n_seeds": 5, "medium_min": 1.0, "high_min": 2.0},
        )
        Fakes().install(monkeypatch)

    @pytest.mark.parametrize("moment,level,rejected", [
        (0.5, "standard", False), (1.5, "standard", False), (2.5, "standard", True),
        (0.5, "strict", False), (1.5, "strict", True), (2.5, "strict", True),
    ])
    def test_tier_rejection_by_safety_level(self, moment, level, rejected) -> None:
        assert (run([candidate(moment=moment, charge=1.0)], level=level) == []) is rejected

    @pytest.mark.parametrize("moment,level,rejected", [
        (0.5, "standard", False),  # low -> medium, kept at standard
        (0.5, "strict", True),  # low -> medium, rejected at strict
        (1.5, "standard", True),  # medium -> high, rejected at standard
    ])
    def test_lipid_tail_moves_the_peptide_up_one_tier(self, moment, level, rejected) -> None:
        out = run([candidate(moment=moment, charge=1.0, mods=LIPID)], level=level)
        assert (out == []) is rejected

    def test_tier_is_recorded_with_its_cutoffs(self) -> None:
        (out,) = run([candidate(moment=1.5, charge=1.0)])
        irritation = out.predictions["irritation"]
        assert irritation["tier"] == "medium" and irritation["cutoffs"]["high_min"] == 2.0

    def test_no_positive_charge_means_no_irritation_score(self) -> None:
        (out,) = run([candidate(moment=5.0, charge=-2.0)], level="strict")
        assert out.predictions["irritation"]["score"] == 0.0 and out.predictions["irritation"]["tier"] == "low"

    def test_penetration_is_recorded_but_does_not_reject(self, monkeypatch) -> None:
        Fakes().install(monkeypatch)
        (plain,) = run([candidate()], depth="dermis")
        assert plain.predictions["penetration"]["passed"] is False
        (lipid,) = run([candidate(mods=LIPID)], depth="dermis")
        assert lipid.predictions["penetration"]["passed"] is True

    def test_penetration_mw_cutoff_is_configurable_in_the_recorded_result(self, monkeypatch) -> None:
        Fakes().install(monkeypatch)
        (out,) = run([candidate(mw=1200.0)], depth="epidermis")
        assert out.predictions["penetration"]["passed"] is False
        (out,) = run([candidate(mw=1200.0)], {"penetration_epidermis_mw_max": 1500.0}, depth="epidermis")
        assert out.predictions["penetration"]["passed"] is True

    def test_irritation_and_penetration_are_recorded(self, monkeypatch) -> None:
        Fakes().install(monkeypatch)
        (out,) = run([candidate()])
        assert out.predictions["irritation"]["tier"] in ("low", "medium", "high")
        assert out.predictions["penetration"]["passed"] is True


class TestModels:
    @pytest.mark.parametrize("p,rejected", [(0.5, True), (0.49, False)])
    def test_hemolysis_rejects_at_the_cutoff(self, monkeypatch, p, rejected) -> None:
        Fakes(hemolysis={"GHKASS": p}).install(monkeypatch)
        assert (run([candidate()]) == []) is rejected

    @pytest.mark.parametrize("p,rejected", [(0.5, True), (0.49, False)])
    def test_cytotoxicity_rejects_if_either_cell_type_reaches_the_cutoff(self, monkeypatch, p, rejected) -> None:
        Fakes(cytotoxicity={("GHKASS", "Human skin fibroblasts"): p}).install(monkeypatch)
        assert (run([candidate()]) == []) is rejected

    def test_cytotoxicity_runs_both_cell_types_and_records_each(self, monkeypatch) -> None:
        fakes = Fakes(cytotoxicity={("GHKASS", "Human keratinocytes HaCat"): 0.2})
        fakes.install(monkeypatch)
        (out,) = run([candidate()])
        assert {cell for _, cell in fakes.calls["cytotoxicity"]} == {"Human keratinocytes HaCat", "Human skin fibroblasts"}
        assert out.predictions["cytotoxicity"]["by_cell_type"]["Human keratinocytes HaCat"]["p_cytotoxic"] == 0.2

    def test_unavailable_cell_type_uses_the_closest_and_says_so(self, monkeypatch) -> None:
        Fakes(vocab=["Human skin fibroblasts", "Human keratinocytes HaCat"]).install(monkeypatch)
        (out,) = run([candidate()], {"cytotoxicity_cell_types": ["keratinocyte", "fibroblast"]})
        by_cell = out.predictions["cytotoxicity"]["by_cell_type"]
        assert all(v["exact"] is False for v in by_cell.values())
        assert {v["requested"] for v in by_cell.values()} == {"keratinocyte", "fibroblast"}

    LONG = "GHKAS" * 12  # 60 residues, inside the trained length range

    def test_peptides_shorter_than_the_trained_range_are_not_scored_and_never_pass_or_reject(self, monkeypatch) -> None:
        fakes = Fakes(allergen={"GHKASS": 0.99})
        fakes.install(monkeypatch)
        (out,) = run([candidate()])
        assert fakes.calls["allergenicity"] == []
        a = out.predictions["allergenicity"]
        assert a["status"] == "outside_scored_length_range" and a["p_allergen"] is None and a["trained_min_length"] == 50
        assert "allergenicity_not_screened" in out.predictions["safety_flags"]

    @pytest.mark.parametrize("p,rejected", [(0.5, True), (0.49, False)])
    def test_allergenicity_rejects_at_the_cutoff_inside_the_trained_range(self, monkeypatch, p, rejected) -> None:
        Fakes(allergen={self.LONG: p}).install(monkeypatch)
        assert (run([candidate(self.LONG)]) == []) is rejected

    def test_min_length_is_configurable_so_short_peptides_can_be_scored_anyway(self, monkeypatch) -> None:
        Fakes(allergen={"GHKASS": 0.9}).install(monkeypatch)
        assert run([candidate()], {"allergenicity_min_length": 1}) == []
        assert len(run([candidate()], {"allergenicity_min_length": 1, "allergenicity_reject_min": 0.95})) == 1

    def test_ensemble_and_member_probabilities_are_recorded(self, monkeypatch) -> None:
        Fakes(allergen={self.LONG: 0.2}).install(monkeypatch)
        (out,) = run([candidate(self.LONG)])
        a = out.predictions["allergenicity"]
        assert a["status"] == "ok" and a["p_allergen"] == 0.2 and set(a["members"]) == {"svm", "lda", "mlp"}
        assert "allergenicity_not_screened" not in out.predictions["safety_flags"]

    def test_default_min_length_is_the_shortest_training_sequence(self) -> None:
        assert Stage8CosmeticParams().allergenicity_min_length == 50

    def test_half_life_below_the_cutoff_rejects(self, monkeypatch) -> None:
        Fakes(half_life={"GHKASS": 3.0}).install(monkeypatch)
        assert run([candidate()]) == []
        assert len(run([candidate()], {"half_life_reject_min": 2.0})) == 1

    def test_peptide_outside_the_half_life_model_range_is_flagged_not_rejected(self, monkeypatch) -> None:
        fakes = Fakes()
        fakes.install(monkeypatch)
        (out,) = run([candidate("GHK")])
        assert fakes.calls["half_life"] == []
        assert out.predictions["half_life"]["status"] == "skipped_unsupported_sequence"
        assert "half_life_not_screened" in out.predictions["safety_flags"]

    def test_later_models_only_see_candidates_still_standing(self, monkeypatch) -> None:
        fakes = Fakes(hemolysis={"GHKASS": 0.9})
        fakes.install(monkeypatch)
        run([candidate("GHKASS"), candidate("GHKAAS")])
        assert fakes.calls["half_life"] == [["GHKAAS"]]
        assert {s for s, _ in fakes.calls["cytotoxicity"]} == {"GHKAAS"}


class TestCleavage:
    def test_score_is_exp_of_minus_decay_times_summed_severity(self, monkeypatch) -> None:
        Fakes(site_probability={"GHKASS": 0.4}).install(monkeypatch)
        (out,) = run([candidate()])
        cs = out.predictions["cleavage_stability"]
        assert cs["n_sites"] == 3 and cs["n_wound_enzymes_screened"] == 3
        assert cs["score"] == pytest.approx(math.exp(-0.5 * 3 * 0.4))

    def test_no_sites_scores_one(self, monkeypatch) -> None:
        Fakes().install(monkeypatch)
        (out,) = run([candidate()])
        assert out.predictions["cleavage_stability"]["score"] == 1.0

    def test_score_below_the_cutoff_rejects(self, monkeypatch) -> None:
        Fakes(site_probability={"GHKASS": 0.9}).install(monkeypatch)  # exp(-0.5*3*0.9) = 0.26
        assert run([candidate()]) == []
        assert len(run([candidate()], {"cleavage_stability_reject_min": 0.2})) == 1

    def test_exposure_weights_severity(self, monkeypatch) -> None:
        Fakes(site_probability={"GHKASS": 0.9}).install(monkeypatch)
        c = candidate()
        c.predictions["structure"]["exposure_by_position"] = [0.0] * 6
        (out,) = run([c])
        assert out.predictions["cleavage_stability"]["score"] == 1.0

    def test_epidermis_screens_the_three_kallikreins_and_dermis_adds_mmp1(self, monkeypatch) -> None:
        fakes = Fakes()
        fakes.install(monkeypatch)
        run([candidate()], depth="epidermis")
        assert sorted(set(fakes.calls["cleavage"])) == sorted(KLK)
        fakes.calls["cleavage"].clear()
        run([candidate(mods=LIPID)], depth="dermis")
        assert sorted(set(fakes.calls["cleavage"])) == sorted([*KLK, "P03956"])

    def test_enzyme_missing_from_the_panel_is_skipped_and_recorded(self, monkeypatch) -> None:
        Fakes(panel={"Q9Y337", "P49862"}).install(monkeypatch)
        (out,) = run([candidate()])
        cs = out.predictions["cleavage_stability"]
        assert cs["enzymes_missing_from_panel"] == ["Q9P0G3"] and cs["n_wound_enzymes_screened"] == 2

    def test_no_enzyme_in_the_panel_is_a_flag_never_a_pass(self, monkeypatch) -> None:
        Fakes(panel=set()).install(monkeypatch)
        (out,) = run([candidate()])
        assert out.predictions["cleavage_stability"]["score"] is None
        assert "cleavage_not_screened" in out.predictions["safety_flags"]

    def test_cleavage_runs_last_on_survivors_only(self, monkeypatch) -> None:
        fakes = Fakes(hemolysis={"GHKASS": 0.9}, site_probability={"GHKAAS": 0.1})
        fakes.install(monkeypatch)
        out = run([candidate("GHKASS"), candidate("GHKAAS")])
        assert ids(out) == ["GHKAAS"]


class TestDecisionRecord:
    def test_penalties_for_stage_11(self, monkeypatch) -> None:
        Fakes(site_probability={"GHKASS": 0.1}, half_life={"GHKASS": 4.1}).install(monkeypatch)
        (out,) = run([candidate()])
        pen = out.predictions["safety_penalties"]
        assert pen["physchem_flag_count"] == 2
        assert pen["cleavage_stability"] == pytest.approx(out.predictions["cleavage_stability"]["score"])
        assert pen["half_life_log10_seconds"] == 4.1
        assert pen["irritation_tier"] == out.predictions["irritation"]["tier"]

    def test_verdict_carries_versions_timestamp_and_no_reject_reason(self, monkeypatch) -> None:
        Fakes().install(monkeypatch)
        (out,) = run([candidate()])
        v = out.predictions["safety_verdict"]
        assert v["overall"] == "proceed" and v["reject_reason"] is None and v["timestamp"]
        assert {"cytotoxicity_predictor", "hemolysis_predictor", "half_life_predictor", "cleavage_site_predictor"} <= set(v["model_versions"])
        assert out.predictions["structure_ref"]["source"] == "s07_structure_mechanism"


class TestErrors:
    def test_missing_depth_level_and_ph_is_an_error(self, monkeypatch) -> None:
        Fakes().install(monkeypatch)
        with pytest.raises(StageError):
            CosmeticStage8().run([candidate()], StageConfig(enabled=True, params={}),
                                 SimpleNamespace(brief=None, feature_extractor=object()))


class TestSkinSensitization:
    @pytest.mark.parametrize("p,rejected", [(0.55, True), (0.549, False)])
    def test_threshold_is_fixed_at_point_five_five(self, monkeypatch, p, rejected) -> None:
        Fakes(skin={"GHKASS": p}).install(monkeypatch)
        assert (run([candidate()]) == []) is rejected

    def test_threshold_is_not_a_parameter(self, monkeypatch) -> None:
        Fakes(skin={"GHKASS": 0.6}).install(monkeypatch)
        assert run([candidate()], {"skin_sensitization_threshold": 0.9}) == []
        assert "skin_sensitization_threshold" not in Stage8CosmeticParams.model_fields

    def test_rejects_by_default(self) -> None:
        assert Stage8CosmeticParams().reject_skin_sensitizers is True

    def test_flag_only_when_rejection_is_off(self, monkeypatch) -> None:
        Fakes(skin={"GHKASS": 0.9}).install(monkeypatch)
        (out,) = run([candidate()], {"reject_skin_sensitizers": False})
        s = out.predictions["skin_sensitization"]
        assert s["sensitizer"] is True and s["rejects"] is False and s["threshold"] == 0.55
        assert "skin_sensitizer" in out.predictions["safety_flags"]

    def test_non_sensitizer_carries_no_flag_and_records_members(self, monkeypatch) -> None:
        Fakes(skin={"GHKASS": 0.2}).install(monkeypatch)
        (out,) = run([candidate()])
        assert "skin_sensitizer" not in out.predictions["safety_flags"]
        assert set(out.predictions["skin_sensitization"]["members"]) == {"rf", "lda", "svm", "mlp"}

    def test_modifications_are_passed_to_the_model(self, monkeypatch) -> None:
        fakes = Fakes()
        fakes.install(monkeypatch)
        run([candidate(mods={"c_amide": True})])
        assert fakes.calls["skin"][0][1] == [{"c_amide": True}]

    def test_runs_before_the_other_models_so_a_sensitizer_never_reaches_them(self, monkeypatch) -> None:
        fakes = Fakes(skin={"GHKASS": 0.9})
        fakes.install(monkeypatch)
        run([candidate()])
        assert fakes.calls["hemolysis"] == [] and fakes.calls["cytotoxicity"] == []
