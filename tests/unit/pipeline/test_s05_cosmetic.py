# Unit tests for the cosmetic Stage 5: modification-aware properties, rule rejects and flags, and the stage order.
#
# The aggregation and solubility models and the ESM cache are mocked so no real model is loaded.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.base import StageError
from pipeline.s05_physchem_screening import cosmetic_stage
from pipeline.s05_physchem_screening.cosmetic_physchem import (
    Thresholds,
    compute_properties,
    isoelectric_point,
    molecular_weight,
    net_charge,
    rule_flags,
    rule_rejects,
    seed_reference,
)
from pipeline.s05_physchem_screening.cosmetic_stage import CosmeticStage5
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage5CosmeticParams



def thresholds(**overrides) -> Thresholds:
    return Thresholds.from_params(Stage5CosmeticParams(**overrides))


T = thresholds()
PH = 5.5
SEEDS = [{"sequence": s, "modifications": {}} for s in ["GHKA", "GQPR", "KTTKS", "GPQGPQ", "AGHK"]]


class TestProperties:
    def test_blocking_the_termini_removes_their_charge(self) -> None:
        free = net_charge("AAAA", PH)
        assert net_charge("AAAA", PH, {"n_acetyl": True}) == pytest.approx(free - 1, abs=0.05)
        assert net_charge("AAAA", PH, {"c_amide": True}) == pytest.approx(free + 1, abs=0.05)
        both = net_charge("AAAA", PH, {"n_acetyl": True, "c_amide": True})
        assert both == pytest.approx(0.0, abs=1e-6)

    def test_n_terminal_lipid_blocks_the_amine_and_lys_lipid_removes_its_charge(self) -> None:
        free = net_charge("GKA", PH)
        assert net_charge("GKA", PH, {"lipid_tail": {"position": "n_terminus"}}) < free
        assert net_charge("GKA", PH, {"lipid_tail": {"position": 1}}) == pytest.approx(free - 1, abs=0.05)

    def test_pi_shifts_with_the_modifications(self) -> None:
        plain = isoelectric_point("AAKAA")
        acetylated = isoelectric_point("AAKAA", {"n_acetyl": True})
        assert acetylated < plain  # losing the N-terminal + charge lowers the pI
        assert isoelectric_point("AAKAA", {"c_amide": True}) > plain  # losing the C-terminal - charge raises it

    def test_pi_is_none_without_any_ionisable_group(self) -> None:
        assert isoelectric_point("AAAA", {"n_acetyl": True, "c_amide": True}) is None

    def test_molecular_weight_reflects_acetyl_amide_and_stated_lipid_mass(self) -> None:
        base = molecular_weight("GHKA")
        assert molecular_weight("GHKA", {"n_acetyl": True}) == pytest.approx(base + 42.037)
        assert molecular_weight("GHKA", {"c_amide": True}) == pytest.approx(base - 0.985)
        assert molecular_weight("GHKA", {"lipid_tail": {"position": 2, "mass": 238.4}}) == pytest.approx(base + 238.4)

    def test_property_set_is_complete(self) -> None:
        assert set(compute_properties("GHKA", PH)) == {
            "molecular_weight", "net_charge", "isoelectric_point", "hydrophobicity", "hydrophobic_moment",
        }


def props(**overrides) -> dict:
    return {
        "molecular_weight": 800.0, "net_charge": 1.0, "isoelectric_point": 8.0,
        "hydrophobicity": 0.0, "hydrophobic_moment": 0.1, **overrides,
    }


REFERENCE = {"hydrophobicity_low": -0.5, "hydrophobicity_high": 0.5, "moment_high": 0.3, "n_seeds": 5}


class TestRejects:
    def test_clean_peptide_passes(self) -> None:
        assert rule_rejects("GHKA", props(), REFERENCE, T) == []

    def test_molecular_weight_above_3000(self) -> None:
        assert "molecular_weight_too_high" in rule_rejects("GHKA", props(molecular_weight=3000.1), REFERENCE, T)
        assert rule_rejects("GHKA", props(molecular_weight=3000.0), REFERENCE, T) == []

    @pytest.mark.parametrize("charge", [-1.1, 2.1])
    def test_charge_outside_minus1_to_plus2(self, charge: float) -> None:
        assert "net_charge_out_of_range" in rule_rejects("GHKA", props(net_charge=charge), REFERENCE, T)

    @pytest.mark.parametrize("charge", [-1.0, 2.0])
    def test_charge_bounds_are_inclusive(self, charge: float) -> None:
        assert "net_charge_out_of_range" not in rule_rejects("GHKA", props(net_charge=charge), REFERENCE, T)

    @pytest.mark.parametrize("value", [-0.6, 0.6])
    def test_hydrophobicity_outside_seed_percentiles(self, value: float) -> None:
        assert "hydrophobicity_outside_seed_range" in rule_rejects(
            "GHKA", props(hydrophobicity=value), REFERENCE, T
        )

    def test_membrane_active_needs_high_moment_and_charge_together(self) -> None:
        high_moment = props(hydrophobic_moment=0.4)
        assert "membrane_active_profile" in rule_rejects("GHKA", {**high_moment, "net_charge": 2.0}, REFERENCE, T)
        assert "membrane_active_profile" not in rule_rejects("GHKA", {**high_moment, "net_charge": 1.9}, REFERENCE, T)
        assert "membrane_active_profile" not in rule_rejects("GHKA", props(net_charge=2.0), REFERENCE, T)

    @pytest.mark.parametrize("sequence,rejected", [("GAAAK", True), ("GAAK", False), ("GGGAK", True), ("AKAKA", False)])
    def test_three_identical_residues_in_a_row(self, sequence: str, rejected: bool) -> None:
        assert ("repeated_residues" in rule_rejects(sequence, props(), REFERENCE, T)) is rejected

    def test_without_a_seed_reference_the_percentile_checks_are_skipped(self) -> None:
        assert rule_rejects("GHKA", props(hydrophobicity=99.0, hydrophobic_moment=9.0, net_charge=2.0), None, T) == []

    def test_all_reasons_are_reported(self) -> None:
        reasons = rule_rejects("GAAAK", props(molecular_weight=4000.0, net_charge=5.0), REFERENCE, T)
        assert {"molecular_weight_too_high", "net_charge_out_of_range", "repeated_residues"} <= set(reasons)


class TestSeedReference:
    def test_percentile_bounds_from_the_seed_set(self) -> None:
        ref = seed_reference(SEEDS, PH, T)
        assert ref["n_seeds"] == 5 and ref["hydrophobicity_low"] <= ref["hydrophobicity_high"]

    def test_fewer_than_two_seeds_gives_no_reference(self) -> None:
        assert seed_reference(SEEDS[:1], PH, T) is None


class TestFlags:
    def flags(self, sequence: str, mods=None, mw=800.0, thresholds=T) -> dict:
        return rule_flags(sequence, props(molecular_weight=mw), mods, thresholds)

    @pytest.mark.parametrize("sequence", ["ACA", "AMA", "AWA"])
    def test_oxidation(self, sequence: str) -> None:
        assert "oxidation" in self.flags(sequence)

    def test_deamidation_asn_gly_and_optional_lower_risk(self) -> None:
        assert "deamidation" in self.flags("ANGA")
        assert "deamidation" not in self.flags("ANSA")
        assert "deamidation" in self.flags("ANSA", thresholds=thresholds(flag_lower_risk_motifs=True))
        assert "deamidation" in self.flags("ANHA", thresholds=thresholds(flag_lower_risk_motifs=True))

    def test_hydrolysis_asp_pro_and_optional_asp_gly(self) -> None:
        assert "hydrolysis" in self.flags("ADPA")
        assert "hydrolysis" not in self.flags("ADGA")
        assert "hydrolysis" in self.flags("ADGA", thresholds=thresholds(flag_lower_risk_motifs=True))

    def test_terminal_gln_unless_the_n_terminus_is_blocked(self) -> None:
        assert "terminal_gln" in self.flags("QAAK")
        assert "terminal_gln" not in self.flags("QAAK", {"n_acetyl": True})
        assert "terminal_gln" not in self.flags("AQAK")

    def test_unpaired_cys(self) -> None:
        assert "unpaired_cys" in self.flags("ACAA")  # odd
        assert "unpaired_cys" in self.flags("ACAC")  # even but no designed disulfide
        assert "unpaired_cys" not in self.flags("ACAC", {"disulfide_pairing": [[1, 3]]})
        assert "unpaired_cys" not in self.flags("AAAA")

    @pytest.mark.parametrize("mw,flagged", [(999.9, False), (1000.0, True), (1500.0, True), (1500.1, False)])
    def test_mw_high_window(self, mw: float, flagged: bool) -> None:
        assert ("mw_high" in self.flags("GHKA", mw=mw)) is flagged

    def test_clean_peptide_has_no_flags(self) -> None:
        assert self.flags("GHKA") == {}


class FakeFE:
    def __init__(self) -> None:
        self.warmed: list[str] = []

    def get_esm2_embedding_batch(self, sequences):
        self.warmed += list(sequences)


class Models:
    """Stand-ins for the aggregation and solubility models, recording what they were asked."""

    def __init__(self, aggregation=0.1, solubility=0.9) -> None:
        self.aggregation, self.solubility = aggregation, solubility
        self.agg_calls: list[str] = []
        self.sol_calls: list[tuple[str, str]] = []

    def install(self, monkeypatch) -> None:
        models = self
        agg = SimpleNamespace(
            predict_aggregation=lambda seq, fe: (models.agg_calls.append(seq), models.aggregation(seq) if callable(models.aggregation) else models.aggregation)[1]
        )
        sol = SimpleNamespace(
            predict_proba=lambda seq, solvent, fe: (models.sol_calls.append((seq, solvent)), models.solubility(seq) if callable(models.solubility) else models.solubility)[1]
        )
        monkeypatch.setattr(cosmetic_stage, "_get_aggregation_model", lambda: agg)
        monkeypatch.setattr(cosmetic_stage, "_get_solubility_model", lambda: sol)
        monkeypatch.setattr(cosmetic_stage, "PROPY_AVAILABLE", True)


def candidate(sequence: str, mods=None, seeds=SEEDS) -> Candidate:
    return Candidate(
        id=sequence,
        sequence=sequence,
        predictions={"modifications": mods or {}, "seed_set": seeds},
    )


def ctx(brief_ph=PH):
    return SimpleNamespace(
        brief=SimpleNamespace(product=SimpleNamespace(ph=brief_ph)), feature_extractor=FakeFE()
    )


def run(cands, params=None, context=None):
    return CosmeticStage5().run(cands, StageConfig(enabled=True, params=params or {}), context or ctx())


class TestStage:
    def test_clean_candidate_proceeds_with_properties_and_verdict(self, monkeypatch) -> None:
        Models().install(monkeypatch)
        (out,) = run([candidate("GHKAS")])
        p = out.predictions
        assert p["screening_verdict"]["overall"] == "proceed"
        assert {"molecular_weight", "net_charge", "isoelectric_point", "hydrophobicity", "hydrophobic_moment"} <= set(p)
        assert p["solubility"]["decision"] == 1 and p["solubility"]["solvent"] == "Ultrapure water"
        assert p["physchem_flag_count"] == len(p["physchem_flags"])

    def test_rule_reject_stops_before_the_models(self, monkeypatch) -> None:
        models = Models()
        models.install(monkeypatch)
        context = ctx()
        assert run([candidate("GAAAK")], context=context) == []  # three A in a row
        assert models.agg_calls == [] and models.sol_calls == [] and context.feature_extractor.warmed == []

    def test_models_only_see_candidates_that_passed_the_rules(self, monkeypatch) -> None:
        models = Models()
        models.install(monkeypatch)
        out = run([candidate("GAAAK"), candidate("GHKAS")])
        assert [c.sequence for c in out] == ["GHKAS"]
        assert models.sol_calls == [("GHKAS", "Ultrapure water")]

    def test_solubility_zero_rejects_a_plain_peptide(self, monkeypatch) -> None:
        Models(solubility=0.2).install(monkeypatch)
        assert run([candidate("GHKAS")]) == []

    def test_solubility_zero_on_a_lipid_tailed_peptide_is_a_flag(self, monkeypatch) -> None:
        Models(solubility=0.2).install(monkeypatch)
        (out,) = run([candidate("GHKAS", {"lipid_tail": {"position": 2}})])
        assert "solubility_unreliable" in out.predictions["physchem_flags"]

    def test_aggregation_above_the_cutoff_flags_but_never_rejects(self, monkeypatch) -> None:
        Models(aggregation=0.9).install(monkeypatch)
        (out,) = run([candidate("GHKAS")])
        assert "aggregation" in out.predictions["physchem_flags"]

    def test_short_peptide_skips_aggregation_and_says_so(self, monkeypatch) -> None:
        models = Models()
        models.install(monkeypatch)
        (out,) = run([candidate("GHKA")])
        assert models.agg_calls == []
        assert out.predictions["aggregation_tendency"]["status"] == "skipped_too_short"
        assert out.predictions["screening_verdict"]["not_screened"] == ["aggregation"]
        assert "aggregation" not in out.predictions["physchem_flags"]

    def test_flags_are_carried_and_counted_not_rejected(self, monkeypatch) -> None:
        Models().install(monkeypatch)
        (out,) = run([candidate("QCMAK")])  # terminal Gln, Cys, Met
        flags = out.predictions["physchem_flags"]
        assert {"oxidation", "terminal_gln", "unpaired_cys"} <= set(flags)
        assert out.predictions["physchem_flag_count"] == len(flags)

    def test_thresholds_are_overridable_per_run(self, monkeypatch) -> None:
        Models(solubility=0.4).install(monkeypatch)
        assert run([candidate("GHKAS")]) == []
        assert len(run([candidate("GHKAS")], {"solubility_threshold": 0.3})) == 1

    def test_seed_sequences_param_overrides_the_attached_seed_set(self, monkeypatch) -> None:
        Models().install(monkeypatch)
        (out,) = run([candidate("GHKAS", seeds=[])], {"seed_sequences": [s["sequence"] for s in SEEDS]})
        assert out.predictions["screening_verdict"]["seed_reference"]["n_seeds"] == 5

    def test_product_ph_comes_from_the_brief_or_params(self, monkeypatch) -> None:
        Models().install(monkeypatch)
        low = run([candidate("GHKAS")], context=ctx(5.5))[0].predictions["net_charge"]
        high = run([candidate("GHKAS")], context=ctx(7.0))[0].predictions["net_charge"]
        assert low > high
        override = run([candidate("GHKAS")], {"product_ph": 7.0}, ctx(5.5))[0].predictions["net_charge"]
        assert override == pytest.approx(high)

    def test_missing_ph_is_an_error(self, monkeypatch) -> None:
        Models().install(monkeypatch)
        with pytest.raises(StageError):
            run([candidate("GHKAS")], context=SimpleNamespace(brief=None, feature_extractor=FakeFE()))


class TestConfigurableThresholds:
    def test_defaults_are_the_specification(self) -> None:
        p = Stage5CosmeticParams()
        assert (p.molecular_weight_reject_max, p.net_charge_reject_min, p.net_charge_reject_max) == (3000.0, -1.0, 2.0)
        assert (p.hydrophobicity_percentile_low, p.hydrophobicity_percentile_high, p.moment_percentile) == (5.0, 95.0, 90.0)
        assert (p.membrane_active_min_charge, p.repeat_run_length) == (2.0, 3)
        assert (p.molecular_weight_flag_min, p.molecular_weight_flag_max) == (1000.0, 1500.0)

    def test_every_rule_cutoff_moves_with_the_config(self) -> None:
        assert "molecular_weight_too_high" in rule_rejects("GHKA", props(molecular_weight=2000.0), REFERENCE, thresholds(molecular_weight_reject_max=1500.0))
        assert "net_charge_out_of_range" not in rule_rejects("GHKA", props(net_charge=3.0), REFERENCE, thresholds(net_charge_reject_max=3.0))
        assert "repeated_residues" not in rule_rejects("GAAAK", props(), REFERENCE, thresholds(repeat_run_length=4))
        assert "mw_high" in rule_flags("GHKA", props(molecular_weight=2000.0), None, thresholds(molecular_weight_flag_max=2500.0))

    def test_percentile_bounds_follow_the_configured_percentiles(self) -> None:
        wide = seed_reference(SEEDS, PH, thresholds(hydrophobicity_percentile_low=0.0, hydrophobicity_percentile_high=100.0))
        narrow = seed_reference(SEEDS, PH, thresholds(hydrophobicity_percentile_low=40.0, hydrophobicity_percentile_high=60.0))
        assert wide["hydrophobicity_low"] < narrow["hydrophobicity_low"] <= narrow["hydrophobicity_high"] < wide["hydrophobicity_high"]

    def test_the_stage_reads_cutoffs_from_its_config(self, monkeypatch) -> None:
        Models(aggregation=0.5).install(monkeypatch)
        (out,) = run([candidate("GHKASS")], {"aggregation_flag_min": 0.4})
        assert "aggregation" in out.predictions["physchem_flags"]
        (out,) = run([candidate("GHKASS")], {"aggregation_flag_min": 0.9})
        assert "aggregation" not in out.predictions["physchem_flags"]
