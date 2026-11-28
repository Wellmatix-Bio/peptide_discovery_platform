"""Module D: theoretical yield, enzymatic release routes and molecular properties (src/food/d_production)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from food.a_proteins.fasta import Substrate  # noqa: E402
from food.b_digestion.masses import avg_mass  # noqa: E402
from food.d_production.properties import PROTON_MASS, molecular_properties  # noqa: E402
from food.d_production.routes import INTACT, NEEDS_MISSED_CLEAVAGES, NOT_RELEASED, release_routes  # noqa: E402
from food.d_production.yields import theoretical_yield  # noqa: E402


def protein(seq: str) -> Substrate:
    return Substrate(substrate_id="P1", accession="P1", precursor_start=1, precursor_end=len(seq), sequence=seq)


def by_pathway(routes):
    return {r.pathway: r for r in routes}


def test_yield_is_copies_times_peptide_weight_over_protein_weight_times_mass():
    seq = "AAKGGRCCDDKEE"
    assert theoretical_yield(seq, "GGR", 10.0) == pytest.approx(10.0 * avg_mass("GGR") / avg_mass(seq))
    twice = "GGRAAGGR"
    assert theoretical_yield(twice, "GGR", 1.0) == pytest.approx(2 * avg_mass("GGR") / avg_mass(twice))


def test_yield_is_zero_when_the_protein_lacks_the_peptide_and_rejects_bad_input():
    assert theoretical_yield("AAKGGR", "WWW", 5.0) == 0.0
    with pytest.raises(ValueError):
        theoretical_yield("AAKGGR", "GGR", -1.0)
    with pytest.raises(ValueError):
        theoretical_yield("AAKGGX", "GGX", 1.0)


def test_trypsin_releases_a_peptide_between_two_cut_sites_intact():
    r = by_pathway(release_routes("GGR", protein("AAKGGRCCDDKEE")))["trypsin_only"]
    assert r.verdict == INTACT and r.releases_intact and (r.start, r.end) == (4, 6)
    assert r.n_boundary == ("intestinal:trypsin",) and r.c_boundary == ("intestinal:trypsin",)
    assert r.release_stage == "intestinal" and r.missed_cleavages_required == 0


def test_a_peptide_with_a_cut_site_inside_needs_missed_cleavages():
    r = by_pathway(release_routes("AAKGGR", protein("AAKGGRCCDDKEE")))["trypsin_only"]
    assert r.verdict == NEEDS_MISSED_CLEAVAGES and r.missed_cleavages_required == 1
    cut = r.internal_cuts_at_release[0]
    assert (cut.position_in_peptide, cut.residue, cut.cleaved_by) == (3, "K", ("intestinal:trypsin",))
    assert r.n_boundary == ("protein_terminus",)


def test_a_peptide_whose_end_is_not_cut_is_not_released():
    r = by_pathway(release_routes("AAKG", protein("AAKGGRCCDDKEE")))["trypsin_only"]
    assert r.verdict == NOT_RELEASED and r.release_stage is None and r.c_boundary == ()


def test_single_enzyme_and_sequential_pathways_are_all_evaluated():
    routes = by_pathway(release_routes("GGR", protein("AAKGGRCCDDKEE")))
    assert set(routes) == {"pepsin_only", "trypsin_only", "chymotrypsin_only", "trypsin_chymotrypsin", "gastric_to_intestinal"}


def test_every_copy_of_the_peptide_gets_its_own_route_and_a_missing_peptide_raises():
    routes = [r for r in release_routes("GGR", protein("AAKGGRAAKGGR")) if r.pathway == "trypsin_only"]
    assert [(r.start, r.end) for r in routes] == [(4, 6), (10, 12)]
    with pytest.raises(ValueError):
        release_routes("WWW", protein("AAKGGR"))


def test_molecular_properties_of_a_known_peptide():
    p = molecular_properties("GGR")
    assert p.length == 3 and p.cysteine_count == 0 and p.aromatic_count == 0
    assert p.monoisotopic_mass == pytest.approx(57.02146372057 * 2 + 156.10111102360 + 18.0105646837)
    assert p.mz_plus1 == pytest.approx(p.monoisotopic_mass + PROTON_MASS)
    assert p.mz_plus2 == pytest.approx((p.monoisotopic_mass + 2 * PROTON_MASS) / 2)
    assert p.net_charge_ph7_4 > 0 and p.gravy == pytest.approx((-0.4 - 0.4 - 4.5) / 3)


def test_aromatic_and_cysteine_content_and_bad_input():
    p = molecular_properties("FWYCC")
    assert (p.aromatic_count, p.aromatic_fraction, p.cysteine_count) == (3, 0.6, 2)
    for bad in ("", "AXA"):
        with pytest.raises(ValueError):
            molecular_properties(bad)
