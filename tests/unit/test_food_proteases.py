"""The protease registry, the side-by-side rule comparison and the digestion options (src/food/b_digestion)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from food.b_digestion.proteases import compare_proteases, list_proteases  # noqa: E402
from food.b_digestion.settings import DEFAULT_SETTINGS, with_options  # noqa: E402


def test_the_registry_lists_three_proteases_and_warns_where_the_rule_digestion_uses_is_not_validated():
    rows = {r["protease"]: r for r in list_proteases()}
    assert list(rows) == ["pepsin", "trypsin", "chymotrypsin"]
    assert rows["trypsin"]["warning"] == ""
    assert rows["pepsin"]["default_rule"] == "pepsin_ph1.3" and "not validated" in rows["pepsin"]["warning"]
    assert rows["chymotrypsin"]["default_rule"] == "chymotrypsin_high" and "not validated" in rows["chymotrypsin"]["warning"]
    assert [r["name"] for r in rows["chymotrypsin"]["rules"] if r["validated"]] == ["chymotrypsin_low"]


def test_the_comparison_puts_every_rule_side_by_side_with_its_cut_positions():
    rows = {r["rule"]: r for r in compare_proteases("AAKAARAA")}
    assert set(rows) == {"pepsin_ph1.3", "pepsin_ph2.0", "trypsin", "chymotrypsin_high", "chymotrypsin_low"}
    assert rows["trypsin"]["cut_positions"] == [3, 6] and rows["trypsin"]["n_cuts"] == 2 and rows["trypsin"]["validated"] is True
    assert rows["trypsin"]["used_by_digestion"] and not rows["chymotrypsin_low"]["used_by_digestion"]
    assert all(r["settings_version"] == DEFAULT_SETTINGS.name for r in rows.values())


def test_a_short_sequence_gets_the_peptide_size_warning_and_an_unvalidated_rule_its_verdict():
    rows = {r["rule"]: r for r in compare_proteases("AAKAARAA", ["pepsin"])}
    assert set(rows) == {"pepsin_ph1.3", "pepsin_ph2.0"}
    assert any("untested on peptides" in w for w in rows["pepsin_ph1.3"]["warnings"]) and any(w.startswith("not validated") for w in rows["pepsin_ph1.3"]["warnings"])


@pytest.mark.parametrize("sequence, proteases", [("", None), ("AAXA", None), ("AAKA", ["plasmin"])])
def test_the_comparison_refuses_what_it_cannot_cut(sequence, proteases):
    with pytest.raises(ValueError):
        compare_proteases(sequence, proteases)


def test_options_limit_the_run_to_the_chosen_proteases_and_limits():
    s = with_options(DEFAULT_SETTINGS, proteases=["trypsin", "pepsin"], max_missed_cleavages=0, min_length=4)
    assert s.max_missed_cleavages == 0 and s.min_length == 4 and s.max_length == DEFAULT_SETTINGS.max_length
    assert [p.id for p in s.pathways] == ["pepsin_only", "trypsin_only", "pepsin_trypsin"]  # each alone, then together


def test_options_refuse_an_unknown_pathway_and_name_the_available_ones():
    with pytest.raises(ValueError, match="Unknown pathway nope; available: pepsin_only"):
        with_options(DEFAULT_SETTINGS, pathways=["nope"])
