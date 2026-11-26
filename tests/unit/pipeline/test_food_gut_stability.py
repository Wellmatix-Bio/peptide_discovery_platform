"""s06_food_gut_stability from the protease rules: sites per protease, gut survival as the share of bonds left uncut, no model involved."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.food import settings  # noqa: E402
from pipeline.food.digestion.rules import BUILTIN_ENZYMES  # noqa: E402
from pipeline.food.gut_stability.stage import FoodGutStability, cleavage_sites  # noqa: E402
from pipeline.food.screening import checks, screening  # noqa: E402
from schemas.food import FoodCandidate, ParentOrigin  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402


def cand(seq: str) -> FoodCandidate:
    return FoodCandidate(id=seq, sequence=seq, origin_type="DIGESTION_DERIVED", parents=[ParentOrigin(protein_id="P1")], predictions={"screening": {}})


def test_pepsin_and_trypsin_use_the_digestion_rules_and_chymotrypsin_the_low_specificity_rule():
    assert set(settings.GUT_ENZYMES.values()) <= set(BUILTIN_ENZYMES)
    from pipeline.food.digestion.settings import DEFAULT_SETTINGS

    used = {e for p in DEFAULT_SETTINGS.pathways for st in p.stages for e in st.enzymes}
    assert {settings.GUT_ENZYMES["pepsin"], settings.GUT_ENZYMES["trypsin"]} <= used
    assert settings.GUT_ENZYMES["chymotrypsin"] == "chymotrypsin_low" and "chymotrypsin_low" not in used  # chosen for survival, from the protease benchmark


def test_the_low_specificity_rule_cuts_after_leucine_and_methionine_where_the_narrow_rule_does_not():
    assert cleavage_sites("AALAGG")["chymotrypsin"] == [3] and cleavage_sites("AAMAGG")["chymotrypsin"] == [3]
    assert BUILTIN_ENZYMES["chymotrypsin_high"].sites("AALAGG") == [] and BUILTIN_ENZYMES["chymotrypsin_high"].sites("AAMAGG") == []


def test_sites_are_one_based_p1_positions_of_internal_bonds():
    sites = cleavage_sites("AKVLEPKR")
    assert sites["trypsin"] == [2, 7]  # K2-V, K7-R; R8 is the last residue, so no bond after it; K6-P is not cut (P at P1')
    assert all(s < len("AKVLEPKR") for v in sites.values() for s in v)


def test_uncut_bond_fraction_is_the_share_of_internal_bonds_no_protease_cuts():
    assert screening.stability_scores({"pepsin": [], "trypsin": [], "chymotrypsin": []}, 8)["uncut_bond_fraction"] == 1.0
    assert screening.stability_scores({"pepsin": [], "trypsin": [2, 7], "chymotrypsin": []}, 8)["uncut_bond_fraction"] == round(1 - 2 / 7, 3)
    s = screening.stability_scores({"pepsin": [2], "trypsin": [2], "chymotrypsin": [2, 3]}, 5)  # a bond cut by several proteases counts once
    assert s["uncut_bond_fraction"] == 0.5 and (s["cleave_pepsin_sites"], s["cleave_chymotrypsin_sites"]) == (1, 2)
    assert screening.stability_scores({"pepsin": [1, 2, 3, 4], "trypsin": [], "chymotrypsin": []}, 5)["uncut_bond_fraction"] == 0.0


def test_the_stage_scores_every_candidate_and_names_its_rules():
    out = FoodGutStability().run([cand("AKVLEPKR"), cand("GGGGGG")], StageConfig(), SimpleNamespace())
    a, b = (c.predictions["cleavage"] for c in out)
    assert a["cleave_trypsin_sites"] == 2 and 0.0 < a["uncut_bond_fraction"] < 1.0
    assert b["uncut_bond_fraction"] == 1.0 and (b["cleave_pepsin_sites"], b["cleave_trypsin_sites"], b["cleave_chymotrypsin_sites"]) == (0, 0, 0)
    assert a["rules"] == settings.GUT_ENZYMES and "Expasy" in a["method"]


def test_a_rule_based_stability_is_always_limited_and_not_run_without_a_score():
    assert checks.stability_applicability({"length": 10, "uncut_bond_fraction": 0.5}) == "limited"
    assert checks.stability_applicability({"length": 40, "uncut_bond_fraction": 1.0}) == "limited"
    assert checks.stability_applicability({"length": 10, "uncut_bond_fraction": np.nan}) == "not_run"


def test_no_candidates_is_a_no_op():
    assert FoodGutStability().run([], StageConfig(), SimpleNamespace()) == []


def test_the_cut_positions_per_protease_are_reported_next_to_the_counts_and_the_value_says_what_it_is_not():
    s = screening.stability_scores({"pepsin": [3], "trypsin": [2, 7], "chymotrypsin": []}, 8)
    assert s["cleavage_positions"] == {"pepsin": [3], "trypsin": [2, 7], "chymotrypsin": []} and s["cleave_trypsin_sites"] == 2
    assert "does not measure survival in the gastrointestinal tract" in settings.UNCUT_BOND_FRACTION_STATEMENT


def test_the_stage_records_the_positions_and_the_statement_for_every_candidate():
    stage = FoodGutStability()
    (out,) = stage.run([FoodCandidate(id="P1", sequence="AETYNPR", origin_type="SYNTHETIC_DESIGN")], StageConfig(), SimpleNamespace())
    cleavage = out.predictions["cleavage"]
    assert cleavage["cleavage_positions"]["chymotrypsin"] == [4] and cleavage["statement"] == settings.UNCUT_BOND_FRACTION_STATEMENT
    assert cleavage["uncut_bond_fraction"] == round(1 - 1 / 6, 3)
