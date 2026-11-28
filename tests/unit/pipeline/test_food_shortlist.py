"""The biology shortlist rule: eligibility by evidence relationship, a model screen, one per family, at most 4 designed, up to 10, rank order, and a reason for every peptide."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from food.c_discovery.ranking import shortlist  # noqa: E402

LABELS = {"model_cytotoxicity_label": ("predicted_cytotoxic", "cytotoxic"), "model_aggregation_label": ("predicted_aggregation_prone", "aggregation-prone"),
          "model_solubility_label": ("predicted_insoluble", "insoluble"), "model_synthesis_feasibility_label": ("predicted_difficult", "hard to synthesise")}


def peptide(rank, family="f1", origin="DIGESTION_DERIVED", kind="CONTAINS_REFERENCE", effect="positive", **labels):
    return {"peptide_id": f"P{rank}", "sequence": f"SEQ{rank}", "rank": rank, "origin_type": origin, "evidence_type": kind, "evidence_direction": effect, "evidence_gate_passed": kind == "EXACT_SEQUENCE",
            "parent_clusters": f"family:{family}", "rationale": "r", "limitations": "l", **labels}


def by_id(rows):
    return {d["peptide_id"]: d for d in shortlist.shortlist_decisions(rows)}


def test_peptides_without_an_exact_match_are_not_capped():
    rows = [peptide(i, family=f"f{i}", kind="CONTAINS_REFERENCE") for i in range(1, 9)]
    assert [s["rank"] for s in shortlist.make_shortlist(rows)] == list(range(1, 9))  # eight peptides, none with an exact match


@pytest.mark.parametrize("kind", ["EXACT_SEQUENCE", "CONTAINS_REFERENCE", "CONTAINED_IN_REFERENCE"])
def test_an_exact_contains_or_contained_in_relationship_makes_a_peptide_eligible(kind):
    assert by_id([peptide(1, kind=kind)])["P1"]["shortlisted"]


@pytest.mark.parametrize("kind", ["SEQUENCE_SIMILARITY", "NO_EVIDENCE"])
def test_similarity_only_and_no_evidence_are_not_eligible_and_the_reason_says_why(kind):
    d = by_id([peptide(1, kind=kind)])["P1"]
    assert not d["shortlisted"] and d["reason"].startswith("not eligible")


def test_a_relationship_to_a_peptide_with_a_negative_effect_is_not_eligible():
    assert not by_id([peptide(1, effect="negative")])["P1"]["shortlisted"] and by_id([peptide(1, effect="mixed")])["P1"]["shortlisted"]


@pytest.mark.parametrize("label", list(LABELS))
def test_a_peptide_a_model_calls_unfavourable_is_left_off_with_the_reason(label):
    value, words = LABELS[label]
    d = by_id([peptide(1, **{label: value})])["P1"]
    assert not d["shortlisted"] and d["reason"] == f"left off: a model calls it unfavourable ({words})"


def test_ige_binding_and_bitterness_do_not_exclude_a_peptide():
    row = peptide(1, model_cytotoxicity_label="predicted_not_cytotoxic", model_bitterness_label="predicted_bitter", model_allergenicity_label="predicted_IgE_binding")
    assert by_id([row])["P1"]["shortlisted"]


def test_one_peptide_per_parent_family_the_higher_ranked_wins_and_the_reason_says_so():
    d = by_id([peptide(1, family="a"), peptide(2, family="a"), peptide(3, family="b")])
    assert d["P1"]["shortlisted"] and not d["P2"]["shortlisted"] and "same parent protein family" in d["P2"]["reason"] and d["P3"]["shortlisted"]


def test_at_most_four_designed_peptides():
    rows = [peptide(i, family=f"f{i}", origin="SYNTHETIC_DESIGN") for i in range(1, 7)] + [peptide(7, family="f7")]
    d = by_id(rows)
    assert [d[f"P{i}"]["shortlisted"] for i in range(1, 8)] == [True] * 4 + [False, False, True]
    assert d["P5"]["reason"] == "skipped: 4 designed peptides are already on the shortlist"


def test_the_list_holds_at_most_ten_and_the_rest_are_marked_not_needed():
    d = by_id([peptide(i, family=f"f{i}") for i in range(1, 13)])
    assert sum(x["shortlisted"] for x in d.values()) == 10 and d["P11"]["reason"] == "not needed: the shortlist already holds 10 peptides"


def test_the_shortlist_is_in_rank_order_a_lower_ranked_exact_match_does_not_jump_the_queue():
    rows = [peptide(1, family="a", kind="CONTAINS_REFERENCE"), peptide(2, family="b", kind="EXACT_SEQUENCE"), peptide(3, family="c", kind="CONTAINED_IN_REFERENCE")]
    out = shortlist.make_shortlist(rows)
    assert [s["rank"] for s in out] == [1, 2, 3] and [s["shortlist_position"] for s in out] == [1, 2, 3]


def test_every_ranked_peptide_gets_exactly_one_decision_with_a_reason():
    rows = [peptide(1), peptide(2, family="x", kind="NO_EVIDENCE"), peptide(3, family="y", model_cytotoxicity_label="predicted_cytotoxic")]
    decisions = shortlist.shortlist_decisions(rows)
    assert [d["peptide_id"] for d in decisions] == ["P1", "P2", "P3"] and all(d["reason"] for d in decisions)
    assert decisions[0]["reason"] == "on the shortlist (position 1)"
