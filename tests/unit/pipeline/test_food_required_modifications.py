"""The required pipeline modifications of 2026-11-25: evidence labels and gate, IgE binding as its own risk, unavailable results never a pass, digestion product type, observed needs LC-MS/MS."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.food import settings  # noqa: E402
from pipeline.food.evidence import EvidenceLookup, load_evidence  # noqa: E402
from pipeline.food.ranking.stage import MODEL_OBJECTIVES, derive  # noqa: E402
from schemas.food import FoodCandidate, ParentOrigin  # noqa: E402

LOOKUP = EvidenceLookup(load_evidence())
EVIDENCE = {e["sequence"]: e for e in load_evidence()}


@pytest.mark.parametrize("sequence,match,kind", [
    ("LKPT", "exact", "EXACT_SEQUENCE"),                       # the candidate is the evidence peptide
    ("LKPTPEGD", "contains", "CONTAINS_REFERENCE"),            # the candidate contains the whole evidence peptide LKPT
    ("VLP", "contains", "CONTAINED_IN_REFERENCE"),             # the candidate is a subsequence of the evidence peptide VLPL
    ("AETYNPR", "near", "SEQUENCE_SIMILARITY"),                # alignment score at or above the cut-off, no containment
    ("GGGGGGGG", "none", "NO_EVIDENCE"),
])
def test_the_five_evidence_types_follow_the_existing_matching_logic(sequence, match, kind):
    found = LOOKUP.lookup(sequence)
    assert (found["evidence_match"], found["evidence_type"]) == (match, kind)
    if kind == "NO_EVIDENCE":
        assert found["evidence_peptide"] == "" and found["evidence_direction"] == "none"
    elif kind in ("CONTAINS_REFERENCE", "EXACT_SEQUENCE"):
        assert found["evidence_peptide"] in sequence
    elif kind == "CONTAINED_IN_REFERENCE":
        assert sequence in found["evidence_peptide"] and len(found["evidence_peptide"]) >= settings.MIN_FUZZY_LENGTH


def row(kind, direction, **extra):
    return {"digestion_support": 0.6, "developability_rule": 1.0, "evidence_type": kind, "evidence_direction": direction, "evidence_similarity": 1.0, "evidence_match": "exact", **extra}


@pytest.mark.parametrize("kind,direction,passed", [
    ("EXACT_SEQUENCE", "positive", True), ("EXACT_SEQUENCE", "mixed", True), ("EXACT_SEQUENCE", "negative", False),
    ("CONTAINS_REFERENCE", "positive", False), ("CONTAINED_IN_REFERENCE", "positive", False), ("SEQUENCE_SIMILARITY", "positive", False), ("NO_EVIDENCE", "none", False),
])
def test_only_an_exact_sequence_match_with_a_positive_or_mixed_effect_passes_the_evidence_gate(kind, direction, passed):
    assert derive(row(kind, direction), {}, set())["evidence_gate_passed"] is passed


def test_the_old_direct_and_exploratory_labels_are_gone():
    assert "evidence_gate" not in derive(row("EXACT_SEQUENCE", "positive"), {}, set())
    assert not hasattr(settings, "MIN_DIRECT_EVIDENCE") and settings.MIN_EXACT_EVIDENCE == 15 and not hasattr(settings, "SHORTLIST_MAX_WITHOUT_EXACT_EVIDENCE") and settings.SHORTLIST_EVIDENCE == ("EXACT_SEQUENCE", "CONTAINS_REFERENCE", "CONTAINED_IN_REFERENCE")


MODELS = {"allergenicity": {"score": 0.1, "label": "predicted_IgE_binding", "applicability": "in_domain"},
          "cytotoxicity": {"score": 0.9, "label": "predicted_not_cytotoxic", "applicability": "in_domain"}}


def test_ige_binding_is_not_folded_into_the_safety_score():
    assert MODEL_OBJECTIVES["allergenicity"] == "ige_binding" and MODEL_OBJECTIVES["cytotoxicity"] == "safety"
    out = derive(row("NO_EVIDENCE", "none"), MODELS, {"allergenicity", "cytotoxicity"})
    assert out["safety"] == pytest.approx(0.9)  # not min(0.1, 0.9): a poor IgE score does not lower it
    assert out["model_allergenicity_score"] == pytest.approx(0.1) and out["model_allergenicity_label"] == "predicted_IgE_binding" and out["applicability_allergenicity"] == "in_domain"


def test_a_model_that_did_not_run_is_never_a_pass():
    only_ige = derive(row("NO_EVIDENCE", "none"), {"allergenicity": MODELS["allergenicity"]}, {"allergenicity"})
    assert math.isnan(only_ige["safety"]) and only_ige["predictions_not_run"].split(";").count("cytotoxicity") == 1  # no cytotoxicity result: no safety score, not 1.0
    assert only_ige["applicability_cytotoxicity" if "applicability_cytotoxicity" in only_ige else "applicability_assessed"] != "in_domain"
    none = derive(row("NO_EVIDENCE", "none"), {}, set())
    assert math.isnan(none["safety"]) and "allergenicity" in none["predictions_not_run"] and none["coverage"] < 1


def test_a_peptide_is_final_or_intermediate_by_whether_an_enzyme_could_still_cut_inside_it():
    final = ParentOrigin(protein_id="P1:1-9", is_limit_product=True)
    middle = ParentOrigin(protein_id="P1:1-9", is_limit_product=False)
    assert (final.digestion_product_type, middle.digestion_product_type, ParentOrigin(protein_id="P1").digestion_product_type) == ("FINAL_PRODUCT", "INTERMEDIATE_PRODUCT", None)
    def peptide(*parents, origin="DIGESTION_DERIVED"):
        return FoodCandidate(id="X", sequence="AAAA", origin_type=origin, parents=list(parents))
    assert peptide(middle, final).digestion_product_type == "FINAL_PRODUCT"  # final in at least one pathway
    assert peptide(middle).digestion_product_type == "INTERMEDIATE_PRODUCT"
    assert peptide(final, origin="SYNTHETIC_DESIGN").digestion_product_type is None  # a designed peptide is not a digestion product
    assert peptide(final).model_dump()["digestion_product_type"] == "FINAL_PRODUCT"  # and it is saved with the candidate


def test_a_peptide_is_observed_only_with_lcms_evidence_and_the_pipeline_never_sets_it():
    parent = ParentOrigin(protein_id="P1:1-9")
    with pytest.raises(ValueError, match="OBSERVED only with LC-MS/MS evidence"):
        FoodCandidate(id="X", sequence="AAAA", origin_type="DIGESTION_DERIVED", parents=[parent], evidence_class="OBSERVED")
    seen = FoodCandidate(id="X", sequence="AAAA", origin_type="DIGESTION_DERIVED", parents=[parent], evidence_class="OBSERVED", lcms_evidence=[{"run": "r1"}])
    assert seen.evidence_class == "OBSERVED" and FoodCandidate(id="Y", sequence="AAAA", origin_type="DIGESTION_DERIVED", parents=[parent]).evidence_class == "THEORETICAL"


def test_no_stage_output_is_observed():
    from pipeline.food.digestion.engine import run as digest
    from pipeline.food.digestion.settings import DEFAULT_SETTINGS
    from pipeline.food.protein_library.protein import Protein

    _, peptides, _ = digest([Protein(substrate_id="P1:1-30", accession="P1", sequence="MKWVTFISLLLLFSSAYSRGVFRRDTHKSE", food_source="Whey", precursor_start=1, precursor_end=30)], DEFAULT_SETTINGS)
    assert peptides and {p["evidence_class"] for p in peptides} == {"THEORETICAL"}


def test_a_model_that_skipped_this_peptide_is_listed_as_not_run_and_is_not_counted_as_a_pass():
    models = {n: {"score": 0.8, "label": "x", "applicability": "in_domain"} for n in ("allergenicity", "cytotoxicity", "solubility", "synthesis_feasibility", "bitterness")}  # no aggregation
    ready = set(MODEL_OBJECTIVES)  # the aggregation model ran for other peptides
    out = derive(row("NO_EVIDENCE", "none", uncut_bond_fraction=0.8), models, ready)
    assert out["predictions_not_run"] == "aggregation" and out["applicability_not_assessed"] == "aggregation" and out["developability"] == pytest.approx(0.8)
    assert out["developability_basis"] != "none" and "model_aggregation_score" not in out  # left out of the score, not given a pass value
    assert derive(row("NO_EVIDENCE", "none", uncut_bond_fraction=0.8), {**models, "aggregation": {"score": 0.3, "label": "x", "applicability": "limited"}}, ready)["predictions_not_run"] == ""
