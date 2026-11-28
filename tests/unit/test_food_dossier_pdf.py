"""The dossier PDF: an overview page, a shared page, then one page per peptide, in plain words for a biology lead, nothing invented, the biology lead's cells empty."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from food.shared.dossier import build_dossier
from food.shared.dossier_pdf import MAX_PARENT_ROWS, MODEL, _text, cut_text, dossier_pdf, edit_text, page_scale, pathway_text, plain_text, split_outside_brackets

RUN = Path(__file__).resolve().parents[1] / "fixtures" / "food" / "run" / "s07_food_ranking_decision.jsonl"
CONTROLS = [{"role": "positive_control", "sequence": "LKPT", "study": "PMID1", "assay": "STC-1", "n_observations": "3"},
            {"role": "negative_control", "sequence": "DLVDK", "study": "PMID2", "assay": "STC-1", "n_observations": "2"}]
ROWS = [json.loads(line) for line in RUN.read_text(encoding="utf-8").splitlines()]
DOSSIERS = [build_dossier(r, CONTROLS) for r in ROWS]
DESIGNED = next(d for d in DOSSIERS if d.origin_type == "SYNTHETIC_DESIGN")
DIGESTED = next(d for d in DOSSIERS if d.origin_type == "DIGESTION_DERIVED")
REVIEW = {"peptide_id": DESIGNED.peptide_id, "sequence": DESIGNED.sequence, "decision": "hold", "reason": "Needs an IgE follow-up first.", "reviewer": "Hazara",
          "decided_at": "2026-11-25T08:00:00+00:00"}


def pages(dossiers):
    return [" ".join(p.extract_text().split()) for p in PdfReader(io.BytesIO(dossier_pdf(dossiers))).pages]


def test_there_are_an_overview_page_and_a_shared_page_then_exactly_one_page_per_dossier():
    assert len(pages(DOSSIERS)) == len(DOSSIERS) + 2 == 6 and len(pages(DOSSIERS[:1])) == 3


@pytest.mark.parametrize("dossier", DOSSIERS, ids=lambda d: d.sequence)
def test_the_page_carries_what_a_biology_lead_needs_under_plain_headings(dossier):
    text = pages([dossier])[2]
    for must in (dossier.sequence, dossier.peptide_id, dossier.status[:30].capitalize(), "Why it ranks here", "Peptide properties", "Where it comes from", "Compare with", "Safety screening",
                 "Predicted IgE binding (its own risk)", "Screening models", "Evidence", "Predicted cuts by gut enzymes (theoretical)", "Uncut-bond fraction", "Review", "Decision (approve / hold / reject)",
                 "Reason", "Reviewer"):
        assert must in text, must
    for name in dossier.screening["models"]:
        assert MODEL[name] in text


@pytest.mark.parametrize("dossier", DOSSIERS, ids=lambda d: d.sequence)
def test_no_field_name_python_value_or_internal_code_is_printed_on_any_page(dossier):
    text = " ".join(pages([dossier]))
    for banned in ("True", "False", "None", "net_charge", "evidence_basis", "gut_survival", "uncut_bond_fraction", "pareto", "DIGESTION_DERIVED", "SYNTHETIC_DESIGN", "THEORETICAL",
                   "predicted_", "_sites", "md5", "generator", "seed", "uncalibrated", "limit_product", "in_domain", "trypsin_only", "gastric_to_intestinal", "SEQUENCE_SIMILARITY",
                   "EXACT_SEQUENCE", "CONTAINS_REFERENCE", "CONTAINED_IN_REFERENCE", "NO_EVIDENCE", "FINAL_PRODUCT", "INTERMEDIATE_PRODUCT", "evidence_type", "evidence_gate", "Direct evidence"):
        assert banned not in text, banned


def test_the_evidence_is_typed_sourced_and_says_what_it_applies_to():
    text = pages([DESIGNED])[2]
    assert "This peptide: sequence similarity" in text and "Source of that evidence" in text and "Evidence gate: not passed" in text
    assert "Its template (shown, not transferred)" in text and "not proof of GLP-1 activity" in text
    exact = pages([build_dossier(next(r for r in ROWS if r["sequence"] == "IPAVFK"), CONTROLS)])[2]
    assert "This peptide: exact sequence" in exact and "Evidence gate: passed" in exact and "Its template" not in exact


def test_a_designed_peptide_says_it_is_designed_and_shows_template_edits_and_reference_in_words():
    text = pages([DESIGNED])[2]
    design = DESIGNED.provenance["design"]
    assert "This peptide was designed" in text and "How it was designed" in text and design["template_sequence"] in text and design["guide_peptide"] in text
    assert edit_text(design["edits"]) in text and "Digestion product" not in text


def test_a_digested_peptide_shows_where_it_is_released_and_whether_it_is_a_final_product():
    text = pages([DIGESTED])[2]
    assert "Digestion routes that release it" in text and "Digestion product" in text and "Predicted by digestion rules. Not observed by LC-MS/MS." in text
    assert "This peptide was designed" not in text and "How it was designed" not in text and "Released from a food protein by simulated digestion" in text


def test_ige_binding_is_its_own_risk_and_oxidation_risks_and_missing_predictions_are_in_words():
    row = json.loads(json.dumps(ROWS[0]))
    del row["predictions"]["models"]["cytotoxicity"]
    row["sequence"] = "AMDGYNPR"
    row["predictions"]["screening"]["liabilities"] = "methionine_oxidation;asp_gly"
    row["predictions"]["ranking"]["predictions_not_run"] = "cytotoxicity"
    text = pages([build_dossier(row, CONTROLS)])[2]
    assert "Predicted IgE binding (its own risk)" in text and "IgE-binding score" in text and "higher means more likely to bind IgE" in text and "M2: oxidation" in text
    assert "D3: deamidation or isoaspartate" in text and "Cytotoxicity Not run. Its absence is not a pass." in text and "Cytotoxicity | none" in text


def test_the_cut_positions_and_what_the_uncut_bond_fraction_is_not_are_stated():
    text = pages([DESIGNED])[2]
    assert "chymotrypsin: after Y4" in text and "does not measure survival in the gastrointestinal tract" in text and "The uncut-bond fraction is the share" in text
    assert cut_text("AETYNPR", [4]) == "after Y4" and cut_text("AETYNPR", []) == "none"


def test_the_comparison_materials_are_named_and_the_controls_stay_on_the_shared_page():
    text = pages([DESIGNED])[2]
    assert "Parent proteins" in text and "Parent hydrolysates to prepare" in text and "not existing records" in text and "Template fragment" in text
    assert "Reference peptide the design was steered toward" in text and "DLVDK" not in text


def test_a_saved_decision_and_reason_are_printed_and_the_overview_shows_the_review_status():
    overview, _, *peptides = pages([build_dossier(r, CONTROLS, REVIEW if r["id"] == DESIGNED.peptide_id else None) for r in ROWS])
    assert "On hold" in overview and "Pending" in overview
    designed_page = next(p for p in peptides if DESIGNED.peptide_id in p)
    assert "Decision On hold" in designed_page and "Needs an IgE follow-up first." in designed_page and "Reviewer Hazara" in designed_page and "On hold by the biology lead" in designed_page
    assert "Decision (approve / hold / reject)" not in designed_page


def test_no_value_is_written_into_the_biology_leads_cells():
    _, shared, peptide = [[line.strip() for line in p.extract_text().splitlines()] for p in PdfReader(io.BytesIO(dossier_pdf(DOSSIERS[:1]))).pages]
    labels = ["GLP-1 endpoint (active or total)", "Concentration range", "Exposure time", "Cell line", "Vehicle", "Viability control", "Interference control",
              "Biological replicates", "Technical replicates"]
    start = shared.index(labels[0])
    assert shared[start:start + 9] == labels and peptide[peptide.index("Decision (approve / hold / reject)"):][:3] == ["Decision (approve / hold / reject)", "Reason", "Reviewer"]


def test_the_overview_page_lists_the_shortlisted_peptides_in_shortlist_order_and_nothing_else():
    overview = pages(DOSSIERS)[0]
    for must in ("Shortlist for biological review", f"{len(DOSSIERS)} peptides", "Overview", "How to read the peptide pages", "Predicted IgE binding", "Evidence type", "Uncut bonds"):
        assert must in overview, must
    order = sorted(DOSSIERS, key=lambda d: (d.shortlist_position is None, d.shortlist_position or 0, d.rank))
    assert [overview.index(d.sequence) for d in order] == sorted(overview.index(d.sequence) for d in DOSSIERS)
    assert "Top " not in overview and "does not change the computed rank" in overview


@pytest.mark.parametrize("dossier", DOSSIERS, ids=lambda d: d.sequence)
def test_the_type_stays_readable_for_the_real_peptides(dossier):
    assert 0.85 <= page_scale(dossier) <= 1


def test_a_peptide_with_many_parents_still_fits_one_page_and_says_how_many_are_left_out():
    row = json.loads(json.dumps(ROWS[0]))
    row["parents"] = [{**row["parents"][0], "accession": f"P{i:03d}", "protein_id": f"P{i:03d}:1-99", "start_in_precursor": 1, "end_in_precursor": 6} for i in range(MAX_PARENT_ROWS + 15)]
    dossier = build_dossier(row, CONTROLS)
    text = pages([dossier])[2]
    assert len(pages([dossier])) == 3 and "15 more parent site(s)" in text and f"P{MAX_PARENT_ROWS - 1:03d}" in text and f"P{MAX_PARENT_ROWS:03d}" not in text


def test_the_pdf_does_not_print_the_word_uncalibrated_but_keeps_the_rest_of_each_model_detail():
    assert _text("p_allergen=0.68 (IgE binding only, not a food-safety clearance; uncalibrated)") == "p_allergen=0.68 (IgE binding only, not a food-safety clearance)"
    assert "calibrated" not in " ".join(pages(DOSSIERS)).lower()


def test_text_the_font_cannot_draw_is_replaced_and_markup_is_escaped():
    assert _text("a ≥ b → c") == "a &gt;= b -&gt; c" and _text("<b>&") == "&lt;b&gt;&amp;" and _text("5 µM · x") == "5 µM · x" and _text("α") == "?"


def test_the_shared_page_holds_what_is_the_same_for_every_peptide_and_the_peptide_pages_do_not_repeat_it():
    overview, shared, *peptides = pages(DOSSIERS)
    for must in ("Shared information", "Reference controls", "LKPT", "Not established", "Assay plan: set by the biology lead", "Concentration range", "Limitations that apply to every peptide"):
        assert must in shared, must
    for page in peptides:
        for repeated in ("Not established", "Reference controls", "Assay plan", "Concentration range", "Limitations that apply to every peptide"):
            assert repeated not in page, repeated
    assert "pareto_front" not in " ".join([overview, shared, *peptides])  # the ranking block names the Pareto front in words, never the field


def test_a_limitation_every_peptide_has_is_on_the_shared_page_only_and_one_only_some_have_stays_on_their_page():
    _, shared, *peptides = pages(DOSSIERS)
    designed_only = "activity of the edit is untested"
    assert designed_only not in shared and sum(designed_only in page for page in peptides) == sum(d.origin_type == "SYNTHETIC_DESIGN" for d in DOSSIERS)
    assert "Prior-art alert is not a freedom-to-operate opinion" in shared


def test_the_plain_helpers_only_rename_and_explain():
    assert edit_text("A1>S;K2>del") == "A1 to S; K2 deleted"
    assert plain_text("weak on model_allergenicity_score; digestion_plausibility") == "weak on allergenicity score; digestion plausibility"
    assert pathway_text("gastric_to_intestinal") == "stomach (pepsin, pH 1.3) then intestine (trypsin + chymotrypsin, pH 7.5)" and pathway_text("trypsin_only") == "intestine (trypsin, pH 7.5)"
    assert pathway_text("some_new_pathway") == "some new pathway"
    assert split_outside_brackets("a (b; c); d; e (f)") == ["a (b; c)", "d", "e (f)"]


def test_the_screening_table_shows_each_models_own_output_and_what_a_higher_value_means():
    import re

    text = " ".join(pages(DOSSIERS))
    assert "Model output (0-1)" in text and "A higher value means" in text
    for phrase in ("More likely to bind IgE", "More likely to be cytotoxic", "More likely to be soluble in water", "More likely to be aggregation-prone",
                   "More likely that synthesis is feasible", "More likely to be bitter"):
        assert phrase in text, phrase
    for banned in ("1 minus", "higher is better", "favourable outcome", "Developability score", "Cytotoxicity score", "Ranking value", "predicted probability"):
        assert banned not in text, banned
    # the value on the page is the model's own output: for the designed peptide AETYNPR the IgE-binding output is p_allergen, not 1 - p_allergen
    p = float(re.search(r"p_allergen=([0-9.]+)", ROWS[0]["predictions"]["models"]["allergenicity"]["detail"]).group(1))
    shown = f"{p:.2f}".rstrip("0").rstrip(".")
    page = pages([DESIGNED])[2]
    assert f"IgE binding Predicted IgE binding {shown} Within range More likely to bind IgE" in page or f"IgE binding Predicted IgE binding {shown} Limited More likely to bind IgE" in page
    assert f"IgE-binding score {shown}" in page


def test_the_overview_shows_the_ige_binding_score_and_the_cytotoxicity_output_not_inverted_values():
    overview = pages(DOSSIERS)[0]
    assert "IgE-binding score" in overview and "Cytotoxicity output" in overview and "Developability" not in overview and "Cytotoxicity score" not in overview



def test_a_peptide_the_aggregation_model_skipped_shows_it_as_not_run_with_the_reason():
    row = json.loads(json.dumps(ROWS[0]))
    row["sequence"] = "HPAS"
    del row["predictions"]["models"]["aggregation"]
    row["predictions"]["ranking"]["predictions_not_run"] = "aggregation"
    dossier = build_dossier(row, CONTROLS)
    assert dossier.safety_screening["predictions_not_run"] == ["aggregation"]
    text = pages([dossier])[2]
    assert "Aggregation Not run - Not run The aggregation model needs at least 5 amino acids; this peptide has 4." in text
    assert "Models not run | left out of the safety score (outside their range) Aggregation | none" in text
    full = pages([DESIGNED])[2]
    assert "Aggregation Predicted not aggregation prone" in full and "Aggregation Not run" not in full


def test_the_ranking_strengths_and_weaknesses_are_in_words_about_what_the_models_predict():
    from food.shared.dossier_pdf import reasons

    text = reasons("Pareto front 1 of 4, cluster 52; near pool-best on coverage, model_cytotoxicity_score, model_solubility_score; weak on model_allergenicity_score, model_bitterness_score, "
                   "digestion_plausibility; evidence gate: x")
    assert text == ("Among the best of the pool on: coverage, low predicted cytotoxicity, high predicted solubility. "
                    "Weak on: high predicted IgE binding, high predicted bitterness, digestion plausibility.")
    assert "allergenicity score" not in pages([DESIGNED])[2] and "Weak on: high predicted IgE binding" in pages([DESIGNED])[2]
