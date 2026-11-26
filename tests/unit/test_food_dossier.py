"""Dossier and experiment sheet of a food run, built from four real ranked peptides of a real-weights run and Phase 1's reference controls."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import pytest

from schemas.food_dossier import (ASSAY_DESIGN_FIELDS, DECIDED_STATUS, EVIDENCE_TYPE_MEANING, MODEL_OUTPUT_MEANING, MODEL_SCORE_MEANING, PROPOSED_STATUS, SHEET_COLUMNS, UNCUT_BOND_FRACTION_STATEMENT, build_dossier, dossier_html,
                                  experiment_sheet, experiment_sheet_csv)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "food"
CANDIDATES = [json.loads(line) for line in (FIXTURES / "run" / "s07_food_ranking_decision.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
CONTROLS = list(csv.DictReader((FIXTURES / "phase1" / "reference_controls.csv").open(encoding="utf-8")))
BY = {c["sequence"]: c for c in CANDIDATES}
DESIGNED = BY["AETYNPR"]  # rank 1, designed, shortlist 1, SEQUENCE_SIMILARITY
NOT_SHORTLISTED = BY["LKPTPEGD"]  # rank 7, digested, CONTAINS_REFERENCE
EXACT = BY["IPAVFK"]  # rank 24, digested, EXACT_SEQUENCE, shortlist 3
REVIEW = {"peptide_id": EXACT["id"], "sequence": "IPAVFK", "decision": "approve", "reason": "Exact match to a published primary-assay peptide.", "reviewer": "Hazara",
          "decided_at": "2026-11-25T08:00:00+00:00"}


def test_the_dossier_traces_the_peptide_to_its_parents_evidence_and_screening():
    d = build_dossier(EXACT, CONTROLS)
    assert d.sequence == "IPAVFK" and d.rank == 24 and d.shortlist_position == 3 and d.status == PROPOSED_STATUS
    assert d.provenance["parents"] and d.provenance["parents"][0]["protein_id"] and d.provenance["parent_clusters"]
    assert set(d.screening["models"]) == {"allergenicity", "cytotoxicity", "solubility", "aggregation", "synthesis_feasibility", "bitterness"}
    assert d.screening["applicability_overall"] and "safety" in d.screening and d.summary and d.limitations and d.properties["length"] == len(d.sequence)
    assert d.gut_stability["cut_sites_per_protease"].keys() == {"pepsin", "trypsin", "chymotrypsin"} and d.gut_stability["rules"] == EXACT["predictions"]["cleavage"]["rules"]


def test_the_evidence_has_a_type_a_source_and_says_what_it_applies_to():
    d = build_dossier(EXACT, [])
    c = d.evidence["candidate"]
    assert (c["applies_to"], c["evidence_type"], c["evidence_peptide"]) == ("CANDIDATE", "EXACT_SEQUENCE", "IPAVFK") and c["meaning"] == EVIDENCE_TYPE_MEANING["EXACT_SEQUENCE"]
    assert c["reported_effect"] in ("positive", "mixed") and c["source"]["study"] and c["source"]["assay"] and c["source"]["food_derivation"]
    assert "not proof of GLP-1 activity" in c["note"] and d.evidence["gate"]["passed"] is True and d.evidence["template"] is None
    similar = build_dossier(NOT_SHORTLISTED, []).evidence
    assert similar["candidate"]["evidence_type"] == "CONTAINS_REFERENCE" and similar["gate"]["passed"] is False  # only an exact sequence match passes the gate


def test_a_designed_peptides_template_evidence_is_shown_but_never_transferred():
    d = build_dossier(DESIGNED, [])
    own, template = d.evidence["candidate"], d.evidence["template"]
    assert own["applies_to"] == "CANDIDATE" and template["applies_to"] == "TEMPLATE" and d.evidence["gate"]["passed"] is False
    screening = DESIGNED["predictions"]["screening"]
    assert template["evidence_type"] == screening["template_evidence_type"] and own["evidence_type"] == screening["evidence_type"]
    assert d.evidence["designed_peptide_rule"] and "never transferred" in d.evidence["designed_peptide_rule"]


def test_a_designed_peptide_shows_its_template_edits_and_guide_a_digested_one_does_not():
    designed = build_dossier(DESIGNED, [])
    assert designed.provenance["design"]["template_sequence"] and designed.provenance["design"]["edits"] and designed.provenance["design"]["guide_peptide"]
    assert designed.provenance["generation_method"] == "reference_guided_edit" and designed.digestion["digestion_product_type"] is None
    assert "Not a digestion product" in designed.digestion["product_type_meaning"]
    assert "design" not in build_dossier(EXACT, []).provenance


def test_a_digested_peptide_says_it_is_predicted_not_observed_and_whether_it_is_a_final_product():
    d = build_dossier(EXACT, [])
    assert d.evidence_class == "THEORETICAL" and d.digestion["observation_status"] == "PREDICTED" and "Not observed by LC-MS/MS" in d.digestion["statement"]
    assert d.digestion["digestion_product_type"] == "FINAL_PRODUCT" and d.digestion["lcms_evidence"] == []
    assert {p["digestion_product_type"] for p in d.provenance["parents"]} <= {"FINAL_PRODUCT", "INTERMEDIATE_PRODUCT"}


def test_ige_binding_is_its_own_risk_beside_the_other_safety_model_the_liabilities_and_what_did_not_run():
    s = build_dossier(EXACT, []).safety_screening
    ige = s["ige_binding"]
    import re

    p = float(re.search(r"p_allergen=([0-9.]+)", ige["detail"]).group(1))
    assert ige["status"] == "RUN" and ige["label"] and ige["ige_binding_score"] == pytest.approx(p, abs=0.006) and "not a food-safety clearance" in ige["note"]
    assert ige["ranking_score"] == pytest.approx(1 - p, abs=0.006) and "predicted_probability" not in ige and "not a measured probability" in ige["note"]  # the report shows the model's own value, not 1 minus it
    assert set(s["other_safety_models"]) == {"cytotoxicity"} and "allergenicity" not in s["other_safety_models"] and s["other_safety_models"]["cytotoxicity"]["higher_means"] == "more likely to be cytotoxic"
    assert s["predictions_not_run"] == [] and s["applicability_limits"]["allergenicity"] and s["rule"].endswith("never counted as a pass.")


def test_a_model_that_did_not_run_is_not_a_pass_and_oxidation_risks_are_named_in_words():
    row = json.loads(json.dumps(EXACT))
    del row["predictions"]["models"]["allergenicity"]
    row["predictions"]["screening"]["liabilities"] = "methionine_oxidation;cysteine;asp_gly"
    row["predictions"]["ranking"]["predictions_not_run"] = "allergenicity"
    s = build_dossier(row, []).safety_screening
    assert s["ige_binding"]["status"] == "NOT_RUN" and "not a pass" in s["ige_binding"]["note"] and s["predictions_not_run"] == ["allergenicity"]
    assert [x["name"] for x in s["oxidation_risks"]] == ["methionine_oxidation"] and "oxidation risk" in s["oxidation_risks"][0]["meaning"]
    assert [x["name"] for x in s["sequence_liabilities"]] == ["methionine_oxidation", "cysteine", "asp_gly"]


def test_the_uncut_bond_fraction_comes_with_cut_positions_and_the_statement_that_it_is_not_survival():
    g = build_dossier(DESIGNED, []).gut_stability
    positions = g["predicted_cleavage_positions"]
    assert g["uncut_bond_fraction"] == DESIGNED["predictions"]["cleavage"]["uncut_bond_fraction"] and positions.keys() == {"pepsin", "trypsin", "chymotrypsin"}
    assert [len(positions[k]) for k in ("pepsin", "trypsin", "chymotrypsin")] == [g["cut_sites_per_protease"][k] for k in ("pepsin", "trypsin", "chymotrypsin")]
    assert g["statement"] == UNCUT_BOND_FRACTION_STATEMENT and "does not measure survival in the gastrointestinal tract" in g["statement"] and "gut_survival" not in g


def test_the_comparison_materials_name_parent_proteins_hydrolysates_and_reference_peptides():
    cm = build_dossier(DESIGNED, CONTROLS).comparison_materials
    assert cm["parent_proteins"] and all(p["accession"] for p in cm["parent_proteins"])
    assert cm["hydrolysates"] and all(h["pathway"] and h["stages"] for h in cm["hydrolysates"]) and "not existing records" in cm["note"]
    roles = [r["role"] for r in cm["reference_peptides"]]
    assert "template fragment" in roles and "reference peptide the design was steered toward" in roles and roles.count("positive_control") == 3 and roles.count("negative_control") == 3
    assert any(r["role"].startswith("closest evidence peptide") for r in cm["reference_peptides"])


def test_the_dossier_sets_no_value_the_biology_lead_must_choose_and_claims_no_efficacy():
    d = build_dossier(DESIGNED, CONTROLS)
    assert d.assay_plan["set_by_biology_lead"].keys() == set(ASSAY_DESIGN_FIELDS) and all(v is None for v in d.assay_plan["set_by_biology_lead"].values())
    assert d.review["decision"] is None and d.review["allowed_decisions"] == ["approve", "hold", "reject"] and d.review["reviewer"] is None and d.review["reason"] is None
    assert any("GLP-1 classifier is not validated" in s for s in d.not_established) and any("not a safety clearance" in s for s in d.not_established)
    assert d.evidence_class == "THEORETICAL" and [c["role"] for c in d.controls].count("positive_control") == 3


def test_a_saved_decision_and_its_reason_appear_in_the_dossier_and_change_its_status():
    d = build_dossier(EXACT, CONTROLS, REVIEW)
    assert d.review["decision"] == "approve" and d.review["reason"] == REVIEW["reason"] and d.review["reviewer"] == "Hazara" and d.review["decided_at"] == REVIEW["decided_at"]
    assert d.status == DECIDED_STATUS["approve"] and d.rank == 24  # the rank is the ranking's, not the review's
    assert build_dossier(EXACT, CONTROLS, {**REVIEW, "decision": "reject"}).status == DECIDED_STATUS["reject"]


def test_the_experiment_sheet_lists_the_shortlist_in_order_then_the_controls_with_empty_design_cells():
    rows = experiment_sheet(CANDIDATES, CONTROLS)
    candidates = [r for r in rows if r["row_type"] == "candidate"]
    assert [r["shortlist_position"] for r in candidates] == [1, 2, 3] and [r["rank"] for r in candidates] == [1, 4, 24]  # the unshortlisted rank-7 peptide is left out
    assert [r["row_type"] for r in rows[len(candidates):]] == ["positive_control"] * 3 + ["negative_control"] * 3
    for r in candidates:
        assert r["status"] == PROPOSED_STATUS and r["parent_proteins"] and r["decision"] == "" == r["reviewer"] == r["comments"] and r["evidence_type"] and r["predicted_ige_binding"]
        assert all(r[c] == "" for c in ("glp1_endpoint", "concentration_range", "exposure_time", "cell_line", "vehicle", "viability_control", "biological_replicates"))
    assert [r["evidence_type"] for r in candidates] == ["SEQUENCE_SIMILARITY", "SEQUENCE_SIMILARITY", "EXACT_SEQUENCE"] and [r["evidence_gate_passed"] for r in candidates] == ["no", "no", "yes"]
    assert candidates[0]["template_evidence_type"] and candidates[2]["template_evidence_type"] == ""
    assert all(set(r) == set(SHEET_COLUMNS) for r in rows)


def test_the_sheet_carries_the_saved_decisions_and_reasons_and_only_those():
    rows = experiment_sheet(CANDIDATES, CONTROLS, {EXACT["id"]: REVIEW})
    by_seq = {r["sequence"]: r for r in rows if r["row_type"] == "candidate"}
    assert (by_seq["IPAVFK"]["decision"], by_seq["IPAVFK"]["reviewer"], by_seq["IPAVFK"]["comments"], by_seq["IPAVFK"]["status"]) == ("approve", "Hazara", REVIEW["reason"], DECIDED_STATUS["approve"])
    assert by_seq["AETYNPR"]["decision"] == "" and by_seq["AETYNPR"]["status"] == PROPOSED_STATUS


def test_the_csv_round_trips_and_keeps_every_row():
    text = experiment_sheet_csv(CANDIDATES, CONTROLS)
    back = list(csv.DictReader(io.StringIO(text)))
    assert back[0].keys() == set(SHEET_COLUMNS) and len(back) == 3 + 6 and back[0]["sequence"] == DESIGNED["sequence"]


def test_the_html_page_escapes_every_value_and_shows_the_review():
    hostile = json.loads(json.dumps(DESIGNED))
    hostile["predictions"]["ranking"]["rationale"] = "<script>alert(1)</script>"
    hostile["parents"][0]["protein_id"] = '"><img src=x onerror=alert(2)>'
    page = dossier_html(build_dossier(hostile, [{"role": "positive_control", "sequence": "<b>X</b>", "study": "s", "assay": "a", "n_observations": "1"}],
                                      {**REVIEW, "reason": "<i>why</i>"}))
    assert "<script>" not in page and "<img" not in page and "<b>X</b>" not in page and "<i>why</i>" not in page and "&lt;script&gt;" in page and "&lt;i&gt;why&lt;/i&gt;" in page
    assert DESIGNED["sequence"] in page and "to be set by the biology lead" in page and "Predicted IgE binding" in page and "SEQUENCE_SIMILARITY" in page


@pytest.mark.parametrize("length,reachable", [(4, 0), (6, 2), (8, 4), (10, 6), (3, 0)])
def test_pepsin_reach_follows_the_table_of_the_protease_report(length, reachable):
    from schemas.food_dossier import pepsin_reachable_bonds

    assert pepsin_reachable_bonds(length) == reachable


DETAIL_PROBABILITY = {"allergenicity": "p_allergen=", "cytotoxicity": "P(cytotoxic)=", "solubility": "P(soluble in ultrapure water)=", "aggregation": "P(aggregation-prone)=",
                      "synthesis_feasibility": "P(synthesis feasible)=", "bitterness": "P(bitter"}


def detail_probability(name: str, detail: str) -> float:
    """The probability in a model's detail text; the bitterness text of earlier runs says 'P(bitter, threshold-aligned)=', newer runs 'P(bitter)='."""
    pattern = r"P\(bitter[^)]*\)=" if name == "bitterness" else re.escape(DETAIL_PROBABILITY[name])
    return float(re.search(pattern + r"([0-9.]+)", detail).group(1))


def test_every_score_is_what_the_dossier_says_it_is_and_a_higher_score_is_the_favourable_outcome():
    """The score is 1 - p or p, p being the probability in the model's own detail text; checked on every model of every real peptide of the fixture."""
    import re

    assert set(MODEL_SCORE_MEANING) == set(DETAIL_PROBABILITY)
    checked = 0
    for c in CANDIDATES:
        d = build_dossier(c, [])
        assert set(d.screening["score_meaning"]) == set(c["predictions"]["models"])
        for name, m in c["predictions"]["models"].items():
            p = detail_probability(name, m["detail"])
            form = MODEL_SCORE_MEANING[name][0]
            assert m["score"] == pytest.approx(1 - p if form == "inverse" else p, abs=0.006), (c["sequence"], name, m["score"], p)
            assert d.screening["score_meaning"][name] == MODEL_SCORE_MEANING[name][1]
            checked += 1
    assert checked == len(CANDIDATES) * 6
    # in every inverse model a high probability of the bad outcome gives a low score, and the label names that outcome
    bad = {"allergenicity": "predicted_IgE_binding", "cytotoxicity": "predicted_cytotoxic", "aggregation": "predicted_aggregation_prone", "bitterness": "predicted_bitter"}
    for c in CANDIDATES:
        for name, outcome in bad.items():
            m = c["predictions"]["models"][name]
            assert (m["label"] == outcome) == (m["score"] <= 0.5), (c["sequence"], name, m["score"], m["label"])


def test_the_developability_text_says_what_it_is_made_of():
    d = build_dossier(CANDIDATES[0], [])
    assert "solubility, aggregation, synthesis-feasibility and bitterness" in d.screening["developability_meaning"] and "sequence-liability" in d.screening["developability_meaning"]


def test_every_model_is_reported_as_its_own_output_for_what_it_predicts_never_as_one_minus_it():
    """The output is the probability in the model's own detail text; where the ranking turns it around (score = 1 - output) the report still shows the output."""
    import re

    assert set(MODEL_OUTPUT_MEANING) == set(DETAIL_PROBABILITY)
    for c in CANDIDATES:
        outputs = build_dossier(c, []).screening["model_outputs"]
        assert set(outputs) == set(c["predictions"]["models"])
        for name, m in c["predictions"]["models"].items():
            p = detail_probability(name, m["detail"])
            o = outputs[name]
            assert o["output"] == pytest.approx(p, abs=0.006), (c["sequence"], name, o["output"], p)
            assert o["higher_means"] == MODEL_OUTPUT_MEANING[name][2] and o["predicts"] == MODEL_OUTPUT_MEANING[name][1]
            if MODEL_OUTPUT_MEANING[name][0] == "inverse":
                assert o["output"] == pytest.approx(1 - m["score"], abs=0.006) and (m["label"].startswith("predicted_") and ("not" not in m["label"] and "non" not in m["label"])) == (o["output"] >= 0.5)


def test_a_saved_output_is_used_as_it_is_and_an_older_run_gets_it_recovered_from_the_ranking_value():
    from schemas.food_dossier import model_outputs

    saved = model_outputs({"cytotoxicity": {"score": 0.2, "output": 0.81, "label": "predicted_cytotoxic", "applicability": "in_domain"}})
    assert saved["cytotoxicity"]["output"] == 0.81
    older = model_outputs({"cytotoxicity": {"score": 0.2, "label": "predicted_cytotoxic", "applicability": "in_domain"}, "solubility": {"score": 0.9, "label": "x", "applicability": "limited"}})
    assert older["cytotoxicity"]["output"] == 0.8 and older["solubility"]["output"] == 0.9  # inverse models: 1 - score; direct models: score


def with_ends(candidate, n, c):
    return {**candidate, "parents": [{**p, "protein_n_terminus": n, "protein_c_terminus": c} for p in candidate["parents"]]}


def test_the_ends_of_the_parent_protein_chain_are_in_the_json_html_and_sheet_and_not_recorded_when_absent():
    from schemas.food_dossier import build_dossier, dossier_html
    d = build_dossier(with_ends(EXACT, "SIGNAL_CLEAVED", "SEQ_END"), CONTROLS)
    assert d.comparison_materials["parent_proteins"][0]["protein_n_terminus"] == "SIGNAL_CLEAVED"
    page = dossier_html(d)
    assert "Protein N-terminus" in page and "SIGNAL_CLEAVED (a signal peptide was cut off at this end)" in page and "SEQ_END (last residue of the UniProt sequence" in page
    row = [r for r in experiment_sheet([with_ends(EXACT, "SIGNAL_CLEAVED", "SEQ_END")], CONTROLS) if r["row_type"] == "candidate"][0]
    assert row["parent_n_terminus"].endswith(":SIGNAL_CLEAVED") and row["parent_c_terminus"].endswith(":SEQ_END")
    assert "not recorded" in dossier_html(build_dossier(EXACT, CONTROLS))
