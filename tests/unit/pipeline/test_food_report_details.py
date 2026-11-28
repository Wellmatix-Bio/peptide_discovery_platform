"""Digestion detail and its consistency check, evidence alignment, liability positions, masses of designed peptides, and the run summary of the ranking."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from food.shared import liabilities  # noqa: E402
from food.b_digestion import masses  # noqa: E402
from food.b_digestion import DEFAULT_SETTINGS, run  # noqa: E402
from food.b_digestion.explain import explain_candidates  # noqa: E402
from food.pipeline.stages.digestion import FoodDigestion  # noqa: E402
from food.c_discovery.evidence import load_evidence  # noqa: E402
from food.c_discovery.evidence.lookup import EvidenceLookup  # noqa: E402
from food.a_proteins.protein import Protein  # noqa: E402
from food.c_discovery.screening import screening  # noqa: E402
from food.shared.candidate import FoodCandidate, ParentOrigin  # noqa: E402
from food.shared.dossier import ranking_summary  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402

PROTEIN = Protein(substrate_id="P1:11-40", accession="P1", precursor_start=11, precursor_end=40, sequence="MKAAKGGRCCFLKDGPQQQQNGAAAAWR", food_source="Whey")


@pytest.fixture(scope="module")
def digested():
    return FoodDigestion().run([], StageConfig(), SimpleNamespace(proteins=[PROTEIN]))


@pytest.fixture(scope="module")
def lookup():
    return EvidenceLookup(load_evidence())


# --- 1. digestion detail and its consistency ------------------------------------------------------------------

def test_every_origin_of_a_digested_peptide_gets_its_stages_cuts_flanking_residues_and_product_type(digested):
    warnings = explain_candidates(digested, [PROTEIN])
    assert warnings == []  # the recorded origins agree with the digestion run again
    seen_final = seen_intermediate = False
    for c in digested:
        for p in c.parents:
            d = p.digestion_detail
            assert d["pathway"] == p.pathway and d["stages"] and d["protein_id"] == "P1:11-40"
            start, end = p.start_in_product, p.end_in_product
            assert d["residue_before"] == (PROTEIN.sequence[start - 2] if start > 1 else None) and d["residue_after"] == (PROTEIN.sequence[end] if end < len(PROTEIN.sequence) else None)
            for side in ("n_side_cut", "c_side_cut"):
                cut = d[side]
                assert cut == "protein terminus" or (cut["cleaved_by"] and cut["p1_residue"] and cut["p1_prime_residue"])
            if d["n_side_cut"] != "protein terminus":  # P1 is the residue before the fragment and P1' its first residue
                assert (d["n_side_cut"]["p1_residue"], d["n_side_cut"]["p1_prime_residue"]) == (d["residue_before"], c.sequence[0])
            assert (d["product_type"] == "FINAL_PRODUCT") == (not d["cuts_inside_after_the_pathway"]) == bool(p.is_limit_product)  # status agrees with the simulation
            seen_final |= d["product_type"] == "FINAL_PRODUCT"
            seen_intermediate |= d["product_type"] == "INTERMEDIATE_PRODUCT"
    assert seen_final and seen_intermediate


def test_a_recorded_origin_that_the_digestion_run_does_not_reproduce_is_a_warning_not_an_error(digested):
    c = next(c for c in digested if any(p.is_limit_product for p in c.parents)).model_copy(deep=True)
    c.parents[0].is_limit_product = not c.parents[0].is_limit_product
    warnings = explain_candidates([c], [PROTEIN])
    assert len(warnings) >= 1 and all(w["check"] == "digestion_consistency" and w["peptide_id"] == c.id for w in warnings)
    assert "recorded as" in warnings[0]["detail"]


def test_a_designed_peptide_is_explained_by_its_template_over_every_pathway_that_releases_it(digested):
    template = next(c for c in digested if len(c.parents) > 3)
    p = template.parents[0]
    design = FoodCandidate(id="PEP-DESIGN", sequence="A" * len(template.sequence), origin_type="SYNTHETIC_DESIGN", evidence_class="THEORETICAL",
                           parents=[ParentOrigin(protein_id=p.protein_id, start_in_product=p.start_in_product, end_in_product=p.end_in_product, pathway="not_the_template_pathway")],
                           predictions={"generation": {"template_sequence": template.sequence}})
    assert explain_candidates([design], [PROTEIN]) == []
    routes = design.predictions["generation"]["template_digestion"]
    assert {r["pathway"] for r in routes} >= {pp.pathway for pp in template.parents} and all(r["of"] == "template fragment" for r in routes)  # the design's own pathway is not what is shown
    assert design.parents[0].digestion_detail is None
    nowhere = design.model_copy(deep=True)
    nowhere.parents[0].start_in_product, nowhere.parents[0].end_in_product = 1, 3
    nowhere.predictions["generation"].pop("template_digestion")
    warnings = explain_candidates([nowhere], [PROTEIN])
    assert nowhere.predictions["generation"]["template_digestion"] == [] and "no pathway releases the template" in warnings[0]["detail"]


def test_the_cut_positions_of_the_detail_match_the_engines_cleavage_map(digested):
    _, _, cleavage_map = run([PROTEIN], DEFAULT_SETTINGS)
    explain_candidates(digested, [PROTEIN])
    sites = {(r["pathway"], r["stage_index"], r["precursor_p1_position"]) for r in cleavage_map}
    for c in digested:
        for p in c.parents:
            for cut in (p.digestion_detail["n_side_cut"], p.digestion_detail["c_side_cut"]):
                if cut != "protein terminus":
                    last = len(next(pw for pw in DEFAULT_SETTINGS.pathways if pw.id == p.pathway).stages)
                    assert (p.pathway, last, cut["position_in_precursor"]) in sites


# --- 2. evidence alignment ------------------------------------------------------------------------------------

def test_a_similarity_is_shown_as_an_alignment_with_both_lengths_identity_and_both_coverages_and_the_score_is_the_stored_one(lookup):
    found = lookup.lookup("AETYNPR")
    a = lookup.alignment("AETYNPR", found["evidence_peptide"])
    assert found["evidence_type"] == "SEQUENCE_SIMILARITY" and a["relationship"] == "similar"
    assert (a["candidate_length"], a["reference_length"]) == (7, len(found["evidence_peptide"]))
    assert a["identical_positions"] == a["midline"].count("|") and a["identity"] == round(a["identical_positions"] / a["aligned_columns"], 3) < 1
    assert a["candidate_coverage"] == round((a["candidate_end"] - a["candidate_start"] + 1) / 7, 3) and 0 < a["reference_coverage"] <= 1
    assert round(a["alignment_score"] / min(lookup._selfscore("AETYNPR"), lookup._selfscore(found["evidence_peptide"])), 3) == found["evidence_similarity"]  # display only: same score as the lookup


@pytest.mark.parametrize("seq,kind,relationship,candidate_cover,reference_cover", [("LKPT", "EXACT_SEQUENCE", "identical", 1.0, 1.0), ("LKPTPEGD", "CONTAINS_REFERENCE", "reference inside candidate", 0.5, 1.0)])
def test_exact_and_containment_matches_are_told_apart_from_similarity(lookup, seq, kind, relationship, candidate_cover, reference_cover):
    found = lookup.lookup(seq)
    a = lookup.alignment(seq, found["evidence_peptide"])
    assert found["evidence_type"] == kind and a["relationship"] == relationship and a["identity"] == 1.0
    assert (a["candidate_coverage"], a["reference_coverage"]) == (candidate_cover, reference_cover)


def test_no_alignment_without_an_evidence_peptide(lookup):
    assert lookup.alignment("AETYNPR", "") is None


def test_the_alignment_does_not_change_the_evidence_score_or_type(lookup):
    before = lookup.lookup("AETYNPR")
    lookup.alignment("AETYNPR", before["evidence_peptide"])
    assert lookup.lookup("AETYNPR") == before


# --- 5. liabilities with positions and masses -----------------------------------------------------------------

@pytest.mark.parametrize("seq", ["QMAWCDGPNGAAAAK", "AAAAK", "MMM", "DPDG", "QQQQ", "KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK"])
def test_the_liability_sites_name_the_same_liabilities_the_screening_counts(seq):
    names = [n for n in screening.physchem(seq)["liabilities"].split(";") if n]
    sites = liabilities.liability_sites(names, seq)
    assert [s["name"] for s in sites] == names
    for s in sites:
        if s["name"] not in liabilities.WHOLE_SEQUENCE:
            assert s["sites"], (seq, s)  # a residue-level liability always has a position


def test_liabilities_are_reported_with_residue_positions():
    sites = {s["name"]: s for s in liabilities.liability_sites(["methionine_oxidation", "tryptophan_oxidation", "asp_gly", "asp_pro", "n_term_gln", "homopolymer_run", "cysteine", "long_for_synthesis"],
                                                                    "QMAWCDGPDPAAAAK")}
    assert sites["methionine_oxidation"]["text"] == "M2: oxidation" and sites["tryptophan_oxidation"]["text"] == "W4: oxidation"
    assert sites["asp_gly"]["sites"] == ["D6"] and sites["asp_pro"]["sites"] == ["D9"] and sites["n_term_gln"]["sites"] == ["Q1"] and sites["cysteine"]["sites"] == ["C5"]
    assert sites["homopolymer_run"]["sites"] == ["A11-A14"] and sites["long_for_synthesis"]["text"].startswith("whole sequence")


def test_a_designed_peptide_gets_its_mass_and_the_terminal_state_stated():
    from food.shared.dossier import build_dossier

    row = {"id": "PEP-X", "sequence": "AETYNPR", "origin_type": "SYNTHETIC_DESIGN", "evidence_class": "THEORETICAL", "parents": [],
           "predictions": {"ranking": {"rank": 1, "rationale": "", "limitations": ""}, "screening": {"liabilities": "methionine_oxidation"}, "models": {}}}
    chemistry = build_dossier(row).chemistry
    assert chemistry["monoisotopic_mass_da"] == round(masses.mono_mass("AETYNPR"), 5) and chemistry["average_mass_da"] == masses.avg_mass("AETYNPR")
    assert "free N- and C-termini" in chemistry["terminal_modifications"] and chemistry["mass_source"].startswith("calculated for this report")
    assert chemistry["measured_chemical_stability"].startswith("Not measured") and "gut-protease" not in chemistry["protease_susceptibility"] and "cleavage" in chemistry["protease_susceptibility"]
    row["mono_mass"], row["avg_mass"] = 1.0, 2.0  # a mass the pipeline stored is used as saved
    assert build_dossier(row).chemistry["monoisotopic_mass_da"] == 1.0


def test_the_generation_stage_stores_the_mass_of_every_designed_peptide():
    from food.c_discovery.generation import stage as generation

    source = Path(generation.__file__).read_text(encoding="utf-8")
    assert "mono_mass=round(masses.mono_mass(" in source and "chemistry_notes=masses.chemistry_notes(" in source


# --- 4. the run summary ---------------------------------------------------------------------------------------

AUDIT = [{"kind": "candidate_stage", "stage": "s03_food_digestion", "n_in": 0, "n_out": 100, "rejection_reasons": {}},
         {"kind": "candidate_stage", "stage": "s05_food_screening", "n_in": 100, "n_out": 40, "rejection_reasons": {"too_short": 10, "not_in_model_shortlist": 50}},
         {"kind": "candidate_stage", "stage": "s07_food_ranking_decision", "n_in": 40, "n_out": 25, "rejection_reasons": {}}, {"kind": "failure", "stage": "x"}]
REPORT = {"gate": {"min_exact_evidence_requested": 15, "exact_evidence_in_final": 2, "shortfall": 13, "dropped_for_diversity": ["AAAA"], "dropped_for_evidence_gate": ["CCCC", "DDDD"]},
          "facts": {"objectives_used": ["coverage", "evidence"], "objectives_unavailable": ["model_aggregation_score"], "n_pareto_fronts": 4, "n_clusters": 9},
          "rules": {"min_length": 4, "max_length": 40, "max_per_cluster": 2, "max_per_parent_family": 3, "final_count": 25, "model_stage_pool": 400, "tie_break": ["evidence"]}}


def test_the_summary_reports_the_pool_each_filter_the_gate_the_caps_the_objectives_and_the_tie_break_from_the_runs_own_numbers():
    s = ranking_summary(REPORT, AUDIT, 8)
    assert [(x["stage"], x["n_in"], x["n_out"]) for x in s["stages"]] == [("s03_food_digestion", 0, 100), ("s05_food_screening", 100, 40), ("s07_food_ranking_decision", 40, 25)]
    assert s["ranked"] == 25 and s["shortlisted"] == 8 and s["pool_sent_to_the_models"] == 40
    assert s["hard_rules"]["removed"] == {"shorter than the minimum length": 10} and s["hard_rules"]["cut_to_the_model_pool"] == 50 and s["hard_rules"]["min_length"] == 4
    assert s["objectives"]["used"] == ["coverage", "evidence"] and s["objectives"]["not_used_because_a_pooled_peptide_lacks_them"] == ["model_aggregation_score"]
    assert s["pareto"]["weights"].startswith("None") and s["pareto"]["tie_break"] == ["evidence"] and s["pareto"]["n_fronts"] == 4
    assert (s["evidence_gate"]["exact_evidence_requested"], s["evidence_gate"]["shortfall"], s["evidence_gate"]["dropped_for_the_gate"]) == (15, 13, ["CCCC", "DDDD"])
    assert (s["caps"]["max_per_sequence_cluster"], s["caps"]["max_per_parent_family"], s["caps"]["dropped_for_the_caps"]) == (2, 3, ["AAAA"])


def test_a_run_that_did_not_record_something_gets_none_never_a_guess():
    s = ranking_summary({}, [], None)
    assert s["stages"] == [] and s["ranked"] is None and s["hard_rules"]["min_length"] is None and s["pareto"]["tie_break"] is None and s["caps"]["max_per_sequence_cluster"] is None
    assert s["objectives"]["used"] is None and s["evidence_gate"]["shortfall"] is None


def test_the_dossier_keeps_rank_pareto_front_and_shortlist_position_apart_and_lists_the_objective_values():
    from food.shared.dossier import build_dossier

    row = {"id": "PEP-X", "sequence": "LKPT", "origin_type": "DIGESTION_DERIVED", "evidence_class": "THEORETICAL", "parents": [{"protein_id": "P1:1-9"}], "parent_clusters": ["family:Whey"],
           "predictions": {"ranking": {"rank": 7, "pareto_front": 2, "n_fronts": 4, "shortlist_position": None, "shortlist_reason": "not eligible: its evidence is no evidence", "cluster": 5, "coverage": 0.75,
                                       "evidence": 0.9, "rationale": "", "limitations": ""}, "screening": {"developability_rule": 1.0}, "models": {}}}
    r = build_dossier(row, run_summary=ranking_summary(REPORT, AUDIT, 8)).ranking
    assert (r["computational_rank"], r["pareto_front"], r["n_pareto_fronts"], r["shortlist_position"], r["ranked_peptides"]) == (7, 2, 4, None, 25)
    assert r["objectives"] == [{"name": "coverage", "value": 0.75}, {"name": "evidence", "value": 0.9}]
    assert build_dossier(row).ranking["objectives"] is None  # a run that did not record its objectives is not guessed at


def test_the_pdf_has_a_page_on_how_the_list_was_made_only_when_the_run_recorded_it():
    from pypdf import PdfReader

    from food.shared.dossier import build_dossier
    from food.shared.dossier_pdf import dossier_pdf

    row = {"id": "PEP-X", "sequence": "LKPT", "origin_type": "DIGESTION_DERIVED", "evidence_class": "THEORETICAL", "parents": [{"protein_id": "P1:1-9"}],
           "predictions": {"ranking": {"rank": 7, "pareto_front": 2, "n_fronts": 4, "shortlist_position": 1, "rationale": "", "limitations": ""}, "screening": {}, "models": {}}}

    def text(dossier):
        return [p.extract_text() for p in PdfReader(io.BytesIO(dossier_pdf([dossier]))).pages]

    assert len(text(build_dossier(row))) == 3
    pages = text(build_dossier(row, run_summary=ranking_summary(REPORT, AUDIT, 8)))
    assert len(pages) == 4 and "How the list was made" in pages[2] and "Evidence gate" in pages[2] and "Cluster and parent-family caps" in pages[2] and "Weights: None" in pages[2].replace("\n", " ")
