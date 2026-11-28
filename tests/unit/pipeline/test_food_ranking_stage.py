"""s07_food_ranking_decision against Phase 1: the 400 pooled peptides are rebuilt from `screening_scored.csv` and must rank, derive and explain exactly as Phase 1 did."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from food.c_discovery import settings  # noqa: E402
from food.c_discovery.screening import checks  # noqa: E402
from food.shared.dossier_pdf import split_outside_brackets  # noqa: E402
from food.pipeline.stages.ranking import MODEL_OBJECTIVES, FoodRankingDecision  # noqa: E402
from food.shared.candidate import FoodCandidate, ParentOrigin  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "food" / "phase1"


def read(name):
    with open(FIX / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


POOL, RANKED, SHORTLIST = read("screening_scored.csv"), read("ranked_candidates.csv"), read("biology_shortlist.csv")
RUN = json.loads((FIX / "screening_run.json").read_text(encoding="utf-8"))

#: The stage A columns of Phase 1 (what s05 writes); nothing derived is passed in.
SCREEN = ["length", "origin_type", "net_charge_pH7_4", "isoelectric_point", "gravy", "hydrophobic_fraction", "aromaticity", "liabilities", "n_liabilities",
          "developability_rule", "evidence_match", "evidence_direction", "evidence_peptide", "evidence_similarity", "evidence_study", "evidence_assay",
          "evidence_food_derivation", "evidence_tier", "evidence_origin_type", "evidence", "nearest_positive", "nearest_positive_sim", "nearest_negative",
          "nearest_negative_sim", "evidence_grade", "evidence_basis", "digestion_support", "prior_art_alert", "parent_allergen_flag"]
NUMERIC = {"length": int, "developability_rule": float, "evidence": float, "digestion_support": float, "evidence_similarity": float}


def to_candidate(r: dict) -> FoodCandidate:
    screening = {k: (NUMERIC[k](r[k]) if k in NUMERIC else r[k]) for k in SCREEN}
    # Phase 1 called the value gut_survival; the stage now reads it as uncut_bond_fraction
    cleavage = {("uncut_bond_fraction" if k == "gut_survival" else k): (float(v) if v != "" else np.nan) for k, v in r.items()
                if k.startswith("cleave_") or k in ("gut_survival", "stability")}
    cleavage = {k: (int(v) if k.endswith("_sites") else v) for k, v in cleavage.items()}
    models = {n: {"score": float(r[f"model_{n}_score"]), "label": r[f"model_{n}_label"], "applicability": r[f"applicability_{n}"]}
              for n in MODEL_OBJECTIVES if r.get(f"model_{n}_score", "") != ""}
    clusters = [c for c in r["parent_clusters"].split(";") if c]
    # Phase 1 passed any match with a positive or mixed effect; the gate now passes only EXACT_SEQUENCE. To compare the unchanged Pareto selection like for like,
    # every Phase 1 match is fed in as EXACT_SEQUENCE, which gives the gate flag Phase 1 had.
    screening["evidence_type"] = "EXACT_SEQUENCE" if r["evidence_match"] != "none" else "NO_EVIDENCE"
    return FoodCandidate(id=r["peptide_id"], sequence=r["sequence"], origin_type=r["origin_type"], parent_clusters=clusters,
                         parents=[ParentOrigin(protein_id=s) for s in r["parent_substrates"].split(";")],
                         predictions={"screening": screening, "models": models, "cleavage": cleavage})


@pytest.fixture(scope="module")
def ranked():
    stage = FoodRankingDecision()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "SHORTLIST_SIZE", 8)  # Phase 1's shortlist had 8 peptides; the platform allows up to 10, so compare like for like
        out = stage.run([to_candidate(r) for r in POOL], StageConfig(), SimpleNamespace())
    return stage, out


def test_the_same_peptides_are_ranked_in_the_same_order(ranked):
    _, out = ranked
    assert [c.sequence for c in out] == [r["sequence"] for r in RANKED]
    assert [c.predictions["ranking"]["rank"] for c in out] == list(range(1, 26))


def test_objectives_gate_and_cluster_counts_match(ranked):
    stage, _ = ranked
    assert stage.facts["objectives_used"] == RUN["objectives_used"]
    assert stage.facts["n_pareto_fronts"] == RUN["n_pareto_fronts"] and stage.facts["n_clusters"] == RUN["n_clusters"]
    assert stage.gate["exact_evidence_in_final"] == RUN["selection"]["direct_in_final"] and stage.gate["shortfall"] == RUN["selection"]["shortfall"]


def test_derived_fields_match_phase_1(ranked):
    _, out = ranked
    text = ["developability_basis", "developability_flagged_models", "applicability_evidence",
            "applicability_stability", "applicability_overall", "applicability_assessed", "applicability_not_assessed",
            "glp1_classifier_status", "pareto_front", "cluster"]
    number = ["digestion_plausibility", "coverage", "developability"]
    for c, want in zip(out, RANKED):
        got = c.predictions["ranking"]
        assert got["evidence_gate_passed"] == (want["evidence_gate"] == "direct"), want["sequence"]
        # a model that did not run for this peptide is listed as not run (Phase 1 left it out silently when only this peptide lacked it, e.g. aggregation under 5 residues)
        assert got["predictions_not_run"] == ";".join(n for n in MODEL_OBJECTIVES if want.get(f"model_{n}_score", "") == ""), want["sequence"]
        # IgE binding is its own risk now: the safety score is the other safety model (cytotoxicity) alone, computed by the same rule as before
        cyt = [("cytotoxicity", float(want["model_cytotoxicity_score"]), want["applicability_cytotoxicity"])] if want.get("model_cytotoxicity_score", "") != "" else []
        value, basis, flagged = checks.combine_objective(cyt)
        assert (got["safety"] is None and value != value) or got["safety"] == pytest.approx(round(value, 3), abs=1e-9), want["sequence"]
        assert (got["safety_basis"], got["safety_flagged_models"]) == (basis, flagged), want["sequence"]
        assert got["model_allergenicity_score"] == pytest.approx(float(want["model_allergenicity_score"]), abs=1e-9)
        for k in text:
            assert str(got[k]) == want[k], (want["sequence"], k, got[k], want[k])
        for k in number:
            assert got[k] == pytest.approx(float(want[k]), abs=1e-9), (want["sequence"], k)


KEPT_BITS = ("Pareto front", "near pool-best", "weak on", "applicability:", "prior-art alert", "liabilities")


def kept(rationale: str) -> list[str]:
    """The parts of a rationale that did not change: the evidence wording and the old 'evidence gate:' prefix are left out."""
    bits = [b.split("evidence gate: ", 1)[-1].split("; ", 1)[-1] if b.startswith("evidence gate:") else b for b in split_outside_brackets(rationale)]
    return [b for b in bits if b.startswith(KEPT_BITS)]


def test_the_rationale_keeps_phase_1_wording_apart_from_the_evidence_sentences(ranked):
    _, out = ranked
    for c, want in zip(out, RANKED):
        assert kept(c.predictions["ranking"]["rationale"]) == kept(want["rationale"]), want["sequence"]
        assert "evidence gate" not in c.predictions["ranking"]["rationale"] and "direct" not in c.predictions["ranking"]["rationale"].lower()


def test_the_limitations_say_what_the_uncut_bond_fraction_and_the_evidence_are_and_are_not(ranked):
    _, out = ranked
    for c in out:
        text = c.predictions["ranking"]["limitations"]
        for must in ("the uncut-bond fraction comes from Expasy protease rules", "does not measure survival in the gut", "a model that did not run is never counted as a pass",
                     "a sequence relationship, not proof of GLP-1 activity", "no match means no data, not inactive"):
            assert must in text, must


def test_the_shortlist_keeps_its_properties_and_every_ranked_peptide_has_a_reason_and_the_controls_match_phase_1(ranked):
    stage, out = ranked  # the shortlist rule changed on purpose, so it no longer equals Phase 1's (see test_food_phase1_parity)
    shortlisted = [c for c in out if c.predictions["ranking"]["shortlist_position"] is not None]
    assert [c.predictions["ranking"]["shortlist_position"] for c in shortlisted] == list(range(1, len(shortlisted) + 1)) and len(shortlisted) == len(stage.shortlist) > 0
    assert len({c.parent_clusters[0] for c in shortlisted}) == len(shortlisted) and sum(c.origin_type == "SYNTHETIC_DESIGN" for c in shortlisted) <= settings.SHORTLIST_MAX_SYNTHETIC
    for c in out:
        r = c.predictions["ranking"]
        assert r["shortlist_reason"] and (r["shortlist_reason"].startswith("on the shortlist") == (r["shortlist_position"] is not None))
    assert {c["role"] for c in stage.controls} == {"positive_control", "negative_control"} and len(stage.controls) == 2 * settings.N_CONTROLS


def test_a_run_with_no_candidates_ranks_nothing():
    stage = FoodRankingDecision()
    assert stage.run([], StageConfig(), SimpleNamespace()) == [] and stage.shortlist == []


def test_the_glp1_classifier_is_reported_not_validated_and_never_ranked_on(ranked):
    stage, out = ranked
    assert {c.predictions["ranking"]["glp1_classifier_status"] for c in out} == {"Not validated"}
    assert not any("glp1" in o for o in stage.facts["objectives_used"])
