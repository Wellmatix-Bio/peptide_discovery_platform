"""Dossier and experiment sheet of a food run, for the biology review (spec gate G2: 5-10 dossiers, assay plan and controls).

Built only from what the run saved: the ranked candidates (`candidates/s07_food_ranking_decision.jsonl`), the reference controls (`ranking_report.json`) and the biology lead's saved
decisions (`review/decisions.jsonl`). Nothing is estimated or filled in here. Every value the biology lead has to choose (endpoint, concentrations, time, cell line, replicates,
vehicle, viability control) is left empty.

This module also holds the biological approval gate (the biology lead's decisions, kept apart from the ranking) and the run summary of how the ranking was made. The PDF is
`schemas/food_dossier_pdf.py`.
"""
from __future__ import annotations

import csv
import html
import io
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from common.food_digestion import masses
from common.food_liabilities import liability_sites
from common.protein_fasta import TERMINUS_MEANING


def terminus_text(code: str | None) -> str:
    """A protein end as `CODE (what it means)`; 'not recorded' when the input did not say."""
    return f"{code} ({TERMINUS_MEANING.get(code, 'unknown code')})" if code else "not recorded"

#: Status of a proposed shortlist entry (the wording Phase 1 uses in `biology_shortlist.csv`).
PROPOSED_STATUS = "proposed: biology lead must approve the shortlist, assay concentrations and controls"

#: Status once the biology lead has decided.
DECIDED_STATUS = {
    "approve": "approved by the biology lead; assay concentrations and controls are still set by the biology lead",
    "hold": "on hold by the biology lead: see the reason",
    "reject": "rejected by the biology lead: see the reason",
}

PRIMARY_ENDPOINT = "endogenous GLP-1 secretion (not GLP-1 receptor agonism, weight loss or glucose control)"

#: What the biology lead sets for each experiment (spec, Module E). Empty until they decide.
ASSAY_DESIGN_FIELDS = ["glp1_endpoint (active or total)", "concentration_range", "exposure_time", "cell_line", "vehicle", "viability_control",
                       "interference_control", "biological_replicates", "technical_replicates"]

REVIEW_DECISIONS = ("approve", "hold", "reject")

#: Always true of a candidate of this pipeline.
NOT_ESTABLISHED = [
    "GLP-1 secretion: no efficacy prediction is made. The GLP-1 classifier is not validated and is never used to rank.",
    "Safety: in-vitro and in-silico screens are not a safety clearance.",
    "Digestion: a cleavage site is a hypothesis. A fragment from the digestion rules is theoretical until LC-MS/MS observes it.",
    "Claims: an in-vitro GLP-1 result does not support consumer claims or systemic efficacy.",
]

#: How a candidate relates to an evidence peptide. These are sequence relationships, not proof of biological activity.
EVIDENCE_TYPE_MEANING = {
    "EXACT_SEQUENCE": "The sequence is the same as the evidence peptide.",
    "CONTAINS_REFERENCE": "The sequence contains the whole evidence peptide (4 residues or more).",
    "CONTAINED_IN_REFERENCE": "The sequence is a subsequence of the evidence peptide.",
    "SEQUENCE_SIMILARITY": "The alignment score reaches the similarity cut-off, with no exact or containment match.",
    "NO_EVIDENCE": "No evidence peptide qualifies. This means no data, not inactive.",
}
EVIDENCE_NOTE = "A sequence relationship, not proof of GLP-1 activity."
EVIDENCE_GATE_RULE = "The evidence gate passes only with an EXACT_SEQUENCE match to an evidence peptide whose reported effect is positive or mixed."

#: What `uncut_bond_fraction` is and is not; shown wherever the value is shown.
UNCUT_BOND_FRACTION_STATEMENT = ("uncut_bond_fraction is the share of the peptide's internal bonds that the rules for pepsin, trypsin and chymotrypsin do not cut. "
                                 "It is a rule-based prediction. It does not measure survival in the gastrointestinal tract.")

#: The sequence liabilities the screening stage can report, in plain words (from the comments of `screening.physchem`).
LIABILITY_TEXT = {
    "n_term_gln": "N-terminal glutamine (can form pyroglutamate)",
    "asp_gly": "Asp or Asn followed by Gly (deamidation or isoaspartate)",
    "asp_pro": "Asp-Pro bond (acid-labile)",
    "homopolymer_run": "a run of four or more identical residues",
    "cysteine": "cysteine present",
    "methionine_oxidation": "methionine present (oxidation risk)",
    "tryptophan_oxidation": "tryptophan present (oxidation risk)",
    "hydrophobic_solubility_risk": "high share of hydrophobic residues (solubility risk)",
    "long_for_synthesis": "long for synthesis",
}
OXIDATION_LIABILITIES = ("methionine_oxidation", "tryptophan_oxidation")

#: What each screening score is, read from the adapters (`pipeline/food/screening/adapters.py`) and the models' detail texts. Every score is built so that a higher value is the favourable
#: outcome of that model; `inverse` means score = 1 - p, `direct` means score = p, where p is the probability in the model's detail text. Checked on real data in tests/unit/test_food_dossier.py.
MODEL_SCORE_MEANING = {
    "allergenicity": ("inverse", "a ranking value only: 1 minus the IgE-binding score (the model's own output, reported as ige_binding_score)"),
    "cytotoxicity": ("inverse", "1 minus the predicted probability that the peptide is cytotoxic"),
    "solubility": ("direct", "the predicted probability that the peptide is soluble in water"),
    "aggregation": ("inverse", "1 minus the predicted probability that the peptide is aggregation-prone"),
    "synthesis_feasibility": ("direct", "the predicted probability that synthesis is feasible"),
    "bitterness": ("inverse", "1 minus the predicted probability that the peptide is bitter"),
}
#: What each model predicts, and what a higher value of its own output means. The reports show this output, never 1 minus it. `orientation` links it to the ranking value `score`
#: (`inverse`: score = 1 - output, `direct`: score = output); checked on real data in tests/unit/test_food_dossier.py.
MODEL_OUTPUT_MEANING = {
    "allergenicity": ("inverse", "IgE binding", "more likely to bind IgE"),
    "cytotoxicity": ("inverse", "cytotoxicity", "more likely to be cytotoxic"),
    "solubility": ("direct", "solubility in water", "more likely to be soluble in water"),
    "aggregation": ("inverse", "aggregation", "more likely to be aggregation-prone"),
    "synthesis_feasibility": ("direct", "synthesis feasibility", "more likely that synthesis is feasible"),
    "bitterness": ("inverse", "bitterness", "more likely to be bitter"),
}


def model_outputs(models: dict) -> dict:
    """Per model: the model's own output (as saved, or recovered from the ranking value for a run saved before the output was kept), what it predicts and what higher means."""
    out = {}
    for name, m in models.items():
        if name not in MODEL_OUTPUT_MEANING or m.get("score") is None:
            continue
        orientation, predicts, higher = MODEL_OUTPUT_MEANING[name]
        value = m.get("output")
        if value is None:
            value = 1 - m["score"] if orientation == "inverse" else m["score"]
        out[name] = {"predicts": predicts, "output": float(value), "higher_means": higher, "label": m.get("label"), "applicability": m.get("applicability")}
    return out


DEVELOPABILITY_MEANING = ("the lowest of the solubility, aggregation, synthesis-feasibility and bitterness scores that count (a model that did not run, or is outside its range, is left out) "
                          "and the sequence-liability score")

UNAVAILABLE_RULE = "A model that did not run is reported as not run. It is never counted as a pass."


class FoodDossier(BaseModel):
    peptide_id: str
    sequence: str
    length: int
    rank: int
    shortlist_position: int | None = None
    status: str = PROPOSED_STATUS
    origin_type: str
    evidence_class: str
    material_class: str = "peptide"
    summary: str = Field(description="Why the peptide ranks where it does.")
    properties: dict = Field(default_factory=dict)
    digestion: dict = Field(default_factory=dict, description="Whether it is predicted or observed, and whether it is a final or an intermediate digestion product.")
    provenance: dict = Field(default_factory=dict, description="Parents, digestion path, clusters, and the template and guide of a designed peptide.")
    evidence: dict = Field(default_factory=dict, description="The evidence for the candidate and, for a designed peptide, for its template, each with its type and source; the evidence gate.")
    safety_screening: dict = Field(default_factory=dict, description="Predicted IgE binding as its own risk, the other safety models, sequence liabilities, missing predictions and applicability limits.")
    screening: dict = Field(default_factory=dict, description="Per model: score, label, applicability, detail; the safety and developability objectives and their basis.")
    gut_stability: dict = Field(default_factory=dict)
    ranking: dict = Field(default_factory=dict, description="Computational rank, Pareto front and shortlist position kept apart, with the peptide's value on each objective the run used.")
    ranking_summary: dict = Field(default_factory=dict, description="How the run narrowed its candidates (the same for every peptide of a run): see the results route.")
    chemistry: dict = Field(default_factory=dict, description="Mass and terminal state, the predicted liabilities with their residue positions, and what is not measured. Predicted liabilities, measured chemical stability and protease susceptibility are separate.")
    comparison_materials: dict = Field(default_factory=dict, description="Parent proteins, the hydrolysates they define, and reference peptides to compare with.")
    limitations: str = ""
    not_established: list[str] = Field(default_factory=lambda: list(NOT_ESTABLISHED))
    controls: list[dict] = Field(default_factory=list, description="Reference positive and negative controls proposed by rule.")
    assay_plan: dict = Field(default_factory=dict)
    review: dict = Field(default_factory=dict, description="The biology lead's decision and reason; empty until they decide.")


def _pick(source: dict, keys: list[str]) -> dict:
    return {k: source[k] for k in keys if source.get(k) not in (None, "")}


def pepsin_reachable_bonds(length: int) -> int:
    """Bonds of a peptide the pepsin rule can cut at all: it reads two residues on each side of a bond, so the two bonds at each end are out of reach.
    Checked against the rule itself in tests/unit/pipeline/test_food_dossier_rules.py."""
    return max(length - 4, 0)


def _evidence_block(source: dict, prefix: str, applies_to: str) -> dict | None:
    """One evidence record: its type, what it applies to and where it comes from. `prefix` is "" for the candidate and "template_" for a designed peptide's template."""
    kind = source.get(f"{prefix}evidence_type")
    if kind is None:
        return None
    found = kind != "NO_EVIDENCE"
    def get(key):
        return (source.get(f"{prefix}{key}") or None) if found else None
    return {"applies_to": applies_to, "evidence_type": kind, "meaning": EVIDENCE_TYPE_MEANING.get(kind, ""), "note": EVIDENCE_NOTE,
            "evidence_peptide": get("evidence_peptide"), "reported_effect": get("evidence_direction"), "similarity": source.get(f"{prefix}evidence_similarity"),
            "source": {"study": get("evidence_study"), "assay": get("evidence_assay"), "evidence_tier": get("evidence_tier"), "food_derivation": get("evidence_food_derivation")},
            "alignment": source.get(f"{prefix}evidence_alignment") if found else None}


def _pathway_stages(pathway_id: str) -> list[dict]:
    """The stages (stage name, enzymes, pH) of a digestion pathway, from the fixed digestion settings; empty for an unknown pathway."""
    from common.food_digestion.settings import DEFAULT_SETTINGS

    for pathway in DEFAULT_SETTINGS.pathways:
        if pathway.id == pathway_id:
            return [{"stage": stage.name, "enzymes": list(stage.enzymes), "ph": stage.ph} for stage in pathway.stages]
    return []


def _comparison_materials(candidate: dict, screening: dict, generation: dict, controls: list[dict]) -> dict:
    """What to compare the candidate with, all identified from the run: the parent proteins, the hydrolysates those proteins and the pathways of the run define, and reference peptides."""
    parents = candidate.get("parents") or []
    proteins: dict[str, dict] = {}
    hydrolysates: dict[tuple, dict] = {}
    for p in parents:
        proteins.setdefault(p.get("protein_id"), {"protein_id": p.get("protein_id"), "accession": p.get("accession"), "food_source": p.get("food_source"),
                                                  "allergen_source": p.get("allergen_source"), "protein_n_terminus": p.get("protein_n_terminus"), "protein_c_terminus": p.get("protein_c_terminus"), "residues_in_uniprot_entry": f"{p.get('start_in_precursor')}-{p.get('end_in_precursor')}"})
        if p.get("pathway"):
            hydrolysates.setdefault((p.get("protein_id"), p["pathway"]), {"parent_protein": p.get("protein_id"), "food_source": p.get("food_source"), "pathway": p["pathway"],
                                                                         "stages": _pathway_stages(p["pathway"])})
    references: list[dict] = []
    if generation.get("template_sequence"):
        references.append({"role": "template fragment", "sequence": generation["template_sequence"], "source": "the digestion fragment this design was edited from"})
    if generation.get("guide_peptide"):
        references.append({"role": "reference peptide the design was steered toward", "sequence": generation["guide_peptide"], "source": generation.get("guide_study") or None})
    if screening.get("evidence_peptide") and screening.get("evidence_type") != "NO_EVIDENCE":
        references.append({"role": f"closest evidence peptide ({screening.get('evidence_type')})", "sequence": screening["evidence_peptide"], "source": screening.get("evidence_study") or None})
    for label, key, sim_key in (("nearest positive peptide", "nearest_positive", "nearest_positive_sim"), ("nearest negative peptide", "nearest_negative", "nearest_negative_sim")):
        if screening.get(key):
            references.append({"role": label, "sequence": screening[key], "similarity": screening.get(sim_key)})
    references += [{"role": c.get("role"), "sequence": c.get("sequence"), "source": c.get("study") or None} for c in controls]
    return {"parent_proteins": list(proteins.values()), "hydrolysates": list(hydrolysates.values()), "reference_peptides": references,
            "note": "Hydrolysates are defined by the parent protein and the digestion pathway of this run. They are materials to prepare, not existing records."}


def _safety_screening(models: dict, screening: dict, ranking: dict, sequence: str = "") -> dict:
    """IgE binding is reported as its own risk; the other safety models, the sequence liabilities, what did not run and each model's applicability sit beside it."""
    ige = models.get("allergenicity")
    liabilities = [n for n in (screening.get("liabilities") or "").split(";") if n]
    return {
        "ige_binding": ({"status": "RUN", "label": ige.get("label"), "ige_binding_score": model_outputs({"allergenicity": ige})["allergenicity"]["output"],
                         "ranking_score": ige.get("score"), "applicability": ige.get("applicability"), "detail": ige.get("detail"),
                         "note": ("The model's own IgE-binding score (0 to 1): higher means more likely to bind IgE, and 0.5 is the model's decision boundary. It is a model score, not a measured "
                                  "probability, and not a food-safety clearance. ranking_score (1 minus the score) is only the value the ranking maximises.")}
                        if ige else {"status": "NOT_RUN", "note": "The IgE-binding model did not run for this peptide; its absence is not a pass."}),
        "parent_allergen_flag": screening.get("parent_allergen_flag"),
        "other_safety_models": {n: v for n, v in model_outputs(models).items() if n == "cytotoxicity"},
        "safety_score": ranking.get("safety"), "safety_score_note": "A ranking value from the safety models other than IgE binding (higher is the favourable outcome). null means no such model could be counted.",
        "sequence_liabilities": [{"name": n, "meaning": LIABILITY_TEXT.get(n, n.replace("_", " ")), "sites": s["sites"], "text": s["text"]} for n, s in zip(liabilities, liability_sites(liabilities, sequence))],
        "oxidation_risks": [{"name": n, "meaning": LIABILITY_TEXT[n], "sites": s["sites"], "text": s["text"]} for n, s in zip(liabilities, liability_sites(liabilities, sequence)) if n in OXIDATION_LIABILITIES],
        "predictions_not_run": [n for n in (ranking.get("predictions_not_run") or "").split(";") if n],
        "models_left_out_of_the_safety_score": [n for n in (ranking.get("safety_flagged_models") or "").split(";") if n],
        "applicability_limits": {k.removeprefix("applicability_"): v for k, v in ranking.items() if k.startswith("applicability_") and k not in ("applicability_assessed", "applicability_not_assessed")},
        "rule": UNAVAILABLE_RULE,
    }


def _chemistry(candidate: dict, screening: dict) -> dict:
    """Mass, terminal state and the predicted liabilities with their residue positions. A peptide saved without a mass (a designed peptide from an earlier run) gets it computed here from the
    sequence. Measured chemical stability and protease susceptibility are not part of the liabilities: the first is not measured, the second is the gut-protease section."""
    seq = candidate["sequence"]
    stored = candidate.get("mono_mass") is not None
    try:
        mono, avg = (candidate["mono_mass"], candidate["avg_mass"]) if stored else (round(masses.mono_mass(seq), 5), masses.avg_mass(seq))
    except KeyError:
        mono = avg = None
    names = [n for n in (screening.get("liabilities") or "").split(";") if n]
    return {"monoisotopic_mass_da": mono, "average_mass_da": avg,
            "mass_source": "calculated in the pipeline" if stored else "calculated for this report from the sequence" if mono is not None else "not available",
            "terminal_modifications": "none: free N- and C-termini, linear peptide (the mass assumes this)",
            "chemistry_notes": candidate.get("chemistry_notes") or masses.chemistry_notes(seq),
            "predicted_liabilities": liability_sites(names, seq),
            "predicted_liabilities_meaning": "Predicted from the sequence by pattern rules; none is measured.",
            "measured_chemical_stability": "Not measured: no stability data enter this pipeline.",
            "protease_susceptibility": "Predicted separately: see the cleavage by gut proteases."}


def build_dossier(candidate: dict, controls: list[dict] | None = None, review: dict | None = None, run_summary: dict | None = None) -> FoodDossier:
    """`candidate` is one row of the ranked-candidates file; `review` is the biology lead's current decision on it, if any."""
    controls = controls or []
    preds = candidate.get("predictions") or {}
    screening, ranking, cleavage, generation = preds.get("screening") or {}, preds.get("ranking") or {}, preds.get("cleavage") or {}, preds.get("generation") or {}
    models = preds.get("models") or {}
    both = {**screening, **ranking}
    length = len(candidate["sequence"])
    provenance = {"parents": candidate.get("parents") or [], "parent_clusters": candidate.get("parent_clusters") or [],
                  "any_parent_allergen_source": candidate.get("any_parent_allergen_source"), "generation_method": candidate.get("generation_method") or None}
    if generation:
        provenance["design"] = _pick(generation, ["template_sequence", "template_peptide_id", "template_origin_type", "edits", "n_edits", "edit_kinds", "guide_peptide",
                                                  "guide_study", "guide_food_derivation", "sim_to_guide", "template_sim_to_guide", "seed", "generator_version", "evidence_db_md5"])
    designed = candidate["origin_type"] == "SYNTHETIC_DESIGN"
    kind = candidate.get("digestion_product_type")
    observed = candidate["evidence_class"] == "OBSERVED"
    digestion = {"observation_status": "OBSERVED" if observed else "PREDICTED", "lcms_evidence": candidate.get("lcms_evidence") or [],
                 "statement": "Observed by LC-MS/MS." if observed else "Predicted by digestion rules. Not observed by LC-MS/MS.",
                 "digestion_product_type": kind,
                 "product_type_meaning": ("Not a digestion product: it was designed." if designed else
                                          "Final product: no enzyme of the pathway can cut inside it any more." if kind == "FINAL_PRODUCT" else
                                          "Intermediate product: an enzyme of the pathway could still cut inside it." if kind == "INTERMEDIATE_PRODUCT" else "Not recorded.")}
    evidence = {"candidate": _evidence_block(both, "", "CANDIDATE"), "template": _evidence_block(both, "template_", "TEMPLATE") if designed else None,
                "gate": {"passed": ranking.get("evidence_gate_passed"), "rule": EVIDENCE_GATE_RULE},
                "nearest_positive": _pick({"sequence": screening.get("nearest_positive"), "similarity": screening.get("nearest_positive_sim")}, ["sequence", "similarity"]) or None,
                "nearest_negative": _pick({"sequence": screening.get("nearest_negative"), "similarity": screening.get("nearest_negative_sim")}, ["sequence", "similarity"]) or None,
                "prior_art_alert": screening.get("prior_art_alert") or None, "digestion_support": screening.get("digestion_support"),
                "designed_peptide_rule": ("The designed sequence is looked up as itself. Its template's evidence is shown, never transferred to it." if designed else None)}
    routes = [p["digestion_detail"] for p in candidate.get("parents") or [] if p.get("digestion_detail")] if not designed else generation.get("template_digestion") or []
    digestion["routes_of"] = "the template fragment (a designed peptide has no digestion of its own)" if designed else "the peptide"
    seen: set = set()
    routes = [r for r in routes if (key := (r.get("protein_id"), r.get("start_in_precursor"), r.get("pathway"))) not in seen and not seen.add(key)]  # one row per protein site and route, not per stage
    digestion["routes"] = routes
    digestion["routes_meaning"] = ("Each route is the digestion run again for this fragment: the stages with their enzymes and pH, the cut before and after it (the residue before the cut, P1, and "
                                   "the residue after it, P1', with P1's position in the UniProt entry and the enzymes that cut there), the cuts still inside it, and final or intermediate "
                                   "(final: no enzyme of the route can cut inside it). Predicted by rules, not observed by LC-MS/MS.") if routes else "Not recorded for this run."
    objectives = ((run_summary or {}).get("objectives") or {}).get("used")
    run_ranking = {"computational_rank": ranking["rank"], "ranked_peptides": (run_summary or {}).get("ranked"), "pareto_front": ranking.get("pareto_front"),
                   "n_pareto_fronts": ranking.get("n_fronts"), "dominates_n_at_pick": ranking.get("dominates_n_at_pick") or None,
                   "shortlist_position": ranking.get("shortlist_position"), "shortlist_reason": ranking.get("shortlist_reason"),
                   "sequence_cluster": ranking.get("cluster"), "parent_families": candidate.get("parent_clusters") or [],
                   "objectives": [{"name": o, "value": ranking.get(o, screening.get(o))} for o in objectives] if objectives else None,
                   "meaning": "Rank is the order in which the Pareto selection took the peptide. Pareto front 1 means no peptide of the pool beats it on every objective. The shortlist position is "
                              "separate: the shortlist is taken from the ranked list by its own rule. Every objective is 0 to 1, higher is the favourable outcome."}
    gut = {"uncut_bond_fraction": cleavage.get("uncut_bond_fraction"), "statement": cleavage.get("statement") or UNCUT_BOND_FRACTION_STATEMENT,
           "cut_sites_per_protease": {k.removeprefix("cleave_").removesuffix("_sites"): v for k, v in cleavage.items() if k.startswith("cleave_") and k.endswith("_sites")},
           "predicted_cleavage_positions": cleavage.get("cleavage_positions") or {},
           "positions_meaning": "Position p means the bond between residue p and residue p + 1 is cut by that protease's rule.",
           "rules": cleavage.get("rules"), "method": cleavage.get("method"),
           "pepsin_reachable_bonds": f"{pepsin_reachable_bonds(length)} of {max(length - 1, 0)} internal bonds",
           "note": "The pepsin rule cannot cut the two bonds nearest each end, so a short peptide's uncut-bond fraction is overstated for pepsin."}
    review = review or {}
    decision = review.get("decision")
    return FoodDossier(
        peptide_id=candidate["id"], sequence=candidate["sequence"], length=length, rank=ranking["rank"], shortlist_position=ranking.get("shortlist_position"),
        status=DECIDED_STATUS.get(decision, PROPOSED_STATUS),
        origin_type=candidate["origin_type"], evidence_class=candidate["evidence_class"], material_class=candidate.get("material_class") or "peptide",
        summary=ranking.get("rationale", ""), limitations=ranking.get("limitations", ""),
        properties={**_pick(screening, ["length", "net_charge_pH7_4", "isoelectric_point", "gravy", "hydrophobic_fraction", "aromaticity", "liabilities", "n_liabilities"]),
                    **_pick(candidate, ["mono_mass", "avg_mass", "chemistry_notes"])},
        digestion=digestion, provenance=provenance, evidence=evidence,
        safety_screening=_safety_screening(models, screening, ranking, candidate["sequence"]), chemistry=_chemistry(candidate, screening), ranking=run_ranking, ranking_summary=run_summary or {},
        screening={"models": models, "model_outputs": model_outputs(models), "score_meaning": {n: MODEL_SCORE_MEANING[n][1] for n in models if n in MODEL_SCORE_MEANING},
                   "developability_meaning": DEVELOPABILITY_MEANING, "predictions_not_run": [n for n in (ranking.get("predictions_not_run") or "").split(";") if n],
                   **_pick(ranking, ["safety", "safety_basis", "developability", "developability_basis", "developability_flagged_models",
                                     "applicability_overall", "applicability_assessed", "applicability_not_assessed", "coverage", "pareto_front", "cluster", "glp1_classifier_status"]),
                   "applicability": {k.removeprefix("applicability_"): v for k, v in ranking.items() if k.startswith("applicability_") and k != "applicability_assessed"
                                     and k != "applicability_not_assessed" and k != "applicability_overall"}},
        gut_stability=gut, comparison_materials=_comparison_materials(candidate, screening, generation, controls), controls=controls,
        assay_plan={"primary_endpoint": PRIMARY_ENDPOINT, "material_type": "purified peptide (not intact protein or hydrolysate)",
                    "set_by_biology_lead": {name: None for name in ASSAY_DESIGN_FIELDS}},
        review={"decision": decision, "reason": review.get("reason") or None, "reviewer": review.get("reviewer"), "decided_at": review.get("decided_at"),
                "allowed_decisions": list(REVIEW_DECISIONS)},
    )


SHEET_COLUMNS = ["row_type", "shortlist_position", "rank", "peptide_id", "sequence", "length", "origin_type", "parent_proteins", "food_source", "parent_allergen_flag", "parent_n_terminus", "parent_c_terminus",
                 "predicted_ige_binding", "oxidation_risks", "evidence_type", "evidence_gate_passed", "evidence_peptide", "evidence_study", "template_evidence_type",
                 "applicability_overall", "predictions_not_run", "safety_flagged_models", "prior_art_alert",
                 *[f.split(" (")[0] for f in ASSAY_DESIGN_FIELDS], "status", "decision", "reviewer", "comments"]


def experiment_sheet(candidates: list[dict], controls: list[dict], reviews: dict[str, dict] | None = None) -> list[dict]:
    """One row per shortlisted peptide, in shortlist order, then one per reference control. The assay-design cells stay empty; decision, reviewer and comments (the reason) are filled
    only from the biology lead's saved decisions."""
    blank = {c: "" for c in SHEET_COLUMNS}
    reviews = reviews or {}
    shortlisted = sorted((c for c in candidates if (c.get("predictions") or {}).get("ranking", {}).get("shortlist_position") is not None),
                         key=lambda c: c["predictions"]["ranking"]["shortlist_position"])
    rows = []
    for c in shortlisted:
        ranking, screening = c["predictions"]["ranking"], c["predictions"].get("screening") or {}
        models = c["predictions"].get("models") or {}
        parents = c.get("parents") or []
        review = reviews.get(c["id"], {})
        liabilities = [n for n in (screening.get("liabilities") or "").split(";") if n in OXIDATION_LIABILITIES]
        rows.append({**blank, "row_type": "candidate", "shortlist_position": ranking["shortlist_position"], "rank": ranking["rank"], "peptide_id": c["id"], "sequence": c["sequence"],
                     "length": len(c["sequence"]), "origin_type": c["origin_type"], "parent_proteins": ";".join(dict.fromkeys(p["protein_id"] for p in parents)),
                     "food_source": ";".join(dict.fromkeys(p["food_source"] for p in parents if p.get("food_source"))),
                     "parent_allergen_flag": screening.get("parent_allergen_flag", ""),
                     "parent_n_terminus": ";".join(dict.fromkeys(f"{p['protein_id']}:{p.get('protein_n_terminus') or 'not recorded'}" for p in parents)),
                     "parent_c_terminus": ";".join(dict.fromkeys(f"{p['protein_id']}:{p.get('protein_c_terminus') or 'not recorded'}" for p in parents)),
                     "predicted_ige_binding": (models.get("allergenicity") or {}).get("label", "not run"), "oxidation_risks": ";".join(liabilities),
                     "evidence_type": ranking.get("evidence_type") or screening.get("evidence_type") or "",
                     "evidence_gate_passed": "yes" if ranking.get("evidence_gate_passed") else "no",
                     "evidence_peptide": ranking.get("evidence_peptide") or screening.get("evidence_peptide") or "", "evidence_study": screening.get("evidence_study", ""),
                     "template_evidence_type": screening.get("template_evidence_type", ""),
                     "applicability_overall": ranking.get("applicability_overall", ""), "predictions_not_run": ranking.get("predictions_not_run", ""),
                     "safety_flagged_models": ranking.get("safety_flagged_models", ""), "prior_art_alert": screening.get("prior_art_alert", ""),
                     "status": DECIDED_STATUS.get(review.get("decision"), PROPOSED_STATUS), "decision": review.get("decision") or "", "reviewer": review.get("reviewer") or "",
                     "comments": review.get("reason") or ""})
    for control in controls:
        rows.append({**blank, "row_type": control["role"], "sequence": control["sequence"], "length": len(control["sequence"]), "origin_type": control.get("origin_type", ""),
                     "food_source": control.get("food_derivation", ""), "evidence_study": control.get("study", ""), "evidence_type": f"{control.get('n_observations', '')} observations",
                     "status": control.get("note", "")})
    return rows


def experiment_sheet_csv(candidates: list[dict], controls: list[dict], reviews: dict[str, dict] | None = None) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=SHEET_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(experiment_sheet(candidates, controls, reviews))
    return out.getvalue()


def _show(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_show(v)}" for k, v in value.items() if v not in (None, "", [], {}))
    if isinstance(value, list):
        return ", ".join(_show(v) for v in value)
    return str(value)


def _table(rows: list[tuple[str, Any]]) -> str:
    return "<table>" + "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(_show(v))}</td></tr>" for k, v in rows if v not in (None, "", [], {})) + "</table>"


def _cut_text(c) -> str:
    return "protein terminus" if isinstance(c, str) else c["p1_residue"] + str(c["position_in_precursor"]) + " | " + c["p1_prime_residue"] + " (" + ", ".join(c["cleaved_by"]) + ")"


def _stage_text(stages: list[dict]) -> str:
    return "; ".join(s["stage"] + ": " + "+".join(s["enzymes"]) + ", pH " + str(s["ph"]) for s in stages)


def _inside_text(cuts: list[dict]) -> str:
    return "; ".join("after residue " + str(x["position_in_peptide"]) + ": " + x["p1_residue"] + "|" + x["p1_prime_residue"] for x in cuts) or "none"


def _routes_html(routes: list[dict]) -> str:
    e = html.escape
    rows = "".join("<tr>" + "".join("<td>" + e(str(v)) + "</td>" for v in (r["pathway"], _stage_text(r["stages"]), _cut_text(r["n_side_cut"]), _cut_text(r["c_side_cut"]),
                                                                           _inside_text(r["cuts_inside_after_the_pathway"]), r["product_type"])) + "</tr>" for r in routes)
    return "<table><tr><th>Route</th><th>Stages</th><th>Cut before it</th><th>Cut after it</th><th>Cuts inside</th><th>Product</th></tr>" + rows + "</table>" if routes else "<p>Not recorded.</p>"


def _alignment_html(ev: dict) -> str:
    e = html.escape
    out = []
    for label, block in (("Candidate", ev.get("candidate")), ("Template (shown, not transferred)", ev.get("template"))):
        a = (block or {}).get("alignment")
        if a:
            identity = str(a["identical_positions"]) + " of " + str(a["aligned_columns"]) + " aligned positions (" + str(a["identity"]) + ")"
            out.append("<h3>" + e(label) + ": " + e(a["relationship"]) + "</h3><pre>" + "\n".join(e(a[k]) for k in ("aligned_candidate", "midline", "aligned_reference")) + "</pre>"
                       + _table([("Candidate length", a["candidate_length"]), ("Reference length", a["reference_length"]), ("Identity", identity), ("Candidate coverage", a["candidate_coverage"]),
                                 ("Reference coverage", a["reference_coverage"]), ("Aligned candidate residues", str(a["candidate_start"]) + "-" + str(a["candidate_end"])),
                                 ("Aligned reference residues", str(a["reference_start"]) + "-" + str(a["reference_end"])), ("Alignment score", a["alignment_score"])])
                       + '<p class="small">A sequence similarity is not experimentally confirmed GLP-1 activity.</p>')
    return "".join(out) or "<p>No alignment: no evidence peptide qualifies.</p>"


def dossier_html(d: FoodDossier) -> str:
    """A readable page for the reviewer. Every value is escaped."""
    e = html.escape
    prov, ev, safety, gut, cmp_ = d.provenance, d.evidence, d.safety_screening, d.gut_stability, d.comparison_materials
    parents = "".join(f"<tr><td>{e(str(p.get('protein_id')))}</td><td>{e(str(p.get('food_source') or ''))}</td><td>{e(str(p.get('allergen_source')))}</td><td>{e(terminus_text(p.get('protein_n_terminus')))}</td><td>{e(terminus_text(p.get('protein_c_terminus')))}</td>"
                      f"<td>{e(str(p.get('start_in_product')))}-{e(str(p.get('end_in_product')))}</td><td>{e(str(p.get('pathway') or ''))}</td>"
                      f"<td>{e(', '.join(p.get('enzymes') or []) if isinstance(p.get('enzymes'), list) else str(p.get('enzymes') or ''))}</td><td>{e(str(p.get('ph')))}</td>"
                      f"<td>{e(str(p.get('digestion_product_type') or ''))}</td></tr>" for p in prov.get("parents", []))
    models = "".join(f"<tr><td>{e(n)}</td><td>{e(str(m.get('score')))}</td><td>{e(str(m.get('label')))}</td><td>{e(str(m.get('applicability')))}</td><td>{e(str(m.get('detail') or ''))}</td></tr>"
                     for n, m in d.screening.get("models", {}).items())
    controls = "".join(f"<tr><td>{e(c['role'])}</td><td>{e(c['sequence'])}</td><td>{e(str(c.get('study')))}</td><td>{e(str(c.get('assay')))}</td><td>{e(str(c.get('n_observations')))}</td></tr>" for c in d.controls)
    plan = "".join(f"<tr><th>{e(k)}</th><td>to be set by the biology lead</td></tr>" for k in d.assay_plan.get("set_by_biology_lead", {}))
    refs = "".join(f"<tr><td>{e(str(r.get('role')))}</td><td>{e(str(r.get('sequence')))}</td><td>{e(str(r.get('source') or r.get('similarity') or ''))}</td></tr>" for r in cmp_.get("reference_peptides", []))
    review = d.review
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Dossier {e(d.sequence)}</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#182638}}table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.2rem}}
th,td{{border:1px solid #dce4ee;padding:6px 9px;text-align:left;vertical-align:top}}th{{background:#f2f6fa;white-space:nowrap}}.banner{{background:#fff3df;border-left:4px solid #a35a00;padding:.6rem 1rem}}
.small{{color:#627389;font-size:12px}}</style></head><body>
<h1>{e(d.sequence)} <span class="small">{e(d.peptide_id)}</span></h1>
<p class="banner">{e(d.status)}. Shortlist position {e(str(d.shortlist_position or 'none'))}, rank {e(str(d.rank))}. Origin {e(d.origin_type)}, {e(d.evidence_class.lower())}.</p>
<h2>Why it ranks here</h2><p>{e(d.summary)}</p>
<h2>Properties</h2>{_table(list(d.properties.items()))}
<h2>Digestion</h2>{_table([("Status", d.digestion.get("statement")), ("Product type", d.digestion.get("digestion_product_type")), ("Meaning", d.digestion.get("product_type_meaning"))])}
<h2>Where it comes from</h2><table><tr><th>Protein</th><th>Food source</th><th>Allergen source</th><th>Protein N-terminus</th><th>Protein C-terminus</th><th>Position</th><th>Pathway</th><th>Enzymes</th><th>pH</th><th>Product type</th></tr>{parents}</table>
{_table([("Parent clusters", ", ".join(prov.get("parent_clusters", []))), ("Any parent an allergen source", prov.get("any_parent_allergen_source")), ("Generation method", prov.get("generation_method"))])}
{("<h3>Design</h3>" + _table(list(prov["design"].items()))) if prov.get("design") else ""}
<h2>Evidence</h2><p class="small">{e(ev.get("gate", {}).get("rule", ""))} Gate passed: {e(str(ev.get("gate", {}).get("passed")))}. {e(str(ev.get("designed_peptide_rule") or ""))}</p>
<h3>For the candidate</h3>{_table(list((ev.get("candidate") or {}).items()))}
{("<h3>For its template (shown, not transferred)</h3>" + _table(list(ev["template"].items()))) if ev.get("template") else ""}
{_table([("Nearest positive peptide", ev.get("nearest_positive")), ("Nearest negative peptide", ev.get("nearest_negative")), ("Prior-art alert", ev.get("prior_art_alert")), ("Digestion support", ev.get("digestion_support"))])}
<h2>Safety screening</h2><h3>Predicted IgE binding (its own risk)</h3>{_table(list(safety.get("ige_binding", {}).items()))}
{_table([("Parent allergen flag", safety.get("parent_allergen_flag")), ("Other safety models", safety.get("other_safety_models")), ("Safety score", safety.get("safety_score")),
         ("Sequence liabilities", [x["meaning"] for x in safety.get("sequence_liabilities", [])]), ("Oxidation risks", [x["meaning"] for x in safety.get("oxidation_risks", [])]),
         ("Models not run", ", ".join(safety.get("predictions_not_run", [])) or "none"), ("Models left out of the safety score", safety.get("models_left_out_of_the_safety_score")),
         ("Applicability limits", safety.get("applicability_limits")), ("Rule", safety.get("rule"))])}
<h2>Digestion detail</h2><p class="small">{e(str(d.digestion.get("routes_meaning") or ""))}</p>{_routes_html(d.digestion.get("routes") or [])}
<h2>Evidence alignment</h2>{_alignment_html(ev)}
<h2>Ranking</h2>{_table([(k, v) for k, v in d.ranking.items() if k != "objectives"])}{_table([(o["name"], o["value"]) for o in d.ranking.get("objectives") or []])}
<h2>Chemistry</h2>{_table(list(d.chemistry.items()))}
<h2>Screening models</h2><table><tr><th>Model</th><th>Ranking value (0-1; higher is the favourable outcome of that model; the model's own output is under "model outputs")</th><th>Label</th><th>Applicability</th><th>Detail</th></tr>{models}</table>
{_table([(k, v) for k, v in d.screening.items() if k not in ("models", "applicability", "predictions_not_run")])}
<h2>Predicted cleavage by gut proteases</h2>{_table([(k, v) for k, v in gut.items()])}
<h2>Comparison materials</h2>{_table([("Parent proteins", [f"{p.get('accession')} ({p.get('food_source')})" for p in cmp_.get("parent_proteins", [])]),
                                       ("Hydrolysates to prepare", [f"{h.get('parent_protein')} by {h.get('pathway')}" for h in cmp_.get("hydrolysates", [])]), ("Note", cmp_.get("note"))])}
<table><tr><th>Reference peptide</th><th>Sequence</th><th>Source or similarity</th></tr>{refs}</table>
<h2>Limitations</h2><p>{e(d.limitations)}</p>
<h2>Not established</h2><ul>{"".join(f"<li>{e(s)}</li>" for s in d.not_established)}</ul>
<h2>Reference controls (proposed by rule)</h2><table><tr><th>Role</th><th>Sequence</th><th>Study</th><th>Assay</th><th>Observations</th></tr>{controls}</table>
<h2>Assay plan</h2><p>Primary endpoint: {e(d.assay_plan.get("primary_endpoint", ""))}. Material: {e(d.assay_plan.get("material_type", ""))}.</p><table>{plan}</table>
<h2>Review</h2><table><tr><th>Decision</th><td>{e(str(review.get("decision") or "approve / hold / reject: not decided"))}</td></tr><tr><th>Reason</th><td>{e(str(review.get("reason") or ""))}</td></tr>
<tr><th>Reviewer</th><td>{e(str(review.get("reviewer") or ""))}</td></tr><tr><th>Decided at</th><td>{e(str(review.get("decided_at") or ""))}</td></tr></table>
</body></html>"""


# ----------------------------------------------------------------------------------------------------
# The biological approval gate. The biological approval gate: the biology lead's decisions on ranked peptides, kept apart from the computed ranking.
#
# The ranking is computed by stage s07 and is never changed by a review. A decision is stored as one line of `review/decisions.jsonl` in the run folder; the file only grows, so the history
# of every decision is kept and the latest line for a peptide is its current decision. Only peptides the biology lead approved move on.


GATE_RULE = "Ranks are computed before the review and are never changed by it. Only peptides the biology lead approved move on."


class ReviewRequest(BaseModel):
    """One decision of the biology lead on one ranked peptide."""

    decision: Literal["approve", "hold", "reject"]
    reason: str = Field(default="", description="Why. Shown in the dossier and written to the experiment sheet.")
    reviewer: str = Field(min_length=1, description="Who decided. Free text: there is no sign-in yet.")


class ReviewRecord(BaseModel):
    peptide_id: str
    sequence: str
    decision: Literal["approve", "hold", "reject"]
    reason: str = ""
    reviewer: str
    decided_at: str = Field(description="UTC time the server saved the decision, ISO 8601.")


assert set(REVIEW_DECISIONS) == {"approve", "hold", "reject"}


def new_record(peptide_id: str, sequence: str, request: ReviewRequest) -> ReviewRecord:
    return ReviewRecord(peptide_id=peptide_id, sequence=sequence, decision=request.decision, reason=request.reason.strip(), reviewer=request.reviewer.strip(),
                        decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def parse_log(text: str) -> list[ReviewRecord]:
    """Every decision in the order it was saved."""
    return [ReviewRecord.model_validate_json(line) for line in text.splitlines() if line.strip()]


def latest(records: list[ReviewRecord]) -> dict[str, dict]:
    """The current decision of each peptide: its last saved one."""
    return {r.peptide_id: r.model_dump() for r in records}


def gate_summary(shortlisted: list[str], current: dict[str, dict]) -> dict:
    """Where the biological approval gate stands for the shortlisted peptides. Decisions on ranked peptides that are not on the shortlist are kept but not counted."""
    by = {d: [p for p in shortlisted if current.get(p, {}).get("decision") == d] for d in REVIEW_DECISIONS}
    pending = [p for p in shortlisted if p not in current]
    status = ("NO_SHORTLIST" if not shortlisted else "AWAITING_REVIEW" if len(pending) == len(shortlisted) else "REVIEW_COMPLETE" if not pending else "IN_REVIEW")
    return {"status": status, "shortlisted": len(shortlisted), "approved": by["approve"], "on_hold": by["hold"], "rejected": by["reject"], "pending": pending, "rule": GATE_RULE}


# ----------------------------------------------------------------------------------------------------
# How a run narrowed its candidates. How a food run narrowed its candidates to the ranked list, in the run's own numbers: the stage counts (from the run's audit log), the hard rules, the evidence gate, the caps and the
# selection rule (from the ranking report). Pure functions on saved run data, so the results route, the dossier and the PDF show the same facts.


STAGE_LABEL = {
    "s03_food_digestion": "Digested peptides, every protein and pathway",
    "s04_food_generation": "Designed peptides added",
    "s05_food_screening": "Hard rules, then the best of the pool sent to the models",
    "s06_food_gut_stability": "Predicted cleavage by gut proteases",
    "s07_food_ranking_decision": "Pareto ranking with the evidence gate and the caps",
}
CUT_TO_MODEL_POOL = "not_in_model_shortlist"
REASON_LABEL = {"non_standard_residue": "a residue other than the 20 standard amino acids", "too_short": "shorter than the minimum length", "too_long": "longer than the maximum length",
                CUT_TO_MODEL_POOL: "not among the best of the pool on the three cheap objectives (cut before the slow models run, near-duplicates dropped)"}
SELECTION_RULE = ("No weights. The ranking repeatedly takes, from Pareto front 1 of what remains, the peptide that dominates the most remaining candidates, puts it next in the list and removes "
                  "it. A peptide dominates another when it is at least as good on every objective and better on one. Ties are broken by the fixed order in tie_break (higher first), then by sequence.")
CAP_RULE = ("After the picks, the first peptide in pick order that takes a sequence cluster or a parent protein family over its cap is dropped for good and its place is refilled by the same rule, "
            "until nothing is over a cap.")


def stage_rows(audit: list[dict]) -> list[dict]:
    """One row per candidate stage of the run, from the audit log: how many came in, how many went out, and why the others were removed."""
    rows = []
    for a in audit:
        if a.get("kind") != "candidate_stage":
            continue
        rows.append({"stage": a["stage"], "label": STAGE_LABEL.get(a["stage"], a["stage"]), "n_in": a.get("n_in"), "n_out": a.get("n_out"),
                     "removed": {REASON_LABEL.get(k, k): v for k, v in (a.get("rejection_reasons") or {}).items()}})
    return rows


def ranking_summary(report: dict, audit: list[dict], n_shortlisted: int | None = None) -> dict:
    """What the results route and the reports say about how the ranking was made. A value an older run did not record is None, never a guess."""
    rules, gate, facts = report.get("rules") or {}, report.get("gate") or {}, report.get("facts") or {}
    stages = stage_rows(audit)
    s05 = next((s for s in stages if s["stage"] == "s05_food_screening"), None)
    cut = REASON_LABEL[CUT_TO_MODEL_POOL]
    return {
        "stages": stages,
        "ranked": next((s["n_out"] for s in stages if s["stage"] == "s07_food_ranking_decision"), None),
        "shortlisted": n_shortlisted,
        "pool_sent_to_the_models": s05["n_out"] if s05 else None,
        "hard_rules": {"standard_residues_only": True, "min_length": rules.get("min_length"), "max_length": rules.get("max_length"),
                       "removed": {k: v for k, v in (s05["removed"] if s05 else {}).items() if k != cut},
                       "cut_to_the_model_pool": (s05["removed"].get(cut) if s05 else None)},
        "objectives": {"used": facts.get("objectives_used"), "not_used_because_a_pooled_peptide_lacks_them": facts.get("objectives_unavailable"),
                       "meaning": "Each objective is a value from 0 to 1 where higher is the favourable outcome (for a model it is the ranking value, not the model's own output). An objective is used "
                                  "only when every pooled peptide has it."},
        "pareto": {"n_fronts": facts.get("n_pareto_fronts"), "n_sequence_clusters": facts.get("n_clusters"), "selection_rule": SELECTION_RULE, "tie_break": rules.get("tie_break"),
                   "weights": "None: the ranking is weight-free."},
        "evidence_gate": {"exact_evidence_requested": gate.get("min_exact_evidence_requested"), "exact_evidence_in_final": gate.get("exact_evidence_in_final"), "shortfall": gate.get("shortfall"),
                          "dropped_for_the_gate": gate.get("dropped_for_evidence_gate"),
                          "rule": "Of the ranked peptides, the run asks for a minimum number with an exact sequence match to an evidence peptide with a positive or mixed effect. If fewer exist, "
                                  "the shortfall is reported and the list is not padded."},
        "caps": {"max_per_sequence_cluster": rules.get("max_per_cluster"), "max_per_parent_family": rules.get("max_per_parent_family"), "dropped_for_the_caps": gate.get("dropped_for_diversity"),
                 "rule": CAP_RULE},
        "final_count_requested": rules.get("final_count"), "model_stage_pool_size": rules.get("model_stage_pool"),
    }
