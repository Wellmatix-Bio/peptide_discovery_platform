"""Biology shortlist (5-10 peptides) and reference controls, chosen from the ranked candidates by fixed rules."""
from __future__ import annotations

import csv

from schemas.food_dossier import PROPOSED_STATUS

from .. import settings


def first_parent_cluster(c: dict) -> str:
    return (c.get("parent_clusters") or "").split(";")[0]


#: A model's own unfavourable call (its label at its own 0.5 boundary) that keeps a peptide off the shortlist: the cell assay needs a viable, soluble, synthesisable peptide.
UNFAVOURABLE = {"model_cytotoxicity_label": ("predicted_cytotoxic", "cytotoxic"), "model_aggregation_label": ("predicted_aggregation_prone", "aggregation-prone"),
                "model_solubility_label": ("predicted_insoluble", "insoluble"), "model_synthesis_feasibility_label": ("predicted_difficult", "hard to synthesise")}

EVIDENCE_WORDS = {"EXACT_SEQUENCE": "exact sequence", "CONTAINS_REFERENCE": "contains an evidence peptide", "CONTAINED_IN_REFERENCE": "is part of an evidence peptide",
                  "SEQUENCE_SIMILARITY": "sequence similarity only", "NO_EVIDENCE": "no evidence"}


def unfavourable_calls(c: dict) -> list[str]:
    """What the models themselves call unfavourable for this peptide, in words."""
    return [text for key, (label, text) in UNFAVOURABLE.items() if c.get(key) == label]


def shortlist_decisions(final: list[dict]) -> list[dict]:
    """Walk the ranking in rank order and say, for every ranked peptide, whether it is on the shortlist and why not if it is not. A peptide is skipped when:
    1. its evidence is not an exact, contains or contained-in relationship to an evidence peptide with a positive or mixed effect (SHORTLIST_EVIDENCE);
    2. a model itself calls it unfavourable (cytotoxic, aggregation-prone, insoluble or hard to synthesise);
    3. a higher-ranked peptide of the same parent protein family is already on the list;
    4. SHORTLIST_MAX_SYNTHETIC designed peptides are already on the list;
    or when the list holds SHORTLIST_SIZE peptides. There is no cap on peptides without an exact match. Prior-art alerts are kept and shown."""
    fams, n_syn, taken, out = set(), 0, 0, []
    for c in final:
        fam, syn = first_parent_cluster(c), c["origin_type"] == "SYNTHETIC_DESIGN"
        kind = c.get("evidence_type") or "NO_EVIDENCE"
        bad = unfavourable_calls(c)
        if kind not in settings.SHORTLIST_EVIDENCE or c.get("evidence_direction") not in ("positive", "mixed"):
            decision, reason = False, f"not eligible: its evidence is {EVIDENCE_WORDS.get(kind, kind)}"
        elif bad:
            decision, reason = False, f"left off: a model calls it unfavourable ({', '.join(bad)})"
        elif fam in fams:
            decision, reason = False, "skipped: a higher-ranked peptide of the same parent protein family is already on the shortlist"
        elif syn and n_syn >= settings.SHORTLIST_MAX_SYNTHETIC:
            decision, reason = False, f"skipped: {settings.SHORTLIST_MAX_SYNTHETIC} designed peptides are already on the shortlist"
        elif taken >= settings.SHORTLIST_SIZE:
            decision, reason = False, f"not needed: the shortlist already holds {settings.SHORTLIST_SIZE} peptides"
        else:
            fams.add(fam)
            n_syn += syn
            taken += 1
            decision, reason = True, f"on the shortlist (position {taken})"
        out.append({"peptide_id": c["peptide_id"], "shortlisted": decision, "reason": reason})
    return out


def make_shortlist(final: list[dict]) -> list[dict]:
    """The shortlisted peptides in rank order (see `shortlist_decisions` for the rules)."""
    chosen = {d["peptide_id"] for d in shortlist_decisions(final) if d["shortlisted"]}
    out = []
    for c in final:
        if c["peptide_id"] not in chosen:
            continue
        out.append({"shortlist_position": len(out) + 1, "rank": c["rank"], "peptide_id": c["peptide_id"], "sequence": c["sequence"],
                    "origin_type": c["origin_type"], "evidence_type": c.get("evidence_type"), "evidence_gate_passed": bool(c.get("evidence_gate_passed")), "applicability_overall": c.get("applicability_overall"), "applicability_assessed": c.get("applicability_assessed"),
                    "evidence_peptide": c.get("evidence_peptide"),
                    "parent_family": first_parent_cluster(c), "prior_art_alert": c.get("prior_art_alert", ""), "parent_allergen_flag": c.get("parent_allergen_flag", ""),
                    "rationale": c["rationale"], "limitations": c["limitations"],
                    "status": PROPOSED_STATUS})
    return out


def reference_controls() -> list[dict]:
    """Known positives and negatives of the evidence database, picked by rule: primary-assay evidence, documented food parent
    first, then the most observations. These are proposals; the biology lead sets the controls."""
    with open(settings.EVIDENCE_DB, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    def pick(direction: str, n_col: str) -> list[dict]:
        pool = [r for r in rows if r["direction"] == direction and r["evidence_tier"] == "primary_assay"]
        pool.sort(key=lambda r: (r["food_derivation"] != "food_parent_stated", -int(r[n_col] or 0), r["sequence"]))
        return pool[:settings.N_CONTROLS]

    out = []
    for role, direction, col in (("positive_control", "positive", "n_pos_obs"), ("negative_control", "negative", "n_neg_obs")):
        for r in pick(direction, col):
            out.append({"role": role, "sequence": r["sequence"], "origin_type": "LITERATURE_KNOWN", "study": r["studies"],
                        "assay": r["cell_models"], "n_observations": r[col], "food_derivation": r["food_derivation"],
                        "note": "proposed by rule; the biology lead finalizes controls, matched vehicle and assay system"})
    return out
