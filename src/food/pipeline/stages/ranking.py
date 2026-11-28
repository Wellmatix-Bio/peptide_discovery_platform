"""s07_food_ranking_decision: derived scores, evidence-gated Pareto selection, rationale, biology shortlist and reference controls.

Ported from Phase 1 `7. AI Screening and Ranking/run.py` (after the model stage). No weighted total: objectives are compared by Pareto dominance, each model score is
its own objective, and an out-of-domain model does not veto a peptide. The evidence gate passes only an EXACT_SEQUENCE match to an evidence peptide with a positive or mixed
effect (MIN_EXACT_EVIDENCE of them are wanted in the final set). A designed peptide is looked up as itself; its template's evidence is never transferred to it.
"""
from __future__ import annotations

import numpy as np

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext
from food.c_discovery import settings
from food.b_digestion.explain import explain_candidates
from food.c_discovery.evidence import load_evidence
from food.c_discovery.ranking import ranking, shortlist
from food.c_discovery.screening import checks, glp1_classifier
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

logger = get_logger(__name__)

#: Slot name -> objective it feeds, in Phase 1 order (models_config.json).
MODEL_OBJECTIVES = {"allergenicity": "ige_binding", "cytotoxicity": "safety", "solubility": "developability", "aggregation": "developability",
                    "synthesis_feasibility": "developability", "bitterness": "developability"}


def flat_row(c: Candidate) -> dict:
    """One candidate as the flat row the ranking works on (Phase 1 column names)."""
    cleavage = {k: v for k, v in c.predictions.get("cleavage", {}).items() if k.startswith("cleave_") or k in ("uncut_bond_fraction", "stability")}
    return {"peptide_id": c.id, "sequence": c.sequence, "origin_type": c.origin_type, "parent_clusters": ";".join(c.parent_clusters),
            "n_parent_clusters": len(c.parent_clusters), **c.predictions["screening"], **cleavage}


def derive(row: dict, models: dict[str, dict], ready: set[str]) -> dict:
    """Digestion plausibility, the model objectives, safety and developability, evidence gate, applicability and coverage of one peptide."""
    gut = row.get("uncut_bond_fraction", np.nan)
    s = row["digestion_support"]
    row["digestion_plausibility"] = round((s + gut) / 2, 3) if not np.isnan(gut) else s
    row["glp1_classifier_status"] = glp1_classifier.STATUS  # reported, not ranked on
    row.setdefault("stability", np.nan)
    model_app = []
    for name in MODEL_OBJECTIVES:
        r = models.get(name)
        if r:
            row[f"model_{name}_score"], row[f"model_{name}_label"] = round(float(r["score"]), 3), r.get("label", "")
            row[f"applicability_{name}"] = r["applicability"]
            model_app.append(r["applicability"])
    for obj in ("safety", "developability"):
        sc = [(n, float(models[n]["score"]), models[n]["applicability"]) for n, o in MODEL_OBJECTIVES.items() if o == obj and n in models]
        val, basis, flagged = checks.combine_objective(sc, row["developability_rule"] if obj == "developability" else None)
        row[obj] = round(val, 3) if val == val else np.nan
        row[f"{obj}_basis"], row[f"{obj}_flagged_models"] = basis, flagged
    row["evidence_gate_passed"] = bool(row.get("evidence_type") == "EXACT_SEQUENCE" and row.get("evidence_direction") in ("positive", "mixed"))
    row["applicability_evidence"] = checks.evidence_applicability(row)
    row["applicability_stability"] = checks.stability_applicability(row)
    row["applicability_overall"] = checks.worst([row["applicability_evidence"], row["applicability_stability"]] + model_app)
    dims = {"evidence": row["applicability_evidence"], "stability": row["applicability_stability"],
            **{n: row.get(f"applicability_{n}", "not_run") for n in MODEL_OBJECTIVES}}
    row["coverage"] = checks.coverage(dims)
    row["applicability_assessed"] = f"{sum(v != 'not_run' for v in dims.values())} of {len(dims)}"  # overall = worst of these only
    row["applicability_not_assessed"] = ";".join(k for k, v in dims.items() if v == "not_run")
    # not run for this peptide: the model did not run at all, or it skipped this peptide (the aggregation model needs at least 5 amino acids)
    row["predictions_not_run"] = ";".join([n for n in MODEL_OBJECTIVES if n not in models] + ([] if row["applicability_stability"] != "not_run" else ["cleavage_model"]))
    return row


def limitations(row: dict, n_evidence: int, missing_models: list[str]) -> str:
    text = (f"models not run: {', '.join(missing_models) or 'none'}; the uncut-bond fraction comes from Expasy protease rules (pepsin pH 1.3, trypsin, "
            "chymotrypsin low specificity): a hypothesis with no kinetics, abundance or intestinal exposure, and it does not measure survival in the gut; "
            f"safety/developability exclude out-of-domain models (see *_flagged_models); a model that did not run is never counted as a pass; evidence = sequence lookup against "
            f"{n_evidence} labelled peptides (a sequence relationship, not proof of GLP-1 activity; no match means no data, not inactive; positives are mostly of unverified food origin); "
            "prior-art alert is not a freedom-to-operate opinion")
    if row["origin_type"] == "SYNTHETIC_DESIGN":
        text += ("; SYNTHETIC_DESIGN: its own sequence is looked up for evidence, its template's evidence is shown separately and is not transferred to it, "
                 "similarity to the guide is shown, not scored, activity of the edit is untested, and it is not food-approved")
    return text


def rank_pool(rows: list[dict], n_final: int, n_evidence: int, ready: set[str]) -> tuple[list[dict], dict, dict]:
    """(ranked rows in rank order, evidence-gate report, run facts). `rows` are derived rows of the model-stage pool."""
    wanted = settings.OBJECTIVES + [f"model_{n}_score" for n in MODEL_OBJECTIVES if n in ready] + ["evidence"]  # evidence last
    objs = [o for o in wanted if all(o in r and not np.isnan(r[o]) for r in rows)]
    dropped = [o for o in wanted if o not in objs]
    clusters = ranking.cluster([r["sequence"] for r in rows])
    parents = [shortlist.first_parent_cluster(r) for r in rows]
    chosen, gate, per = ranking.pareto_pick_selection(rows, objs, clusters, parents, n_final, settings.MIN_EXACT_EVIDENCE)
    for k, r in enumerate(rows):
        r["cluster"], r["n_fronts"] = int(clusters[k]), per["n_fronts"]
        r["pareto_front"], r["dominates_n_at_pick"] = per["pareto_front"][k], per["dominates_n_at_pick"][k]
    best = {o: max(r[o] for r in rows) for o in objs}
    final = []
    for rank, i in enumerate(chosen, 1):
        r = dict(rows[i])
        r["rank"] = rank
        r["rationale"] = ranking.rationale(r, objs, best)
        r["limitations"] = limitations(r, n_evidence, [n for n in r["predictions_not_run"].split(";") if n])
        final.append(r)
    return final, gate, {"objectives_used": objs, "objectives_unavailable": dropped, "n_pareto_fronts": per["n_fronts"], "n_clusters": int(clusters.max())}


class FoodRankingDecision(CandidateStage):
    name = "s07_food_ranking_decision"
    requires = {"screening"}
    produces = {"ranking"}

    def run(self, candidates: list[Candidate], config: StageConfig, ctx: RunContext) -> list[Candidate]:
        self.shortlist, self.controls, self.gate, self.facts, self.digestion_warnings = [], [], {}, {}, []
        if not candidates:
            return []
        ready = {n for c in candidates for n in c.predictions.get("models", {})}
        rows = [derive(flat_row(c), c.predictions.get("models", {}), ready) for c in candidates]
        final, self.gate, self.facts = rank_pool(rows, settings.FINAL_COUNT, len(load_evidence()), ready)
        self.shortlist, self.controls = shortlist.make_shortlist(final), shortlist.reference_controls()
        position = {r["peptide_id"]: r["shortlist_position"] for r in self.shortlist}
        reason = {d["peptide_id"]: d["reason"] for d in shortlist.shortlist_decisions(final)}
        by_id = {c.id: c for c in candidates}
        proteins = getattr(ctx, "proteins", None)
        self.digestion_warnings = explain_candidates([by_id[r["peptide_id"]] for r in final], proteins) if proteins else []  # explains the origins; ranks are already fixed
        out = []
        for r in final:
            c = by_id[r["peptide_id"]]
            c.predictions["ranking"] = {**{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items() if k not in c.predictions["screening"]},
                                        "shortlist_position": position.get(r["peptide_id"]), "shortlist_reason": reason[r["peptide_id"]]}
            out.append(c)
        logger.info("food_ranking.done", extra={"n_pool": len(candidates), "n_ranked": len(out), "n_shortlist": len(self.shortlist), "gate": self.gate})
        return out
