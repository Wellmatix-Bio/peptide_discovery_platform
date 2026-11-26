"""s05_food_screening: exclusion rules, physicochemical properties and liabilities, evidence lookup, digestion support, prior-art and allergen checks, then the safety and developability models.

Ported from Phase 1 `7. AI Screening and Ranking/run.py` (stage A and the plug-in models). Peptides that break a hard rule are removed with the rule as the reason; the rest
are cut to the `MODEL_TOP` best on the three cheap objectives (near-duplicates dropped) before the slow models run, as in Phase 1.
"""
from __future__ import annotations

import numpy as np

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext
from pipeline.food import settings
from pipeline.food.evidence import EvidenceLookup, load_evidence
from pipeline.food.ranking.ranking import diverse_shortlist, dominator_counts
from pipeline.food.screening import checks, screening
from pipeline.food.screening.adapters import build_model_slots
from pipeline.food.screening.model_registry import ScreeningModel, run_models, status_table
from schemas.candidate import Candidate
from schemas.food import FoodCandidate
from schemas.run_config import StageConfig

logger = get_logger(__name__)


def screen_candidate(c: FoodCandidate, scorer: EvidenceLookup, prior_art: checks.PriorArt) -> dict:
    """The stage A fields of one candidate, with the Phase 1 column names."""
    seq = c.sequence
    row = {"length": len(seq), "origin_type": c.origin_type, **screening.physchem(seq)}
    row["developability_rule"] = row.pop("developability")
    gen = c.predictions.get("generation")
    row.update(scorer.lookup(seq))
    row["evidence_applies_to"] = "CANDIDATE"
    row["evidence_alignment"] = scorer.alignment(seq, row["evidence_peptide"])  # display only
    if c.origin_type == "SYNTHETIC_DESIGN" and gen:
        # The designed sequence is looked up as itself. Its template's evidence is reported next to it, never scored for it
        # (biological activity is not assigned from the template to the modified peptide).
        template = scorer.lookup(gen["template_sequence"])
        row.update({f"template_{k}": template[k] for k in ("evidence_type", "evidence_direction", "evidence_peptide", "evidence_similarity", "evidence_study",
                                                           "evidence_assay", "evidence_food_derivation", "evidence_tier")})
        row["template_evidence_alignment"] = scorer.alignment(gen["template_sequence"], template["evidence_peptide"])
        row.update(template_evidence_applies_to="TEMPLATE", template_sequence=gen["template_sequence"], edits=gen["edits"],
                   guide_peptide=gen["guide_peptide"], guide_similarity=gen["sim_to_guide"])
    row["digestion_support"] = screening.digestion_support(
        {"is_limit_product_any": str(any(o.is_limit_product for o in c.parents)), "n_parent_clusters": len(c.parent_clusters)})
    row["prior_art_alert"] = prior_art.alert(seq)
    row["parent_allergen_flag"] = checks.parent_allergen_flag(c.any_parent_allergen_source)
    return row


class FoodScreening(CandidateStage):
    name = "s05_food_screening"
    requires = {"origin_type", "parents"}
    produces = {"screening", "models"}

    def build_models(self, ctx: RunContext) -> list[ScreeningModel]:
        """The safety and developability model slots; a test replaces this with fakes."""
        return build_model_slots(ctx.feature_extractor)

    def run(self, candidates: list[Candidate], config: StageConfig, ctx: RunContext) -> list[Candidate]:
        evidence = load_evidence()
        scorer, prior_art = EvidenceLookup(evidence), checks.PriorArt(evidence)

        ok: list[FoodCandidate] = []
        for c in candidates:
            reason = screening.exclusion_reason(c.sequence)
            if reason:
                self.note_rejection(reason)
                continue
            c.predictions["screening"] = screen_candidate(c, scorer, prior_art)
            ok.append(c)

        pool = ok
        if ok:
            X = np.array([[c.predictions["screening"][k] for k in ("evidence", "digestion_support", "developability_rule")] for c in ok])
            order = np.lexsort((-X[:, 0], dominator_counts(X)))  # best Pareto position on the three cheap objectives first
            keep = diverse_shortlist([c.sequence for c in ok], order, settings.MODEL_TOP)
            pool = [ok[i] for i in keep]
            for _ in range(len(ok) - len(pool)):
                self.note_rejection("not_in_model_shortlist")

        models = self.build_models(ctx)
        results = run_models(models, [c.sequence for c in pool], logger.info)
        self.model_status = status_table(models)
        for c in pool:
            c.predictions["models"] = {name: res[c.sequence] for name, res in results.items() if c.sequence in res}
        logger.info("food_screening.done", extra={"n_in": len(candidates), "n_excluded": len(candidates) - len(ok), "n_model_stage": len(pool),
                                                  "models": {k: v["status"] for k, v in self.model_status.items()}})
        return pool
