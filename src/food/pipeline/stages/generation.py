"""s04_food_generation: reference-guided edits of digestion fragments, added to the run as SYNTHETIC_DESIGN candidates. Optional."""
from __future__ import annotations

import hashlib

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext
from food.c_discovery import settings
from food.b_digestion import masses
from food.pipeline.stages.digestion import parent_origin
from food.c_discovery.evidence import load_evidence
from food.c_discovery.generation import generate as G
from food.c_discovery.generation import settings as S
from schemas.candidate import Candidate
from food.shared.candidate import FoodCandidate, ParentOrigin
from schemas.run_config import StageConfig

logger = get_logger(__name__)

#: Phase 1 generated_peptides.csv columns kept under predictions["generation"], so the lineage of an edit stays visible.
LINEAGE = ("template_peptide_id", "template_sequence", "template_origin_type", "edits", "n_edits", "edit_kinds", "guide_peptide",
           "guide_origin_type", "guide_study", "guide_food_derivation", "sim_to_guide", "template_sim_to_guide", "nearest_negative_sim",
           "evidence_match", "net_charge_pH7_4", "liabilities", "seed", "generator_version", "evidence_db_md5")


class FoodGeneration(CandidateStage):
    name = "s04_food_generation"
    requires = {"origin_type", "parents"}
    produces: set[str] = set()

    def run(self, candidates: list[Candidate], config: StageConfig, ctx: RunContext) -> list[Candidate]:
        proteins = getattr(ctx, "proteins", None)
        if not proteins:
            raise ValueError("no proteins in the run context; the parent sequences are needed to check digestion plausibility")
        by_id = {p.substrate_id: p for p in proteins}
        digested = [c for c in candidates if c.origin_type == "DIGESTION_DERIVED"]
        rows = [{"peptide_id": c.id, "sequence": c.sequence, "length": len(c.sequence),
                 "is_limit_product_any": str(any(o.is_limit_product for o in c.parents)),
                 "parent_clusters": ";".join(c.parent_clusters), "n_parent_clusters": len(c.parent_clusters)} for c in digested]
        origins = {c.id: [{"substrate_id": o.protein_id, "accession": o.accession, "food_source": o.food_source,
                           "start_in_product": o.start_in_product, "is_limit_product": o.is_limit_product} for o in c.parents] for c in digested}
        evidence = load_evidence()
        _, guide_scorer, evidence_seqs = G.guides_and_scorers(evidence)
        templates = G.select_templates(rows, evidence_seqs, guide_scorer, S.N_TEMPLATES)
        passed, _, _, reasons = G.generate(
            templates, lambda pid: G.pick_origin(origins[pid]), {p.substrate_id: p.sequence for p in proteins},
            {c.sequence for c in candidates}, evidence, seed=ctx.seed if hasattr(ctx, "seed") else S.SEED,
            evidence_md5=hashlib.md5(settings.EVIDENCE_DB.read_bytes()).hexdigest())
        have = {c.sequence for c in candidates}
        out = list(candidates)
        for r in passed:
            if r["sequence"] in have:
                continue
            protein = by_id[r["parent_substrate"]]
            offset = protein.precursor_start - 1
            parents = [ParentOrigin(
                protein_id=protein.substrate_id, accession=protein.accession, protein_version=protein.version, protein_checksum=protein.checksum,
                food_source=protein.food_source, allergen_source=protein.allergen_source,
                protein_n_terminus=protein.n_terminus or None, protein_c_terminus=protein.c_terminus or None,
                start_in_product=r["template_start_in_parent"], end_in_product=r["template_end_in_parent"],
                start_in_precursor=offset + r["template_start_in_parent"], end_in_precursor=offset + r["template_end_in_parent"],
                pathway=pw, missed_cleavages_final=r["missed_cleavages_final"] if r["missed_cleavages_final"] != "" else None,
                is_limit_product=r["is_limit_product_any"] == "True") for pw in r["plausible_pathways"].split(";")]
            out.append(FoodCandidate(
                id=r["peptide_id"], sequence=r["sequence"], origin_type=S.ORIGIN_TYPE, evidence_class="THEORETICAL",
                generation_method=S.GENERATION_METHOD, parents=parents, duplicate_family_id=r["peptide_id"],
                mono_mass=round(masses.mono_mass(r["sequence"]), 5), avg_mass=masses.avg_mass(r["sequence"]), chemistry_notes=masses.chemistry_notes(r["sequence"]),
                parent_clusters=[protein.cluster] if protein.cluster else [],
                predictions={"generation": {k: r[k] for k in LINEAGE}}))
            have.add(r["sequence"])
        for reason, n in reasons.items():
            logger.info("food_generation.filter", extra={"reason": reason, "n": n})
        logger.info("food_generation.done", extra={"n_templates": len(templates), "n_generated": len(out) - len(candidates)})
        return out
