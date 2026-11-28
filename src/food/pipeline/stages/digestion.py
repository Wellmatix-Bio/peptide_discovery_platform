"""s03_food_digestion: the proteins in the run context become one FoodCandidate per unique digestion peptide, with every origin kept."""
from __future__ import annotations

from collections import defaultdict

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext
from food.b_digestion.engine import run as digest
from food.b_digestion.settings import DEFAULT_SETTINGS
from schemas.candidate import Candidate
from food.shared.candidate import FoodCandidate, ParentOrigin
from schemas.run_config import StageConfig

logger = get_logger(__name__)


def parent_origin(o: dict, protein) -> ParentOrigin:
    """One origin row of the engine as a ParentOrigin, with the protein's provenance copied in."""
    return ParentOrigin(
        protein_id=o["substrate_id"], accession=o["accession"], protein_version=protein.version, protein_checksum=protein.checksum,
        food_source=o["food_source"], allergen_source=protein.allergen_source,
        protein_n_terminus=protein.n_terminus or None, protein_c_terminus=protein.c_terminus or None,
        start_in_product=o["start_in_product"], end_in_product=o["end_in_product"],
        start_in_precursor=o["start_in_precursor"], end_in_precursor=o["end_in_precursor"],
        pathway=o["pathway"], stage=o["stage"], enzymes=o["stage_enzymes"].split("+"),
        ph=None if o["stage_ph"] == "" else float(o["stage_ph"]),
        missed_cleavages_stage=o["missed_cleavages_stage"], missed_cleavages_final=o["missed_cleavages_final"],
        n_term=o["n_term"], c_term=o["c_term"], is_limit_product=o["is_limit_product"])


class FoodDigestion(CandidateStage):
    name = "s03_food_digestion"
    requires: set[str] = set()
    produces = {"origin_type", "parents"}

    def run(self, candidates: list[Candidate], config: StageConfig, ctx: RunContext) -> list[Candidate]:
        proteins = getattr(ctx, "proteins", None)
        if not proteins:
            raise ValueError("no proteins in the run context; the protein library must run first")
        by_id = {p.substrate_id: p for p in proteins}
        origins, peptides, _ = digest(proteins, DEFAULT_SETTINGS)
        by_peptide: dict[str, list[dict]] = defaultdict(list)
        for o in origins:
            by_peptide[o["peptide_id"]].append(o)
        out = []
        for p in peptides:
            rows = by_peptide[p["peptide_id"]]
            clusters = sorted({by_id[o["substrate_id"]].cluster for o in rows if by_id[o["substrate_id"]].cluster})
            out.append(FoodCandidate(
                id=p["peptide_id"], sequence=p["sequence"], origin_type=p["origin_type"], evidence_class=p["evidence_class"],
                generation_method=p["generation_method"], parents=[parent_origin(o, by_id[o["substrate_id"]]) for o in rows],
                duplicate_family_id=p["peptide_id"], parent_clusters=clusters, mono_mass=p["mono_mass"], avg_mass=p["avg_mass"],
                chemistry_notes=p["chemistry_notes"]))
        logger.info("food_digestion.done", extra={"n_proteins": len(proteins), "n_peptides": len(out), "n_origins": len(origins)})
        return out
