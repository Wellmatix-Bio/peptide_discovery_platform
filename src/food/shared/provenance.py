"""Provenance checks of a food run: does every candidate really come from where it says?

Run on the final candidates after the pipeline. A candidate that fails is reported with the check that failed; the worker fails the run on any issue, so a
peptide whose lineage cannot be reproduced never reaches a report as if it could.
"""
from __future__ import annotations

from food.c_discovery import settings
from food.b_digestion.engine import peptide_id
from food.a_proteins.protein import Protein
from food.shared.candidate import FoodCandidate

REQUIRED_PREDICTIONS = ("screening", "models", "cleavage", "ranking")


def check_provenance(candidates: list[FoodCandidate], proteins: list[Protein]) -> list[dict]:
    """Issues found, each {"peptide_id", "check", "detail"}; an empty list means every check passed."""
    by_id = {p.substrate_id: p for p in proteins}
    issues: list[dict] = []

    def fail(c: FoodCandidate, check: str, detail: str) -> None:
        issues.append({"peptide_id": c.id, "check": check, "detail": detail})

    seen: dict[str, str] = {}
    for c in candidates:
        if c.id != peptide_id(c.sequence):
            fail(c, "id_matches_sequence", f"id {c.id} is not the content hash of {c.sequence}")
        if c.sequence in seen:
            fail(c, "no_duplicate_sequence", f"{c.sequence} also appears as {seen[c.sequence]}")
        seen[c.sequence] = c.id
        if set(c.sequence) - settings.STANDARD_AA or not settings.MIN_LENGTH <= len(c.sequence) <= settings.MAX_LENGTH:
            fail(c, "sequence_valid", f"{c.sequence}: only the 20 standard residues, {settings.MIN_LENGTH}-{settings.MAX_LENGTH} long")
        missing = [k for k in REQUIRED_PREDICTIONS if k not in c.predictions]
        if missing:
            fail(c, "stages_complete", f"no result from: {', '.join(missing)}")
        if not c.parents:
            fail(c, "has_a_parent", "no parent protein recorded")
        designed = c.origin_type == "SYNTHETIC_DESIGN"
        template = (c.predictions.get("generation") or {}).get("template_sequence")
        if designed and not template:
            fail(c, "design_has_template", "a designed peptide with no template sequence")
        expected = template if designed else c.sequence  # what the parent coordinates must contain
        clusters = set()
        for p in c.parents:
            protein = by_id.get(p.protein_id)
            if protein is None:
                fail(c, "parent_is_in_the_run", f"parent {p.protein_id} is not among the run's proteins")
                continue
            clusters.add(protein.cluster)
            start, end = p.start_in_product, p.end_in_product
            if start is None or end is None or protein.sequence[start - 1:end] != expected:
                fail(c, "sequence_at_parent_position", f"{p.protein_id} residues {start}-{end} are not {expected}")
            if (p.start_in_precursor, p.end_in_precursor) != (protein.precursor_start + (start or 0) - 1, protein.precursor_start + (end or 0) - 1):
                fail(c, "precursor_coordinates", f"{p.protein_id}: precursor {p.start_in_precursor}-{p.end_in_precursor} do not follow from {start}-{end}")
            for field, want in (("protein_version", protein.version), ("protein_checksum", protein.checksum), ("food_source", protein.food_source), ("allergen_source", protein.allergen_source)):
                if getattr(p, field) != want:
                    fail(c, f"parent_{field}", f"{p.protein_id}: {getattr(p, field)!r} recorded, the protein has {want!r}")
        known = {x for x in clusters if x}
        if not designed and set(c.parent_clusters) != known:
            fail(c, "parent_clusters", f"{sorted(c.parent_clusters)} recorded, the parents are in {sorted(known)}")
        if designed and not set(c.parent_clusters) <= known:
            fail(c, "parent_clusters", f"{sorted(c.parent_clusters)} recorded, the template parents are in {sorted(known)}")
    ranks = [c.predictions.get("ranking", {}).get("rank") for c in candidates]
    if candidates and ranks != list(range(1, len(candidates) + 1)):
        issues.append({"peptide_id": "", "check": "ranks_are_1_to_n", "detail": f"ranks {ranks[:12]}... are not 1..{len(candidates)} in order"})
    return issues
