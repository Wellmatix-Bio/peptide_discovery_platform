"""What each final peptide's parent origins look like when the digestion is run again: the enzymes and pH of each stage, the cut on each side of the fragment with its flanking residues,
the cuts still inside it, and final or intermediate. The recorded origin is compared with the new run; a difference is a warning, never a failure. A designed peptide is explained
by its template fragment, the part of the parent that digestion produces."""
from __future__ import annotations

from food.b_digestion.engine import digest_substrate
from food.b_digestion.settings import DEFAULT_SETTINGS


def _cut(row: dict | None) -> dict | str:
    if row is None:
        return "protein terminus"
    return {"position_in_precursor": row["precursor_p1_position"], "p1_residue": row["p1_residue"], "p1_prime_residue": row["p1p_residue"], "cleaved_by": row["cleaved_by"].split(";")}


def _detail(pathway, protein, start: int, end: int, cmap: list[dict], of: str) -> dict:
    last = len(pathway.stages)
    final = {r["p1_position"]: r for r in cmap if r["stage_index"] == last}  # every cut of the whole pathway, with the enzymes that made it
    n = len(protein.sequence)
    inside = [{"position_in_peptide": pos - start + 1, "p1_residue": r["p1_residue"], "p1_prime_residue": r["p1p_residue"], "cleaved_by": r["cleaved_by"].split(";")}
              for pos, r in sorted(final.items()) if start <= pos < end]
    return {"of": of, "protein_id": protein.substrate_id, "accession": protein.accession, "start_in_precursor": protein.precursor_start + start - 1,
            "end_in_precursor": protein.precursor_start + end - 1, "pathway": pathway.id, "stages": [{"stage": st.name, "enzymes": list(st.enzymes), "ph": st.ph} for st in pathway.stages],
            "n_side_cut": _cut(final.get(start - 1)) if start > 1 else "protein terminus", "c_side_cut": _cut(final.get(end)) if end < n else "protein terminus",
            "residue_before": protein.sequence[start - 2] if start > 1 else None, "residue_after": protein.sequence[end] if end < n else None,
            "cuts_inside_after_the_pathway": inside, "product_type": "FINAL_PRODUCT" if not inside else "INTERMEDIATE_PRODUCT"}


def _run(protein, pathway, cache: dict, settings):
    key = (protein.substrate_id, pathway.id)
    if key not in cache:
        cache[key] = digest_substrate(protein, pathway, settings, settings.registry())
    return cache[key]


def explain_parent(parent, protein, cache: dict, settings=DEFAULT_SETTINGS) -> tuple[dict | None, list[str]]:
    """(detail, problems) for one parent origin of a digested peptide; detail is None when the pathway or the fragment cannot be found."""
    pathway = next((p for p in settings.pathways if p.id == parent.pathway), None)
    if pathway is None or parent.start_in_product is None:
        return None, [f"the pathway {parent.pathway!r} is not a known digestion pathway"]
    origins, cmap = _run(protein, pathway, cache, settings)
    start, end = parent.start_in_product, parent.end_in_product
    same = [o for o in origins if o["start_in_product"] == start and o["end_in_product"] == end and o["stage"] == parent.stage]
    if not same:
        return None, [f"{pathway.id}: the recorded stage {parent.stage!r} does not release residues {start}-{end} of {protein.substrate_id}"]
    problems = [f"{pathway.id}/{parent.stage}: {field} recorded as {recorded!r}, the digestion run gives {same[0][field]!r}"
                for field, recorded in (("is_limit_product", parent.is_limit_product), ("missed_cleavages_final", parent.missed_cleavages_final),
                                        ("missed_cleavages_stage", parent.missed_cleavages_stage), ("n_term", parent.n_term), ("c_term", parent.c_term)) if same[0][field] != recorded]
    detail = _detail(pathway, protein, start, end, cmap, "peptide")
    if parent.is_limit_product is not None and (detail["product_type"] == "FINAL_PRODUCT") != parent.is_limit_product:
        problems.append(f"{pathway.id}: recorded as {'final' if parent.is_limit_product else 'intermediate'}, the digestion run gives {detail['product_type'].lower()}")
    return detail, problems


def explain_template(candidate, protein, cache: dict, settings=DEFAULT_SETTINGS) -> tuple[list[dict], list[str]]:
    """The digestion of a designed peptide's TEMPLATE fragment, over every pathway that releases it (the pathways and the is_limit_product stored on the design's parent origins describe the
    edited sequence, which can differ from the template, so they are not compared)."""
    parent = candidate.parents[0]
    start, end = parent.start_in_product, parent.end_in_product
    details = []
    for pathway in settings.pathways:
        origins, cmap = _run(protein, pathway, cache, settings)
        rows = [o for o in origins if o["start_in_product"] == start and o["end_in_product"] == end]
        if rows:
            details.append(_detail(pathway, protein, start, end, cmap, "template fragment"))
    problems = []
    if not details:
        problems.append(f"no pathway releases the template fragment (residues {start}-{end} of {protein.substrate_id})")
    return details, problems


def explain_candidates(candidates: list, proteins: list) -> list[dict]:
    """Fills `digestion_detail` on the parent origins of digested peptides and `predictions["generation"]["template_digestion"]` for designed ones; returns the warnings, each
    {peptide_id, check, detail}. Nothing the ranking uses is touched."""
    by_id = {p.substrate_id: p for p in proteins}
    cache: dict = {}
    warnings: list[dict] = []
    for c in candidates:
        if c.origin_type == "SYNTHETIC_DESIGN":
            protein = by_id.get(c.parents[0].protein_id) if c.parents else None
            if protein is None or "generation" not in c.predictions:
                continue
            details, problems = explain_template(c, protein, cache)
            c.predictions["generation"]["template_digestion"] = details
        else:
            problems = []
            for parent in c.parents:
                protein = by_id.get(parent.protein_id)
                if protein is None:
                    continue
                parent.digestion_detail, found = explain_parent(parent, protein, cache)
                problems += found
        warnings += [{"peptide_id": c.id, "check": "digestion_consistency", "detail": text} for text in problems]
    return warnings
