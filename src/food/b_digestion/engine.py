"""Virtual digestion: enumerate theoretical fragments for each substrate and pathway.

Model: for stage k the cleavable bonds are the union of the sites of every enzyme in
stages 1..k (cleavage is irreversible). A fragment is a stretch between two cumulative
sites; its *missed cleavages* are the cumulative sites strictly inside it. Every fragment
up to ``max_missed_cleavages`` is enumerated at every stage, so the output is the set of
*possible* products, not a kinetic simulation. Fragments with no internal site at the
final stage are flagged ``is_limit_product``.

All output is THEORETICAL. Nothing here establishes abundance, stability or
intestinal exposure.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

from food.a_proteins.fasta import Substrate
from .masses import avg_masses, mono_mass
from .rules import Enzyme
from .settings import EVIDENCE_CLASS, ORIGIN_TYPE, Pathway, Settings


def peptide_id(seq: str) -> str:
    """Stable, content-derived ID. Identical sequences from any parent or run share it,
    so it doubles as the exact-duplicate family identifier."""
    return "PEP-" + hashlib.sha256(seq.encode()).hexdigest()[:12].upper()


def _site_labels(seq: str, pathway: Pathway, reg: dict[str, Enzyme]):
    """For each stage index: {site -> sorted list of 'stage:enzyme'} (only sites new to
    or re-found at that stage), plus the per-stage cumulative site list."""
    per_stage: list[dict[int, list[str]]] = []
    for st in pathway.stages:
        found: dict[int, list[str]] = {}
        for e in st.enzymes:
            for i in reg[e].sites(seq):
                found.setdefault(i, []).append(f"{st.name}:{e}")
        per_stage.append(found)
    return per_stage


def cumulative_cleavages(seq: str, pathway: Pathway, reg: dict[str, Enzyme]) -> list[dict[int, list[str]]]:
    """Per stage, every cleavable bond once the stage has run: {P1 index (0-based) -> sorted 'stage:enzyme' labels}. Cleavage is irreversible, so each stage includes the earlier ones."""
    cumulative: dict[int, list[str]] = {}
    out: list[dict[int, list[str]]] = []
    for found in _site_labels(seq, pathway, reg):
        for i, labs in found.items():
            known = cumulative.setdefault(i, [])
            known.extend(lab for lab in labs if lab not in known)
        out.append({i: list(labs) for i, labs in cumulative.items()})
    return out


def digest_substrate(sub: Substrate, pathway: Pathway, settings: Settings,
                     reg: dict[str, Enzyme]) -> tuple[list[dict], list[dict]]:
    """Returns (origin rows, cleavage-map rows) for one substrate and pathway."""
    seq = sub.sequence
    n = len(seq)
    origins: list[dict] = []
    cmap: list[dict] = []
    cumulative_by_stage = cumulative_cleavages(seq, pathway, reg)
    cumulative_labels = cumulative_by_stage[-1]
    stage_sites = [sorted(c) for c in cumulative_by_stage]
    final_sites = stage_sites[-1]
    final_set = set(final_sites)

    for si, st in enumerate(pathway.stages):
        sites = stage_sites[si]
        for i in sites:
            cmap.append({
                "substrate_id": sub.substrate_id, "pathway": pathway.id, "stage_index": si + 1,
                "stage": st.name, "p1_position": i + 1, "p1_residue": seq[i], "p1p_residue": seq[i + 1],
                "precursor_p1_position": sub.precursor_start + i,
                "cleaved_by": ";".join(sorted(cumulative_labels[i])),
            })
        bounds = [-1] + sites + [n - 1]
        nb = len(bounds)
        for j in range(nb - 1):
            start = bounds[j] + 1
            for k in range(j + 1, min(nb, j + 2 + settings.max_missed_cleavages)):
                end = bounds[k]
                length = end - start + 1
                if length > settings.max_length:
                    break  # extending only lengthens
                if length < settings.min_length:
                    continue
                pep = seq[start:end + 1]
                internal_final = sum(1 for x in final_sites if start <= x < end)
                origins.append({
                    "peptide_id": peptide_id(pep),
                    "sequence": pep,
                    "substrate_id": sub.substrate_id,
                    "accession": sub.accession,
                    "also_accessions": sub.also,
                    "food_source": sub.food_source,
                    "product_kind": sub.kind,
                    "start_in_product": start + 1,
                    "end_in_product": end + 1,
                    "start_in_precursor": sub.precursor_start + start,
                    "end_in_precursor": sub.precursor_start + end,
                    "pathway": pathway.id,
                    "stage_index": si + 1,
                    "stage": st.name,
                    "stage_ph": "" if st.ph is None else st.ph,
                    "stage_enzymes": "+".join(st.enzymes),
                    "n_term": "protein_terminus" if start == 0 else ";".join(sorted(cumulative_labels[start - 1])),
                    "c_term": "protein_terminus" if end == n - 1 else ";".join(sorted(cumulative_labels[end])),
                    "missed_cleavages_stage": k - j - 1,
                    "missed_cleavages_final": internal_final,
                    "is_limit_product": internal_final == 0,
                    "origin_type": ORIGIN_TYPE,
                    "evidence_class": EVIDENCE_CLASS,
                })
    return origins, cmap


def run(substrates: Iterable[Substrate], settings: Settings):
    """Digest all substrates. Returns (origins, peptides, cleavage_map); all lists are
    deterministically ordered."""
    reg = settings.registry()
    origins: list[dict] = []
    cmap: list[dict] = []
    subs = sorted(substrates, key=lambda s: s.substrate_id)
    for sub in subs:
        for pw in settings.pathways:
            o, c = digest_substrate(sub, pw, settings, reg)
            origins.extend(o)
            cmap.extend(c)

    origins.sort(key=lambda o: (o["substrate_id"], o["pathway"], o["stage_index"],
                                o["start_in_product"], o["end_in_product"]))
    for n, o in enumerate(origins, 1):
        o["origin_id"] = f"ORG-{n:07d}"

    agg: dict[str, dict] = {}
    for o in origins:
        p = agg.get(o["peptide_id"])
        if p is None:
            seq = o["sequence"]
            agg[o["peptide_id"]] = p = {
                "peptide_id": o["peptide_id"], "sequence": seq, "length": len(seq),
                "mono_mass": round(mono_mass(seq), 5), "avg_mass": 0.0,
                "origin_type": ORIGIN_TYPE, "generation_method": "in_silico_digestion",
                "evidence_class": EVIDENCE_CLASS,
                "chemistry_notes": ("linear; free N- and C-termini; no modifications"
                                    + ("; contains Cys (disulfide/oxidation not modelled)" if "C" in seq else "")
                                    + ("; contains Met (oxidation not modelled)" if "M" in seq else "")),
                "_parents": set(), "_pathways": set(), "_limit": False, "_n": 0,
            }
        p["_parents"].add(o["substrate_id"])
        p["_pathways"].add(o["pathway"])
        p["_limit"] = p["_limit"] or o["is_limit_product"]
        p["_n"] += 1
    peptides = []
    for pid in sorted(agg):
        p = agg[pid]
        peptides.append({
            **{k: v for k, v in p.items() if not k.startswith("_")},
            "n_origins": p["_n"], "n_parent_substrates": len(p["_parents"]),
            "pathways": ";".join(sorted(p["_pathways"])),
            "is_limit_product_any": p["_limit"],
            "parent_substrates": ";".join(sorted(p["_parents"])),
        })
    for p, avg in zip(peptides, avg_masses([p["sequence"] for p in peptides])):
        p["avg_mass"] = avg
    return origins, peptides, cmap
