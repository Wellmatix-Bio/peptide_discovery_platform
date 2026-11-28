"""Protein-family clusters for the diversity cap of the ranking: every peptide inherits the cluster of its parent protein.

Order, per protein:
1. the protein family from the source records (UniProt-based, by the data foundation's rules), within its food source, when the protein has one;
2. sequence similarity (MMseqs2, through the `pymmseqs` package) for the proteins without a family;
3. otherwise a cluster of its own, named by its accession (also the result for every such protein when pymmseqs is not installed or clustering fails).
`Protein.cluster_source` says which of the three gave the cluster.

pymmseqs 1.2.0 ships the MMseqs2 binary in its Linux and macOS wheels and has no Windows build, so on Windows every protein falls back to step 3.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from .fasta import QCIssue
from .protein import Protein

# Sequence identity of a cluster. 70% is the highest level that reproduces the curated protein families (adjusted Rand index 0.855; 90% gives 0.167):
# docs/validation/food/clustering_validation.md. 90% only groups near-duplicates, so one family spans dozens of clusters.
IDENTITY = 0.7


def _read_mmseqs(path: Path) -> dict[str, str]:
    """member -> representative from `<prefix>_cluster.tsv` (one `rep<TAB>member` row per member)."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rep, _, member = line.partition("\t")
        if member:
            out[member] = rep
    return out


def cluster_by_sequence(proteins: list[Protein], identity: float = IDENTITY, easy_cluster=None) -> tuple[dict[str, str], str]:
    """({substrate_id: representative substrate_id}, note). The map is empty, with the reason in the note, when pymmseqs is missing or clustering fails.

    `easy_cluster` is `pymmseqs.commands.easy_cluster` (the same as `mmseqs easy-cluster in out tmp --min-seq-id identity`, other settings at their MMseqs2 defaults);
    it is a parameter so a test can supply a stand-in."""
    if easy_cluster is None:
        try:
            from pymmseqs.commands import easy_cluster
        except ImportError:
            return {}, "pymmseqs is not installed (it has no Windows build)"
    ids = {str(i): p.substrate_id for i, p in enumerate(proteins)}  # short ids: MMseqs2 uses the first word of the header
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.fasta").write_text("".join(f">{i}" + chr(10) + p.sequence + chr(10) for i, p in enumerate(proteins)), encoding="utf-8")
        try:
            easy_cluster(str(tmp / "in.fasta"), str(tmp / "out"), str(tmp / "work"), min_seq_id=identity)
            found = _read_mmseqs(tmp / "out_cluster.tsv")
        except Exception as exc:  # noqa: BLE001 - clustering must not stop a run; the note says why every protein stands alone
            return {}, f"pymmseqs failed: {type(exc).__name__}: {exc}"
    return {ids[m]: ids[r] for m, r in found.items()}, f"pymmseqs (MMseqs2) at {identity:.0%} identity"


def assign_clusters(proteins: list[Protein], identity: float = IDENTITY, easy_cluster=None) -> tuple[list[Protein], list[str]]:
    """Proteins with `cluster` and `cluster_source` set, and notes on how the sequence clustering went."""
    need = [p for p in proteins if not p.protein_family]
    by_sequence, note = cluster_by_sequence(need, identity, easy_cluster) if need else ({}, "")
    out = []
    for p in proteins:
        if p.protein_family:
            out.append(replace(p, cluster=f"family:{p.food_source + ': ' if p.food_source else ''}{p.protein_family}", cluster_source="family"))  # the food source is part of the key: "Other seed storage proteins" exists in wheat, pea and soy
        elif p.substrate_id in by_sequence:
            out.append(replace(p, cluster=f"seq:{by_sequence[p.substrate_id]}", cluster_source="sequence"))
        else:
            out.append(replace(p, cluster=f"protein:{p.accession}", cluster_source="protein_id"))
    return out, [note] if note else []


FAMILY_NOT_ESTABLISHED = "family not established"


def approve_families(proteins: list[Protein], issues: list[QCIssue]) -> list[dict]:
    """One row per protein family: {family, food_source, protein_ids, approved, reason}.

    A family is approved when none of its proteins has a QC issue, warnings included. Proteins with no family on record (not in the library) are one row,
    "family not established", and are never approved: MMseqs2 clusters do not give them a family."""
    issues_of: dict[str, list[str]] = {}
    for i in issues:
        issues_of.setdefault(i.substrate_id, []).append(i.code)
    groups: dict[tuple[str, str | None], list[Protein]] = {}
    for p in proteins:
        groups.setdefault((p.food_source, p.protein_family) if p.protein_family else ("", None), []).append(p)
    rows = []
    for (source, family), members in groups.items():
        found = {p.substrate_id: issues_of[p.substrate_id] for p in members if p.substrate_id in issues_of}
        if family is None:
            approved, reason = False, "no family on record for these proteins"
        elif found:
            approved, reason = False, "QC issues: " + "; ".join(f"{sid} {', '.join(codes)}" for sid, codes in found.items())
        else:
            approved, reason = True, "every protein passed the FASTA checks"
        rows.append({"family": family or FAMILY_NOT_ESTABLISHED, "food_source": source, "protein_ids": [p.substrate_id for p in members], "approved": approved, "reason": reason})
    return rows

