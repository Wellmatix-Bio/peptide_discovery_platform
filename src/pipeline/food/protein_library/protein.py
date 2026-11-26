"""The food protein record and its loader. Fields come from the Phase 1 data foundation: the substrate FASTA and `source_records.csv`."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .fasta import QCIssue, Substrate, read_fasta


@dataclass
class Protein(Substrate):
    """A `Substrate` plus the provenance columns of `source_records.csv`; None means not recorded, never a default."""

    version: str | None = None  # UniProt sequence_version
    checksum: str | None = None  # sha256 of the complete sequence
    organism: str | None = None
    taxon_id: str | None = None
    protein_family: str | None = None
    allergen_source: bool | None = None  # allergen_source_flag: True / False, None when the table has no row
    retrieved_at: str | None = None
    uniprot_release: str | None = None
    cluster: str | None = None  # set by clusters.assign_clusters
    cluster_source: str | None = None  # where `cluster` came from: family, sequence or protein_id (clusters.py)


def _read_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_proteins(
    fasta: str | Path,
    source_records: str | Path | None = None,
) -> tuple[list[Protein], list[QCIssue]]:
    """Proteins from a substrate FASTA, joined by accession to the source records when given.

    A record with no row in the source records keeps allergen_source=None and gets a warning; nothing is guessed."""
    subs, issues = read_fasta(fasta)
    records = {r["accession"]: r for r in _read_rows(source_records)} if source_records else {}
    out: list[Protein] = []
    for s in subs:
        r = records.get(s.accession)
        if source_records and r is None:
            issues.append(QCIssue(s.substrate_id, "warning", "NO_SOURCE_RECORD",
                                  f"accession {s.accession} has no row in {Path(source_records).name}; allergen source and provenance are unknown"))
        flag = {"True": True, "False": False}.get(r["allergen_source_flag"]) if r else None
        out.append(Protein(**vars(s), version=r and r["sequence_version"], checksum=r and r["checksum"], organism=r and r["organism"],
                           taxon_id=r and r["taxon_id"], protein_family=r and r["protein_family"], allergen_source=flag,
                           retrieved_at=r and r["retrieved_at"], uniprot_release=r and r["uniprot_release"]))
    return out, issues
