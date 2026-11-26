"""FASTA reading with QC for the food protein library; standard library only, so the API can use it.

Understands the header written by ``extracted_processing.py``::

    >ACC:start-end KIND=chain [NAME=...] FS=Whey LEN=162 SHA256=...

and falls back to plain FASTA (``>id description``) for any user upload; the
first whitespace-delimited token is the substrate ID.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


STANDARD = frozenset("ACDEFGHIKLMNPQRSTVWY")

_ID_RANGE = re.compile(r"^([A-Za-z0-9_.\-|]+?):(\d+)-(\d+)$")
_KV = re.compile(r"(\w+)=(\S+)")


#: What each end code of a library chain means (the Phase 1 data foundation's codes; the header carries NTERM= and CTERM=).
TERMINUS_MEANING = {
    "SIGNAL_CLEAVED": "a signal peptide was cut off at this end",
    "PROPEP_CLEAVED": "a propeptide was cut off at this end",
    "TRANSIT_CLEAVED": "a transit peptide was cut off at this end",
    "INIT_MET_CLEAVED": "the starting methionine was cut off at this end",
    "CHAIN_JUNCTION": "this end meets another chain directly, with nothing removed between them",
    "SEQ_START": "first residue of the UniProt sequence; the protein's real start may lie before it",
    "SEQ_END": "last residue of the UniProt sequence; the protein's real end may lie beyond it",
    "UNANNOTATED": "UniProt has no annotation explaining this end",
}


@dataclass
class Substrate:
    substrate_id: str          # e.g. P02754:17-178 (unique per input)
    accession: str
    precursor_start: int       # 1-based precursor coordinate of residue 1 of the sequence
    precursor_end: int
    sequence: str
    kind: str = "unknown"
    food_source: str = ""
    name: str = ""
    header: str = ""
    also: str = ""             # other accessions carrying the identical sequence
    n_terminus: str = ""       # NTERM= in the header (a TERMINUS_MEANING code); "" when the header has none
    c_terminus: str = ""       # CTERM= in the header


@dataclass
class QCIssue:
    substrate_id: str
    severity: str   # "error" | "warning"
    code: str
    message: str


def read_fasta(path: str | Path) -> tuple[list[Substrate], list[QCIssue]]:
    """Return usable substrates and QC issues. Never silently drops a record:
    every skipped record has an issue explaining why and how to fix it."""
    text = Path(path).read_text(encoding="utf-8")
    records: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            records.append((line[1:], []))
        elif records:
            records[-1][1].append(line)
        else:
            raise ValueError(f"{path}: sequence data before the first '>' header; not a FASTA file")
    if not records:
        raise ValueError(f"{path}: no FASTA records found")

    substrates: list[Substrate] = []
    issues: list[QCIssue] = []
    seen: set[str] = set()
    for header, parts in records:
        token = header.split()[0] if header.split() else ""
        seq = "".join(parts).upper().replace("*", "")
        if not token:
            issues.append(QCIssue("(empty id)", "error", "EMPTY_ID", "header has no identifier; add an ID after '>'"))
            continue
        m = _ID_RANGE.match(token)
        if m:
            acc, s, e = m.group(1), int(m.group(2)), int(m.group(3))
        else:
            acc, s, e = token, 1, len(seq)
        sid = token if m else f"{token}:1-{len(seq)}"
        if sid in seen:
            issues.append(QCIssue(sid, "error", "DUPLICATE_ID",
                                  f"identifier '{sid}' appears more than once; rename one record"))
            continue
        seen.add(sid)
        if not seq:
            issues.append(QCIssue(sid, "error", "EMPTY_SEQUENCE", "record has no sequence"))
            continue
        bad = sorted(set(seq) - STANDARD)
        if bad:
            issues.append(QCIssue(sid, "error", "NONSTANDARD_RESIDUES",
                                  f"non-standard residues {bad}; record skipped. Replace/remove them "
                                  f"(ambiguous X/B/Z/J cannot be digested reliably) and re-run"))
            continue
        if m and (e - s + 1) != len(seq):
            issues.append(QCIssue(sid, "warning", "RANGE_LENGTH_MISMATCH",
                                  f"header range {s}-{e} implies {e - s + 1} residues but sequence has "
                                  f"{len(seq)}; precursor coordinates may be wrong"))
        kv = dict(_KV.findall(header))
        substrates.append(Substrate(sid, acc, s, e, seq, kind=kv.get("KIND", "unknown"),
                                    food_source=kv.get("FS", ""), name=kv.get("NAME", ""), also=kv.get("ALSO", ""), n_terminus=kv.get("NTERM", ""), c_terminus=kv.get("CTERM", ""),
                                    header=header))
    return substrates, issues
