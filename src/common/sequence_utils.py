# Shared sequence manipulation and validation helpers.
from __future__ import annotations

import re
from dataclasses import dataclass, field

STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MIN_LENGTH = 6
DEFAULT_MAX_LENGTH = 50

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParsedSequence:
    id: str  # header text without ">", or "seq_<n>" when the input had none
    sequence: str  # uppercase, whitespace-free, standard residues only


@dataclass(frozen=True)
class Rejection:
    id: str
    sequence: str  # as normalized (whitespace removed, uppercased), before validation
    reason: str  # "empty" | "invalid_residues" | "too_short" | "too_long" | "duplicate"
    detail: str = ""


@dataclass
class ParseResult:
    accepted: list[ParsedSequence] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    @property
    def sequences(self) -> list[str]:
        return [p.sequence for p in self.accepted]


def parse_sequences(
    text: str,
    *,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> ParseResult:
    """Parse FASTA or one-sequence-per-line text into validated, de-duplicated peptides.

    Any ">" line switches to FASTA mode (records may wrap over lines); otherwise each non-blank line is one sequence.
    Nothing is silently dropped: every rejected record is returned with its reason; the first copy of a duplicate wins.
    """
    return _classify(_records(text), min_length, max_length)


def parse_sequence_items(
    items: list[str],
    *,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> ParseResult:
    """Like parse_sequences for a list whose entries are each a bare sequence or a FASTA record.

    Each entry is read on its own, so a ">" in one entry never turns the others into header-less continuation lines;
    duplicates are still removed across the whole list, and generated ids are seq_<position in the list>.
    """
    records: list[tuple[str, str]] = []
    for position, item in enumerate(items, 1):
        records += _records(item, start=position)
    return _classify(records, min_length, max_length)


def _classify(records: list[tuple[str, str]], min_length: int, max_length: int) -> ParseResult:
    if min_length < 1 or max_length < min_length:
        raise ValueError(f"need 1 <= min_length <= max_length, got {min_length}, {max_length}")

    result = ParseResult()
    seen: set[str] = set()
    for record_id, raw in records:
        seq = _WHITESPACE.sub("", raw).upper()
        if not seq:
            result.rejected.append(Rejection(record_id, seq, "empty"))
            continue
        bad = sorted(set(seq) - STANDARD_RESIDUES)
        if bad:
            result.rejected.append(Rejection(record_id, seq, "invalid_residues", "".join(bad)))
        elif len(seq) < min_length:
            result.rejected.append(Rejection(record_id, seq, "too_short", f"{len(seq)} < {min_length}"))
        elif len(seq) > max_length:
            result.rejected.append(Rejection(record_id, seq, "too_long", f"{len(seq)} > {max_length}"))
        elif seq in seen:
            result.rejected.append(Rejection(record_id, seq, "duplicate"))
        else:
            seen.add(seq)
            result.accepted.append(ParsedSequence(record_id, seq))
    return result


def _records(text: str, start: int = 1) -> list[tuple[str, str]]:
    lines = text.lstrip("﻿").splitlines()
    if not any(line.lstrip().startswith(">") for line in lines):
        stripped = [line.strip() for line in lines]
        return [(f"seq_{n}", line) for n, line in enumerate((s for s in stripped if s), start)]

    records: list[tuple[str, list[str]]] = []
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            records.append((line[1:].strip() or f"seq_{start + len(records)}", []))
        elif line:
            if not records:  # sequence text before the first header
                records.append((f"seq_{start + len(records)}", []))
            records[-1][1].append(line)
    return [(rid, "".join(parts)) for rid, parts in records]
