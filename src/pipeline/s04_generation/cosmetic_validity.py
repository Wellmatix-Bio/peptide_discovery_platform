# Cosmetic Stage 4 validity check: alphabet, chemical-modification placement and length; any failure rejects the candidate.
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from schemas.candidate import Candidate

AMINO_ACID_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")

N_TERMINUS, C_TERMINUS = "n_terminus", "c_terminus"


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _position(spec: Any) -> Any:
    """Position of a modification given as a bare flag/position or as {"position": ...}; None for a plain True."""
    if isinstance(spec, dict):
        return spec.get("position")
    return None if spec is True else spec


def _terminal_reasons(
    name: str, spec: Any, terminus: str, terminal_index: int
) -> list[str]:
    """N-acetyl and C-amide sit only on their own terminus, so any other position is a violation."""
    position = _position(spec)
    if position is None or position == terminus or position == terminal_index:
        return []
    return [f"{name}_not_at_{terminus}"]


def validate_sequence_and_modifications(
    sequence: str | None,
    modifications: dict[str, Any] | None,
    min_length: int,
    max_length: int,
) -> list[str]:
    """Reasons the peptide is invalid, empty when valid.

    Modifications use the `Candidate.predictions["modifications"]` keys shared with Stage 9: `n_acetyl` and `c_amide`
    (True or {"position": ...}), `lipid_tail` ({"position": "n_terminus" | residue index}) and `d_amino_acid`
    (a list of residue indices). Other keys are not this stage's concern and are ignored.
    """
    if not sequence:
        return ["empty_sequence"]
    reasons: list[str] = []
    if not set(sequence) <= AMINO_ACID_SET:
        reasons.append("non_standard_alphabet")
    if not min_length <= len(sequence) <= max_length:
        reasons.append("length_out_of_range")

    modifications = modifications or {}
    last = len(sequence) - 1
    lipid = modifications.get("lipid_tail")
    lipid_position = _position(lipid) if lipid else None

    if modifications.get("n_acetyl"):
        reasons += _terminal_reasons(
            "n_acetyl", modifications["n_acetyl"], N_TERMINUS, 0
        )
        # An N-terminal lipid occupies the amine, so the N-terminus is no longer free to acetylate.
        if lipid and lipid_position in (N_TERMINUS, 0):
            reasons.append("n_acetyl_n_terminus_not_free")
    if modifications.get("c_amide"):
        reasons += _terminal_reasons(
            "c_amide", modifications["c_amide"], C_TERMINUS, last
        )
    if lipid:
        if lipid_position == N_TERMINUS:
            pass
        elif not _is_index(lipid_position) or not 0 <= lipid_position <= last:
            reasons.append("lipid_tail_position_invalid")
        elif sequence[lipid_position] != "K":
            reasons.append("lipid_tail_not_on_lys")
    d_positions = modifications.get("d_amino_acid") or []
    if d_positions is True:
        d_positions = []
    if any(not _is_index(i) or not 0 <= i <= last for i in d_positions):
        reasons.append("d_amino_acid_position_out_of_range")
    return reasons


def validity_filter(
    candidates: Iterable[Candidate], min_length: int, max_length: int
) -> tuple[list[Candidate], Counter]:
    """Keep the valid candidates and count the rejection reasons across the rest."""
    kept: list[Candidate] = []
    rejected: Counter = Counter()
    for candidate in candidates:
        reasons = validate_sequence_and_modifications(
            candidate.sequence,
            candidate.predictions.get("modifications"),
            min_length,
            max_length,
        )
        if reasons:
            rejected.update(reasons)
        else:
            kept.append(candidate)
    return kept, rejected
