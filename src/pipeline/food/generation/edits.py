"""Conservative edit operations on a template fragment, optionally steered toward a known positive peptide."""
from __future__ import annotations

from itertools import combinations

from Bio.Align import substitution_matrices

from . import settings as G

AA = "ACDEFGHIKLMNPQRSTVWY"
_B62 = substitution_matrices.load("BLOSUM62")
CONSERVATIVE = {a: [b for b in AA if b != a and _B62[a][b] >= G.CONSERVATIVE_MIN_BLOSUM] for a in AA}


def guided_positions(aligner, template: str, reference: str) -> list[tuple[int, str]]:
    """(0-based template position, reference residue) where the best local alignment of the template to the
    reference shows a mismatch: substituting it moves the template toward the reference."""
    try:
        aln = aligner.align(template, reference)[0]
    except (IndexError, ValueError):
        return []
    out = []
    for (t0, t1), (r0, _r1) in zip(*aln.aligned):
        for k in range(t1 - t0):
            i, j = t0 + k, r0 + k
            if template[i] != reference[j] and reference[j] in AA:
                out.append((i, reference[j]))
    return out


def single_edits(template: str, guided: list[tuple[int, str]]) -> list[tuple]:
    """('sub', pos, old, new, kind) and ('del', pos, old, '', 'deletion'); positions are 0-based."""
    edits, seen = [], set()
    for i, new in guided:
        edits.append(("sub", i, template[i], new, "guided"))
        seen.add((i, new))
    for i, old in enumerate(template):
        for new in CONSERVATIVE[old]:
            if (i, new) not in seen:
                edits.append(("sub", i, old, new, "conservative"))
    if G.ALLOW_DELETION and len(template) - 1 >= G.MIN_LENGTH:
        edits += [("del", i, old, "", "deletion") for i, old in enumerate(template)]
    return edits


def apply_edits(template: str, edits: tuple) -> str:
    seq = list(template)
    for e in edits:
        if e[0] == "sub":
            seq[e[1]] = e[3]
    for e in sorted((e for e in edits if e[0] == "del"), key=lambda e: -e[1]):
        del seq[e[1]]
    return "".join(seq)


def describe(edits: tuple) -> str:
    return ";".join(f"{e[2]}{e[1] + 1}>{e[3] or 'del'}" for e in sorted(edits, key=lambda e: e[1]))


def enumerate_variants(template: str, guided: list[tuple[int, str]]) -> dict[str, tuple]:
    """All variants with <= MAX_EDITS edits (at most one deletion, no two edits at one position).
    Returns {variant sequence: edits}; the variant with the fewest edits (guided first) is kept per sequence."""
    singles = single_edits(template, guided)
    out: dict[str, tuple] = {}
    for n in range(1, G.MAX_EDITS + 1):
        for combo in combinations(singles, n):
            pos = [e[1] for e in combo]
            if len(set(pos)) < n or sum(e[0] == "del" for e in combo) > 1:
                continue
            seq = apply_edits(template, combo)
            if seq != template and seq not in out:
                out[seq] = combo
    return out
