"""Sequence similarity to a model's reference (training) set: "is this peptide like anything the model learned from?"

For a query peptide the best match in the reference set is found in two steps: the 30 reference peptides with the closest
2-mer composition (cosine) are shortlisted, then each is globally aligned to the query (BLOSUM62, gap open -10, extend -1) and the
highest identity wins. The score is identities / query length, so a peptide that
sits inside a longer reference peptide at 100% identity scores 1.0. The shortlist makes this an approximation: the true
best match can be missed, which can only lower the score; calibration uses the same procedure, so the bands are consistent.

Bands come from scripts/add_similarity_profiles.py, which scores training peptides against reference peptides from the OTHER
cross-validation folds (CD-HIT clusters never span folds, so near-duplicates are excluded) and sets "distant" at the
10th and "similar" at the 50th percentile of that distribution.
"""

from __future__ import annotations

import gzip
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

AMINO = "ACDEFGHIKLMNPQRSTVWY"
_PAIR = {a + b: i for i, (a, b) in enumerate(product(AMINO, repeat=2))}
SHORTLIST = 30
REFERENCE_FILE = "reference.txt.gz"


def composition(sequences: Sequence[str]) -> np.ndarray:
    """L2-normalised 2-mer count vectors, one row per sequence."""
    out = np.zeros((len(sequences), len(_PAIR)), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for i in range(len(sequence) - 1):
            index = _PAIR.get(sequence[i : i + 2])
            if index is not None:
                out[row, index] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def _aligner():
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -1.0
    return aligner


def best_identities(
    queries: Sequence[str],
    reference: Sequence[str],
    reference_matrix: np.ndarray | None = None,
    exclude: Iterable[np.ndarray] | None = None,
) -> np.ndarray:
    """Best identity (0-1, over the query's length) of each query to `reference`.

    `exclude`, when given, is one boolean array per query marking reference peptides that must not be matched (used to
    leave out a query's own cross-validation fold when calibrating).
    """
    if reference_matrix is None:
        reference_matrix = composition(reference)
    aligner = _aligner()
    exclusions = list(exclude) if exclude is not None else None
    result = np.zeros(len(queries))
    for start in range(0, len(queries), 128):
        chunk = list(queries[start : start + 128])
        scores = composition(chunk) @ reference_matrix.T
        for offset, query in enumerate(chunk):
            row = scores[offset]
            if exclusions is not None:
                row = np.where(exclusions[start + offset], -np.inf, row)
            size = min(SHORTLIST, int(np.isfinite(row).sum()))
            if size == 0:
                result[start + offset] = np.nan
                continue
            shortlist = np.argpartition(-row, size - 1)[:size]
            best = 0.0
            for index in shortlist:
                identities = aligner.align(reference[index], query)[0].counts().identities
                best = max(best, identities / len(query))
            result[start + offset] = best
    return result


class SimilarityIndex:
    """A model's reference peptides plus the identity cut-offs for its three bands."""

    def __init__(self, reference: list[str], similar_at: float, distant_below: float) -> None:
        self.reference = reference
        self.matrix = composition(reference)
        self.similar_at = similar_at
        self.distant_below = distant_below

    @classmethod
    def load(cls, weights_dir: Path, info: dict[str, Any]) -> "SimilarityIndex | None":
        """None when this model has no similarity profile (older weights): the check is then reported as not run."""
        block = info.get("similarity")
        path = Path(weights_dir) / REFERENCE_FILE
        if not block or not path.is_file():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            reference = [line.strip() for line in handle if line.strip()]
        return cls(reference, float(block["similar_at"]), float(block["distant_below"]))

    def assess(self, sequences: Sequence[str]) -> list[dict[str, Any]]:
        """Per sequence: best identity and its band (`similar`, `moderate` or `distant`)."""
        identities = best_identities(sequences, self.reference, self.matrix)
        return [
            {
                "best_identity": round(float(value), 3),
                "verdict": "similar" if value >= self.similar_at else "distant" if value < self.distant_below else "moderate",
                "similar_at": self.similar_at,
                "distant_below": self.distant_below,
            }
            for value in identities
        ]
