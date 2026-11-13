"""Per-model applicability: does a model's training data actually cover the peptide it is asked about?

A prediction is only as good as the evidence behind it at that input. Each model carries a `domain` block in its
weights' info.json (built from its training set by scripts/add_domain_profiles.py); ReferenceProfile.assess() turns
a peptide's length into one of three statuses plus the reasons, so the platform can say "limited evidence" instead of
printing a confident-looking number.

Checked today: sequence length against the trained range, reference-data support (positive and negative training
examples at that exact length), and, for models whose weights carry a similarity profile, sequence similarity to the
reference set (common/similarity.py; with_similarity() below folds it into the assessment). NOT checked yet, and reported
as such in every result: the ESM embedding / physicochemical feature distribution (docs/TODO.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

#: A length counts as supported only with at least this many positive AND this many negative training peptides.
#: A judgement call, not a fitted value: it is the smallest per-class sample for which a per-length rate or score is
#: more than a handful of anecdotes. It travels with each model's profile, and docs/TODO.md lists calibrating it
#: against out-of-fold performance by length.
MIN_EXAMPLES_PER_CLASS = 10

#: Checks the status does NOT reflect; listed in every assessment so a result never reads as a full domain check.
NOT_CHECKED = ("sequence_similarity", "embedding_distribution")


class Status(str, Enum):
    IN_DOMAIN = "in_domain"
    LIMITED_EVIDENCE = "limited_evidence"
    OUT_OF_DOMAIN = "out_of_domain"


def domain_profile(lengths: Iterable[int], labels: Iterable[int], min_per_class: int = MIN_EXAMPLES_PER_CLASS) -> dict[str, Any]:
    """The `domain` block for info.json: positive and negative training counts at each peptide length."""
    counts: dict[int, list[int]] = {}
    for length, label in zip(lengths, labels):
        counts.setdefault(int(length), [0, 0])[0 if int(label) == 1 else 1] += 1
    return {
        "min_examples_per_class": int(min_per_class),
        "length_counts": {str(length): {"pos": pos, "neg": neg} for length, (pos, neg) in sorted(counts.items())},
    }


_SIMILARITY_CHECK = {"similar": "pass", "moderate": "weak", "distant": "fail"}


def with_similarity(assessment: dict[str, Any], similarity: dict[str, Any]) -> dict[str, Any]:
    """The assessment plus the sequence-similarity answer. `distant` lowers `in_domain` to `limited_evidence`; nothing raises a status."""
    verdict = similarity["verdict"]
    pct = lambda x: f"{round(100 * x)}%"  # noqa: E731
    reason = {
        "similar": f"its best match in the reference set is {pct(similarity['best_identity'])} identical, at or above the {pct(similarity['similar_at'])} that counts as similar",
        "moderate": f"its best match in the reference set is {pct(similarity['best_identity'])} identical, between the {pct(similarity['distant_below'])} and {pct(similarity['similar_at'])} cut-offs",
        "distant": f"its best match in the reference set is only {pct(similarity['best_identity'])} identical, below the {pct(similarity['distant_below'])} that marks a peptide as distant from what the model learned",
    }[verdict]
    status = assessment["status"]
    if verdict == "distant" and status == Status.IN_DOMAIN.value:
        status = Status.LIMITED_EVIDENCE.value
    return {
        **assessment,
        "status": status,
        "reasons": [*assessment["reasons"], reason[0].upper() + reason[1:]],
        "checks": {**assessment["checks"], "sequence_similarity": _SIMILARITY_CHECK[verdict]},
        "similarity": similarity,
        "not_checked": [c for c in assessment["not_checked"] if c != "sequence_similarity"],
    }


@dataclass(frozen=True)
class ReferenceProfile:
    length_min: int
    length_max: int
    #: length -> (positives, negatives) in the training data; None when the model recorded no counts.
    counts: dict[int, tuple[int, int]] | None = None
    min_per_class: int = MIN_EXAMPLES_PER_CLASS

    @classmethod
    def from_info(cls, info: dict[str, Any]) -> "ReferenceProfile":
        training = info["training"]
        domain = info.get("domain") or {}
        raw = domain.get("length_counts")
        counts = {int(k): (int(v["pos"]), int(v["neg"])) for k, v in raw.items()} if raw else None
        return cls(
            int(training["length_min"]),
            int(training["length_max"]),
            counts,
            int(domain.get("min_examples_per_class", MIN_EXAMPLES_PER_CLASS)),
        )

    def assess(self, length: int) -> dict[str, Any]:
        """{status, reasons, checks, reference, not_checked} for a peptide of `length` residues."""
        low, high = self.length_min, self.length_max
        checks = {"length_range": "pass", "reference_support": "unknown"}
        reference = None
        reasons: list[str] = []

        if not low <= length <= high:
            checks["length_range"] = "fail"
            reasons.append(f"length {length} is outside the {low}-{high} residues this model was trained on")
            return self._result(Status.OUT_OF_DOMAIN, reasons, checks, reference)

        if self.counts is None:
            reasons.append("this model recorded no per-length training counts, so its support at this length cannot be verified")
            return self._result(Status.LIMITED_EVIDENCE, reasons, checks, reference)

        pos, neg = self.counts.get(length, (0, 0))
        reference = {"positives": pos, "negatives": neg, "min_per_class": self.min_per_class}
        if pos + neg == 0:
            checks["reference_support"] = "fail"
            reasons.append(f"no peptide of length {length} in the training data (a gap inside the {low}-{high} range)")
            return self._result(Status.OUT_OF_DOMAIN, reasons, checks, reference)
        if pos < self.min_per_class or neg < self.min_per_class:
            checks["reference_support"] = "fail"
            reasons.append(
                f"length {length} has {pos} positive and {neg} negative training peptides; "
                f"at least {self.min_per_class} of each are needed to count it as supported"
            )
            return self._result(Status.LIMITED_EVIDENCE, reasons, checks, reference)

        checks["reference_support"] = "pass"
        reasons.append(f"length {length} is inside {low}-{high} with {pos} positive and {neg} negative training peptides")
        return self._result(Status.IN_DOMAIN, reasons, checks, reference)

    def status_ranges(self, low: int = 6, high: int = 50) -> dict[str, list[list[int]]]:
        """Each status -> the contiguous [first, last] peptide lengths in low..high that get it."""
        ranges: dict[str, list[list[int]]] = {status.value: [] for status in Status}
        for length in range(low, high + 1):
            spans = ranges[self.assess(length)["status"]]
            if spans and spans[-1][1] == length - 1:
                spans[-1][1] = length
            else:
                spans.append([length, length])
        return ranges

    def _result(self, status: Status, reasons: list[str], checks: dict, reference: dict | None) -> dict[str, Any]:
        return {
            "status": status.value,
            "reasons": reasons,
            "checks": checks,
            "reference": reference,
            "not_checked": list(NOT_CHECKED),
        }
