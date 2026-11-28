"""Checks that report next to the objectives instead of inside them: applicability, prior-art alerts, parent allergen flag."""
from __future__ import annotations

import numpy as np

from .. import settings
from ..evidence import EvidenceLookup
from .model_registry import APPLICABILITY_ORDER

_NAME = {v: k for k, v in APPLICABILITY_ORDER.items()}


# ------------------------------------------------------------------ applicability
def evidence_applicability(c: dict) -> str:
    """Is the peptide close enough to the labelled evidence for the lookup to say anything?"""
    if c.get("evidence_match") in ("exact", "contains", "near"):
        return "in_domain"
    return "limited" if c.get("evidence_similarity", 0.0) >= settings.LIMITED_SIMILARITY else "out_of_domain"


def stability_applicability(c: dict) -> str:
    """Protease rules: not_run when no rule was applied; otherwise limited, because a rule site is a hypothesis (no abundance, kinetics or exposure)."""
    gut = c.get("uncut_bond_fraction", np.nan)
    if gut is None or (isinstance(gut, float) and np.isnan(gut)):
        return "not_run"
    return "limited"


def coverage(dims: dict[str, str]) -> float:
    """0-1: share of the assessment that applies. Mean of settings.COVERAGE_WEIGHTS over every dimension, so a dimension that did not run
    or does not apply counts as zero instead of being skipped."""
    if not dims:
        return 0.0
    return round(sum(settings.COVERAGE_WEIGHTS.get(v, 0.0) for v in dims.values()) / len(dims), 3)


def combine_objective(scored: list[tuple[str, float, str]], base: float | None = None) -> tuple[float, str, str]:
    """Worst score over the models that may count for this peptide, without letting an out-of-domain model veto it.

    scored: (model, score, applicability). A model outside settings.SCORING_APPLICABILITY is left out of the score and listed as flagged.
    base: an always-applicable rule score (developability rule). Returns (score, basis, flagged_models); basis is the weakest
    applicability that fed the score, or "out_of_domain_only" when no model qualified and every score was used.
    """
    use = [(m, s, a) for m, s, a in scored if a in settings.SCORING_APPLICABILITY]
    flagged = [f"{m}:{a}" for m, s, a in scored if a not in settings.SCORING_APPLICABILITY]
    if use:
        vals = [s for _, s, _ in use] + ([base] if base is not None else [])
        return min(vals), worst([a for _, _, a in use]), ";".join(flagged)
    vals = [s for _, s, _ in scored] + ([base] if base is not None else [])
    if not vals:
        return float("nan"), "none", ""
    return min(vals), ("out_of_domain_only" if scored else "rule_only"), ";".join(flagged)


def worst(parts: list[str]) -> str:
    """Worst applicability across the dimensions that ran. Never averaged."""
    ranked = [APPLICABILITY_ORDER[p] for p in parts if p in APPLICABILITY_ORDER]
    return _NAME[max(ranked)] if ranked else "not_run"


# ------------------------------------------------------------------ prior-art alerts (not a legal freedom-to-operate opinion)
class PriorArt:
    """Flags a candidate that is exact, contained in, or close to a peptide whose evidence comes from a patent."""

    def __init__(self, evidence: list[dict]):
        pat = [e for e in evidence if any(t.strip().startswith(settings.PATENT_PREFIXES) for t in e["study"].replace("/", ";").split(";"))]
        self.n = len(pat)
        self.lookup = EvidenceLookup(pat) if pat else None

    def alert(self, seq: str) -> str:
        if not self.lookup:
            return ""
        r = self.lookup.lookup(seq)
        if r["evidence_match"] == "none":
            return ""
        return f"{r['evidence_match']} match to patent-sourced peptide {r['evidence_peptide']} ({r['evidence_study']})"


# ------------------------------------------------------------------ parent allergen flag
def parent_allergen_flag(any_parent_allergen_source: bool | None) -> str:
    """From the parents' allergen-source flags in the protein library; never guessed: an unassessed parent is "not_available"."""
    if any_parent_allergen_source is None:
        return "not_available"
    return "allergen_source" if any_parent_allergen_source else "no_flag"
