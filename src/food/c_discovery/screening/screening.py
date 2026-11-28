"""Screening functions: validation, physicochemical properties, liabilities, digestion support, similarity helpers."""
from __future__ import annotations

import re

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from food.shared import liabilities

from .. import settings

# ------------------------------------------------------------------ validation / exclusion


def exclusion_reason(seq: str) -> str:
    """Empty string if the sequence passes the hard rules, otherwise the rule it broke."""
    if not seq or any(a not in settings.STANDARD_AA for a in seq):
        return "non_standard_residue"
    if len(seq) < settings.MIN_LENGTH:
        return "too_short"
    if len(seq) > settings.MAX_LENGTH:
        return "too_long"
    return ""


# ------------------------------------------------------------------ physicochemical properties and liabilities

_LIABILITY_PATTERNS = liabilities.PATTERNS  # shared with the dossier, which also needs the positions


def physchem(seq: str) -> dict:
    pa = ProteinAnalysis(seq)
    hydrophobic_fraction = sum(a in settings.HYDROPHOBIC for a in seq) / len(seq)
    liabilities = [name for name, rx in _LIABILITY_PATTERNS.items() if rx.search(seq)]
    for aa, name in (("C", "cysteine"), ("M", "methionine_oxidation"), ("W", "tryptophan_oxidation")):
        if aa in seq:
            liabilities.append(name)
    if hydrophobic_fraction > settings.HYDROPHOBIC_FRACTION_LIMIT:
        liabilities.append("hydrophobic_solubility_risk")
    if len(seq) > settings.LONG_PEPTIDE:
        liabilities.append("long_for_synthesis")
    score = 1.0 - min(1.0, len(liabilities) / settings.LIABILITY_CAP)
    return {
        "net_charge_pH7_4": round(pa.charge_at_pH(7.4), 2),
        "isoelectric_point": round(pa.isoelectric_point(), 2),
        "gravy": round(pa.gravy(), 3),
        "hydrophobic_fraction": round(hydrophobic_fraction, 3),
        "aromaticity": round(pa.aromaticity(), 3),
        "liabilities": ";".join(liabilities),
        "n_liabilities": len(liabilities),
        "developability": round(score, 3),
    }


# ------------------------------------------------------------------ digestion support (from step 2)

def digestion_support(row: dict) -> float:
    w = settings.DIGESTION_WEIGHTS
    limit = 1.0 if row.get("is_limit_product_any") == "True" else 0.0
    clusters = min(int(row.get("n_parent_clusters") or 0), settings.PARENT_CLUSTER_CAP) / settings.PARENT_CLUSTER_CAP
    return round(w["limit_product"] * limit + w["parent_clusters"] * clusters, 3)


# ------------------------------------------------------------------ similarity helpers

def kmers(seq: str, k: int = 3) -> frozenset:
    return frozenset(seq[i:i + k] for i in range(len(seq) - k + 1)) or frozenset([seq])


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b)


# ------------------------------------------------------------------ model results -> objectives

def stability_scores(sites: dict[str, list[int]], length: int) -> dict:
    """Objectives from the protease rules: higher = more stable. `sites` is {enzyme: 1-based P1 positions of its internal cuts}.
    uncut_bond_fraction = share of the peptide's internal bonds that none of the three proteases cuts. It is not a measure of survival in the gut.
    `cleavage_positions` is {enzyme: [1-based P1 positions]}: position p means the bond between residue p and residue p + 1 is cut."""
    out: dict = {f"cleave_{e}_sites": len(s) for e, s in sites.items()}
    out["cleavage_positions"] = {e: sorted(s) for e, s in sites.items()}
    cut = set().union(*(sites[e] for e in settings.GUT_ENZYMES if e in sites))
    out["uncut_bond_fraction"] = round(1 - len(cut) / (length - 1), 3) if length > 1 and any(e in sites for e in settings.GUT_ENZYMES) else np.nan
    return out
