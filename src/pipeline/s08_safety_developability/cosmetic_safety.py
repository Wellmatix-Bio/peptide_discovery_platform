# Cosmetic Stage 8 rules: irritation proxy and tiers, penetration by depth, cytotoxicity cell-type resolution, skin-protease choice.
from __future__ import annotations

import difflib
from typing import Any

import numpy as np

from pipeline.s05_physchem_screening.cosmetic_physchem import compute_properties, has_lipid

TIERS = ("low", "medium", "high")
#: Tiers rejected per safety level.
REJECT_TIERS = {"standard": {"high"}, "strict": {"medium", "high"}}
DEPTHS = ("outer_layer", "epidermis", "dermis")


def irritation_score(hydrophobic_moment: float, net_charge: float) -> float:
    """Hydrophobic moment x positive net charge; a peptide with no positive charge scores 0."""
    return float(hydrophobic_moment) * max(float(net_charge), 0.0)


def irritation_cutoffs(
    seed_set: list[dict[str, Any]], ph: float, medium_percentile: float, high_percentile: float
) -> dict[str, Any] | None:
    """Seed-set percentiles of the irritation score; None when there are fewer than two seeds to form them."""
    if len(seed_set) < 2:
        return None
    scores = []
    for seed in seed_set:
        props = compute_properties(seed["sequence"], ph, seed.get("modifications"))
        scores.append(irritation_score(props["hydrophobic_moment"], props["net_charge"]))
    medium, high = np.percentile(scores, [medium_percentile, high_percentile])
    return {"n_seeds": len(seed_set), "medium_min": float(medium), "high_min": float(high)}


def irritation_tier(score: float, cutoffs: dict[str, Any], lipid: bool) -> str:
    """low below the medium percentile, medium up to the high percentile, high above it; a lipid tail moves up one tier.

    A score of 0 (no positive charge or no moment) is always low: it cannot exceed seeds whose percentile is also 0.
    """
    if score <= 0:
        tier = 0
    elif score < cutoffs["medium_min"]:
        tier = 0
    elif score <= cutoffs["high_min"]:
        tier = 1
    else:
        tier = 2
    if lipid:
        tier = min(tier + 1, len(TIERS) - 1)
    return TIERS[tier]


def penetration(
    target_depth: str, molecular_weight: float, modifications: dict[str, Any] | None, epidermis_mw_max: float
) -> dict[str, Any]:
    """Depth rule: outer_layer none; epidermis needs MW <= max or a lipid tail; dermis needs a lipid tail."""
    lipid = has_lipid(modifications)
    if target_depth == "outer_layer":
        return {"depth": target_depth, "rule": "none", "passed": True, "lipid_tail": lipid}
    if target_depth == "epidermis":
        passed = molecular_weight <= epidermis_mw_max or lipid
        rule = f"molecular_weight <= {epidermis_mw_max:g} Da or lipid tail"
    elif target_depth == "dermis":
        passed = lipid
        rule = "lipid tail"
    else:
        raise ValueError(f"unsupported target_depth {target_depth!r}; expected one of {DEPTHS}")
    return {
        "depth": target_depth,
        "rule": rule,
        "passed": passed,
        "molecular_weight": molecular_weight,
        "lipid_tail": lipid,
    }


def resolve_cell_type(requested: str, vocabulary: list[str]) -> dict[str, Any]:
    """The model's label for a requested cell type: an exact match, else the closest supported label (recorded)."""
    for label in vocabulary:
        if label.casefold() == requested.casefold():
            return {"requested": requested, "used": label, "exact": True}
    closest = difflib.get_close_matches(requested, vocabulary, n=1, cutoff=0.0)
    return {"requested": requested, "used": closest[0], "exact": False}


def skin_proteases(
    target_depth: str, skin_accessions: list[str], dermis_extra_accessions: list[str]
) -> list[str]:
    """Enzymes to screen: the epidermal set, plus the dermal extras for dermis."""
    if target_depth == "dermis":
        return list(dict.fromkeys([*skin_accessions, *dermis_extra_accessions]))
    return list(skin_accessions)
