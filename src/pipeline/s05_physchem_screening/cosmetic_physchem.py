# Cosmetic Stage 5 rules: modification-aware properties (B), rule rejects (C) and rule flags (D); the models (E) live in the stage.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from Bio.SeqUtils.IsoelectricPoint import (
    negative_pKs,
    pKcterminal,
    pKnterminal,
    positive_pKs,
)
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import PeptideDescriptor

N_TERMINUS = "n_terminus"
#: Average-mass change of acetylating the N-terminus (+C2H2O) and of amidating the C-terminus (OH -> NH2).
ACETYL_DELTA_DA = 42.037
AMIDE_DELTA_DA = -0.985

LOWER_RISK_DEAMIDATION = ("NS", "NH")
LOWER_RISK_HYDROLYSIS = ("DG",)


@dataclass(frozen=True)
class Thresholds:
    """Rule cutoffs, built from the stage params (Stage5CosmeticParams), which hold the defaults."""

    molecular_weight_reject_max: float
    net_charge_reject_min: float
    net_charge_reject_max: float
    hydrophobicity_percentile_low: float
    hydrophobicity_percentile_high: float
    moment_percentile: float
    membrane_active_min_charge: float
    repeat_run_length: int
    molecular_weight_flag_min: float
    molecular_weight_flag_max: float
    flag_lower_risk_motifs: bool

    @classmethod
    def from_params(cls, params: Any) -> "Thresholds":
        return cls(**{name: getattr(params, name) for name in cls.__dataclass_fields__})


def lipid_position(modifications: dict[str, Any] | None) -> Any:
    """Position of the lipid tail ("n_terminus" or a residue index), None when there is none."""
    lipid = (modifications or {}).get("lipid_tail")
    if not lipid:
        return None
    return lipid.get("position") if isinstance(lipid, dict) else None


def has_lipid(modifications: dict[str, Any] | None) -> bool:
    return bool((modifications or {}).get("lipid_tail"))


def n_terminus_blocked(modifications: dict[str, Any] | None) -> bool:
    """The alpha-amine is acylated, by N-acetyl or by an N-terminal lipid tail."""
    mods = modifications or {}
    return bool(mods.get("n_acetyl")) or lipid_position(mods) in (N_TERMINUS, 0)


def net_charge(sequence: str, ph: float, modifications: dict[str, Any] | None = None) -> float:
    """Henderson-Hasselbalch charge; a blocked terminus or an acylated Lys contributes no charge."""
    mods = modifications or {}
    positive = [positive_pKs[a] for i, a in enumerate(sequence) if a in "KRH" and i != lipid_position(mods)]
    negative = [negative_pKs[a] for a in sequence if a in "DECY"]
    if not n_terminus_blocked(mods):
        positive.append(pKnterminal.get(sequence[0], positive_pKs["Nterm"]))
    if not mods.get("c_amide"):
        negative.append(pKcterminal.get(sequence[-1], negative_pKs["Cterm"]))
    return float(
        sum(1.0 / (1.0 + 10 ** (ph - pk)) for pk in positive)
        - sum(1.0 / (1.0 + 10 ** (pk - ph)) for pk in negative)
    )


def isoelectric_point(sequence: str, modifications: dict[str, Any] | None = None) -> float | None:
    """pH of zero net charge by bisection; None when the peptide carries no ionisable group at all."""
    low, high = 0.0, 14.0
    if abs(net_charge(sequence, low, modifications)) < 1e-9 and abs(net_charge(sequence, high, modifications)) < 1e-9:
        return None
    for _ in range(60):
        mid = (low + high) / 2
        if net_charge(sequence, mid, modifications) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def molecular_weight(sequence: str, modifications: dict[str, Any] | None = None) -> float:
    """Average mass in Da; N-acetyl, C-amide and a lipid tail that states its `mass` shift it."""
    mods = modifications or {}
    mass = float(ProteinAnalysis(sequence).molecular_weight())
    if mods.get("n_acetyl"):
        mass += ACETYL_DELTA_DA
    if mods.get("c_amide"):
        mass += AMIDE_DELTA_DA
    lipid = mods.get("lipid_tail")
    if isinstance(lipid, dict) and lipid.get("mass") is not None:
        mass += float(lipid["mass"])
    return mass


def eisenberg(sequence: str) -> tuple[float, float]:
    """Mean Eisenberg hydrophobicity and hydrophobic moment (alpha-helical, 100 deg/residue)."""
    glob = PeptideDescriptor([sequence], "eisenberg")
    glob.calculate_global()
    moment = PeptideDescriptor([sequence], "eisenberg")
    moment.calculate_moment()
    return float(glob.descriptor[0][0]), float(moment.descriptor[0][0])


def compute_properties(
    sequence: str, ph: float, modifications: dict[str, Any] | None = None
) -> dict[str, float | None]:
    hydrophobicity, moment = eisenberg(sequence)
    return {
        "molecular_weight": molecular_weight(sequence, modifications),
        "net_charge": net_charge(sequence, ph, modifications),
        "isoelectric_point": isoelectric_point(sequence, modifications),
        "hydrophobicity": hydrophobicity,
        "hydrophobic_moment": moment,
    }


def seed_reference(
    seed_set: list[dict[str, Any]], ph: float, t: Thresholds
) -> dict[str, Any] | None:
    """Percentile bounds of the seed set; None when there are fewer than two seeds to form an interval."""
    if len(seed_set) < 2:
        return None
    props = [compute_properties(s["sequence"], ph, s.get("modifications")) for s in seed_set]
    hydrophobicity = [p["hydrophobicity"] for p in props]
    low, high = np.percentile(
        hydrophobicity, [t.hydrophobicity_percentile_low, t.hydrophobicity_percentile_high]
    )
    return {
        "n_seeds": len(seed_set),
        "hydrophobicity_low": float(low),
        "hydrophobicity_high": float(high),
        "moment_high": float(np.percentile([p["hydrophobic_moment"] for p in props], t.moment_percentile)),
    }


def rule_rejects(
    sequence: str, props: dict[str, Any], reference: dict[str, Any] | None, t: Thresholds
) -> list[str]:
    reasons: list[str] = []
    if props["molecular_weight"] > t.molecular_weight_reject_max:
        reasons.append("molecular_weight_too_high")
    if not t.net_charge_reject_min <= props["net_charge"] <= t.net_charge_reject_max:
        reasons.append("net_charge_out_of_range")
    if reference is not None:
        if not reference["hydrophobicity_low"] <= props["hydrophobicity"] <= reference["hydrophobicity_high"]:
            reasons.append("hydrophobicity_outside_seed_range")
        if (
            props["hydrophobic_moment"] > reference["moment_high"]
            and props["net_charge"] >= t.membrane_active_min_charge
        ):
            reasons.append("membrane_active_profile")
    if re.search(rf"(.)\1{{{t.repeat_run_length - 1},}}", sequence):
        reasons.append("repeated_residues")
    return reasons


def _motifs(sequence: str, motifs: tuple[str, ...]) -> list[str]:
    return [m for m in motifs if m in sequence]


def rule_flags(
    sequence: str, props: dict[str, Any], modifications: dict[str, Any] | None, t: Thresholds
) -> dict[str, Any]:
    """Flags that never reject, as {flag: detail}."""
    mods = modifications or {}
    flags: dict[str, Any] = {}
    reactive = [a for a in "CMW" if a in sequence]
    if reactive:
        flags["oxidation"] = reactive
    deamidation = _motifs(sequence, ("NG",))
    if t.flag_lower_risk_motifs:
        deamidation += _motifs(sequence, LOWER_RISK_DEAMIDATION)
    if deamidation:
        flags["deamidation"] = deamidation
    hydrolysis = _motifs(sequence, ("DP",))
    if t.flag_lower_risk_motifs:
        hydrolysis += _motifs(sequence, LOWER_RISK_HYDROLYSIS)
    if hydrolysis:
        flags["hydrolysis"] = hydrolysis
    if sequence[0] == "Q" and not n_terminus_blocked(mods):
        flags["terminal_gln"] = True
    cysteines = sequence.count("C")
    if cysteines and (cysteines % 2 == 1 or not mods.get("disulfide_pairing")):
        flags["unpaired_cys"] = cysteines
    if t.molecular_weight_flag_min <= props["molecular_weight"] <= t.molecular_weight_flag_max:
        flags["mw_high"] = props["molecular_weight"]
    return flags
