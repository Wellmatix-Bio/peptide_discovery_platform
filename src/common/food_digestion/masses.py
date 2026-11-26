"""Peptide masses: monoisotopic from a residue table, average with modlAMP's formula, neither needing a dependency."""
from __future__ import annotations

# Monoisotopic residue masses (Da) and water, to the precision pyteomics uses, so masses match it exactly.
_WATER = 18.0105646837
_MONO = {
    "G": 57.02146372057, "A": 71.03711378471, "S": 87.03202840427, "P": 97.05276384885,
    "V": 99.06841391299, "T": 101.04767846841, "C": 103.00918478471, "L": 113.08406397713,
    "I": 113.08406397713, "N": 114.04292744114, "D": 115.02694302383, "Q": 128.05857750528,
    "K": 128.09496301400, "E": 129.04259308797, "M": 131.04048491299, "H": 137.05891185845,
    "F": 147.06841391299, "R": 156.10111102360, "Y": 163.06332853255, "W": 186.07931294986,
}

# Average masses of the free amino acids (g/mol) as modlAMP's `aa_weights()` lists them; modlAMP's molecular weight is their sum minus one water (18.015) per peptide bond, to 2 decimals.
# tests/unit/pipeline/test_food_digestion.py checks this against modlAMP itself.
_AVERAGE = {
    "A": 89.093, "C": 121.158, "D": 133.103, "E": 147.129, "F": 165.189, "G": 75.067, "H": 155.155, "I": 131.173, "K": 146.188, "L": 131.173,
    "M": 149.211, "N": 132.118, "P": 115.131, "Q": 146.145, "R": 174.2, "S": 105.093, "T": 119.119, "V": 117.146, "W": 204.225, "Y": 181.189,
}


def mono_mass(seq: str) -> float:
    """Neutral monoisotopic mass of the free peptide; raises KeyError on a non-standard residue."""
    return sum(_MONO[a] for a in seq) + _WATER


def avg_mass(seq: str) -> float:
    """Average molecular weight, free termini; raises KeyError on a non-standard residue."""
    return round(sum(_AVERAGE[a] for a in seq) - 18.015 * (len(seq) - 1), 2)


def chemistry_notes(seq: str) -> str:
    """The note the digestion engine writes for a peptide: linear, free termini, no modifications, and what is not modelled."""
    return ("linear; free N- and C-termini; no modifications" + ("; contains Cys (disulfide/oxidation not modelled)" if "C" in seq else "")
            + ("; contains Met (oxidation not modelled)" if "M" in seq else ""))


def avg_masses(seqs: list[str]) -> list[float]:
    """Average molecular weights (modlAMP's formula), free termini."""
    return [avg_mass(s) for s in seqs]
