"""Sequence liabilities of a peptide with the residue positions that cause them. Standard library only, so the dossier can use it without the pipeline.

The patterns are the ones `food/c_discovery/screening/screening.py` counts; a test keeps the two in step."""
from __future__ import annotations

import re

PATTERNS = {
    "n_term_gln": re.compile(r"^Q"),          # pyroglutamate formation
    "asp_gly": re.compile(r"[DN]G"),          # deamidation / isoaspartate
    "asp_pro": re.compile(r"DP"),             # acid-labile bond
    "homopolymer_run": re.compile(r"(.)\1{3,}"),
}
RESIDUE_LIABILITIES = {"cysteine": "C", "methionine_oxidation": "M", "tryptophan_oxidation": "W"}
WHOLE_SEQUENCE = ("hydrophobic_solubility_risk", "long_for_synthesis")

#: What the position means, in a few words (the longer meanings are in food/shared/dossier.py LIABILITY_TEXT).
SHORT = {"n_term_gln": "pyroglutamate formation", "asp_gly": "deamidation or isoaspartate", "asp_pro": "acid-labile Asp-Pro bond", "homopolymer_run": "run of identical residues",
         "cysteine": "cysteine", "methionine_oxidation": "oxidation", "tryptophan_oxidation": "oxidation",
         "hydrophobic_solubility_risk": "high share of hydrophobic residues (solubility risk)", "long_for_synthesis": "long for synthesis"}


def sites(name: str, seq: str) -> list[str]:
    """The residues that cause one liability, as `M4` (residue and 1-based position) or `A2-A5` for a run; empty for a whole-sequence liability."""
    if name in RESIDUE_LIABILITIES:
        return [f"{a}{i}" for i, a in enumerate(seq, 1) if a == RESIDUE_LIABILITIES[name]]
    if name == "n_term_gln":
        return ["Q1"] if PATTERNS[name].search(seq) else []
    if name == "homopolymer_run":
        return [f"{m.group(1)}{m.start() + 1}-{m.group(1)}{m.end()}" for m in PATTERNS[name].finditer(seq)]
    if name in ("asp_gly", "asp_pro"):  # the Asp or Asn that sits in front of the bond, overlapping matches included
        pattern = re.compile(r"(?=([DN]G))" if name == "asp_gly" else r"(?=(DP))")
        return [f"{seq[m.start()]}{m.start() + 1}" for m in pattern.finditer(seq)]
    return []


def liability_sites(names: list[str], seq: str) -> list[dict]:
    """For each recorded liability name: its residue sites and the text `M4: oxidation` (`whole sequence: ...` when no residue is to blame)."""
    out = []
    for name in names:
        found = sites(name, seq)
        what = SHORT.get(name, name.replace("_", " "))
        text = "; ".join(f"{s}: {what}" for s in found) if found else f"whole sequence: {what}" if name in WHOLE_SEQUENCE else f"{what}: position not found"
        out.append({"name": name, "sites": found, "text": text})
    return out
