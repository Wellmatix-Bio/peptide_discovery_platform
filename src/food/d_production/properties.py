"""Theoretical molecular properties of a peptide: linear, free termini, no modifications."""
from __future__ import annotations

from dataclasses import dataclass

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from food.b_digestion.masses import mono_mass

PROTON_MASS = 1.007276466621  # Da
PH = 7.4


@dataclass(frozen=True)
class MolecularProperties:
    sequence: str
    length: int
    monoisotopic_mass: float  # Da: residue masses + water
    net_charge_ph7_4: float  # Henderson-Hasselbalch (Biopython ProtParam: pKa of the termini and side chains)
    gravy: float  # Kyte-Doolittle, mean over the residues
    aromatic_count: int  # F, W, Y
    aromatic_fraction: float
    cysteine_count: int
    mz_plus1: float  # (mass + proton) / 1
    mz_plus2: float  # (mass + 2 protons) / 2


def molecular_properties(sequence: str) -> MolecularProperties:
    """Raises ValueError on an empty sequence or a residue other than the 20 standard amino acids."""
    if not sequence:
        raise ValueError("empty sequence")
    try:
        mass = mono_mass(sequence)
    except KeyError as exc:
        raise ValueError(f"non-standard residue {exc} in {sequence}") from exc
    analysis = ProteinAnalysis(sequence)
    aromatic = sum(sequence.count(a) for a in "FWY")
    return MolecularProperties(
        sequence=sequence, length=len(sequence), monoisotopic_mass=mass, net_charge_ph7_4=analysis.charge_at_pH(PH), gravy=analysis.gravy(),
        aromatic_count=aromatic, aromatic_fraction=aromatic / len(sequence), cysteine_count=sequence.count("C"),
        mz_plus1=mass + PROTON_MASS, mz_plus2=(mass + 2 * PROTON_MASS) / 2)
