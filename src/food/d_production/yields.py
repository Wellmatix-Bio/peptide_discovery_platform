"""Theoretical peptide yield from a parent protein."""
from __future__ import annotations

from food.b_digestion.masses import avg_mass


def theoretical_yield(protein_sequence: str, peptide_sequence: str, available_protein_mass: float) -> float:
    """Y_max = n x MW_peptide / MW_protein x available protein mass, in the unit of the mass given.

    n is the number of non-overlapping copies of the peptide in the protein; the yield is 0 when the protein does not contain it. Molecular weights are average masses of the
    free chain and peptide (`avg_mass`). Release is assumed complete: no losses, no missed cleavages, no further cuts. Pass 1 as the mass to get the fraction of the protein mass.
    """
    if available_protein_mass < 0:
        raise ValueError("available_protein_mass must be >= 0")
    copies = protein_sequence.count(peptide_sequence) if peptide_sequence else 0
    if not copies:
        return 0.0
    try:
        return copies * avg_mass(peptide_sequence) / avg_mass(protein_sequence) * available_protein_mass
    except KeyError as exc:
        raise ValueError(f"non-standard residue {exc} in the sequence") from exc
