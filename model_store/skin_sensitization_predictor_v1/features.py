"""Peptide -> structure -> the 14 scalar features the model was trained on (RDKit descriptors + electrophilic alerts)."""

from __future__ import annotations

from typing import Any

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

# The training notebook's column order; the fitted scalers and models depend on it.
FEATURES = [
    "mol_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "formal_charge", "heavy_atoms",
    "aromatic_rings", "fraction_csp3", "heteroatoms", "sulfur_atoms", "carbonyl_groups", "electrophilic_alerts",
]

# Residue templates, N->C, L-configuration; {n} is a ring-closure slot on the backbone N, {s} on the side chain.
# Copied from the training repo's build_peptide_structures.py so structures match the training data.
RESIDUES = {
    "G": "N{n}CC(=O)",
    "A": "N{n}[C@@H](C)C(=O)",
    "R": "N{n}[C@@H](CCCNC(N)=N)C(=O)",
    "N": "N{n}[C@@H](CC(N)=O)C(=O)",
    "D": "N{n}[C@@H](CC(=O)O)C(=O)",
    "C": "N{n}[C@@H](CS{s})C(=O)",
    "Q": "N{n}[C@@H](CCC(N)=O)C(=O)",
    "E": "N{n}[C@@H](CCC(=O)O)C(=O)",
    "H": "N{n}[C@@H](Cc1c[nH]cn1)C(=O)",
    "I": "N{n}[C@@H]([C@@H](C)CC)C(=O)",
    "L": "N{n}[C@@H](CC(C)C)C(=O)",
    "K": "N{n}[C@@H](CCCCN)C(=O)",
    "M": "N{n}[C@@H](CCSC)C(=O)",
    "F": "N{n}[C@@H](Cc1ccccc1)C(=O)",
    "P": "N{n}3CCC[C@H]3C(=O)",
    "S": "N{n}[C@@H](CO)C(=O)",
    "T": "N{n}[C@@H]([C@@H](C)O)C(=O)",
    "W": "N{n}[C@@H](Cc1c[nH]c2ccccc12)C(=O)",
    "Y": "N{n}[C@@H](Cc1ccc(O)cc1)C(=O)",
    "V": "N{n}[C@@H](C(C)C)C(=O)",
}
LYS_ACYLATED = "N{n}[C@@H](CCCCNC(=O)CCCCCCCCCCCCCCC)C(=O)"  # palmitoyl on the epsilon amine
ACETYL = "CC(=O)"
PALMITOYL = "CCCCCCCCCCCCCCCC(=O)"

ALERT_SMARTS = {
    "aldehyde": "[$([CX3H1](=O)[#6]),$([CX3H2]=O)]",
    "michael_acceptor": "[C;X3]=[C;X3]-[C;X3](=[O;X1])",
    "epoxide": "[O;X2;r3]1[C;r3][C;r3]1",
    "aziridine": "[N;X3;r3]1[C;r3][C;r3]1",
    "isocyanate": "[N;X2]=[C;X2]=[O;X1]",
    "isothiocyanate": "[N;X2]=[C;X2]=[S;X1]",
    "acyl_halide": "[C;X3](=[O;X1])-[Cl,Br,I,F]",
    "sulfonyl_halide": "[S;X4](=[O;X1])(=[O;X1])-[Cl,Br,I,F]",
    "alkyl_halide": "[C;X4]-[Cl,Br,I]",
}


def _flip(fragment: str) -> str:
    """Mirror every stereocentre in a residue fragment (L -> D)."""
    return fragment.replace("@@", "\0").replace("@", "@@").replace("\0", "@")


def peptide_smiles(sequence: str, modifications: dict[str, Any] | None = None) -> str:
    """SMILES of a linear peptide with free termini unless modified.

    Modifications use the cosmetic pipeline's keys: `n_acetyl`, `c_amide`, `d_amino_acid` (0-based indices) and
    `lipid_tail` (palmitoyl, at the N-terminus when its position is "n_terminus" or 0, else on that Lys).
    """
    if not sequence or not set(sequence) <= AMINO_ACID_SET:
        raise ValueError(f"sequence must be non-empty and use the 20 uppercase standard amino acids: {sequence!r}")
    mods = modifications or {}
    lipid = mods.get("lipid_tail")
    lipid_position = (lipid.get("position") if isinstance(lipid, dict) else None) if lipid else None
    n_terminal_lipid = bool(lipid) and lipid_position in ("n_terminus", 0, None)
    d_positions = mods.get("d_amino_acid") or []
    if d_positions is True:
        d_positions = []
    last = len(sequence) - 1

    parts = []
    for i, residue in enumerate(sequence):
        template = LYS_ACYLATED if lipid and not n_terminal_lipid and i == lipid_position else RESIDUES[residue]
        fragment = template.format(n="", s="")
        parts.append(_flip(fragment) if i in d_positions else fragment)
    body = "".join(parts) + ("N" if mods.get("c_amide") else "O")

    if n_terminal_lipid:
        return PALMITOYL + body
    return (ACETYL if mods.get("n_acetyl") else "") + body


def _rdkit():
    # Imported on first use so that importing the model package never fails on a machine without RDKit.
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors
    from rdkit.Chem import rdMolDescriptors as rd

    RDLogger.DisableLog("rdApp.*")
    return Chem, Crippen, Descriptors, rd


_alert_patterns: dict[str, Any] = {}


def feature_vector(smiles: str) -> list[float]:
    """The 14 features, in `FEATURES` order, for the largest fragment of `smiles`."""
    Chem, Crippen, Descriptors, rd = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    parent = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
    if not _alert_patterns:
        _alert_patterns.update({name: Chem.MolFromSmarts(s) for name, s in ALERT_SMARTS.items()})
        _alert_patterns["_carbonyl"] = Chem.MolFromSmarts("[CX3]=[OX1]")
    alerts = sum(int(parent.HasSubstructMatch(p)) for name, p in _alert_patterns.items() if name != "_carbonyl")
    return [
        Descriptors.MolWt(parent),
        Crippen.MolLogP(parent),
        rd.CalcTPSA(parent),
        rd.CalcNumHBD(parent),
        rd.CalcNumHBA(parent),
        rd.CalcNumRotatableBonds(parent),
        sum(a.GetFormalCharge() for a in parent.GetAtoms()),
        parent.GetNumHeavyAtoms(),
        rd.CalcNumAromaticRings(parent),
        rd.CalcFractionCSP3(parent),
        rd.CalcNumHeteroatoms(parent),
        sum(a.GetSymbol() == "S" for a in parent.GetAtoms()),
        len(parent.GetSubstructMatches(_alert_patterns["_carbonyl"])),
        alerts,
    ]
