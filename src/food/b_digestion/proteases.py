"""The protease registry and the side-by-side comparison of the cleavage rules, for the digestion page.

Every rule is a deterministic Expasy PeptideCutter baseline. A cleavage site is a hypothesis, not a measured cut, and the rules give no probability.
"""
from __future__ import annotations

from food.a_proteins.fasta import STANDARD

from .rules import BUILTIN_ENZYMES
from .settings import DEFAULT_SETTINGS, PROTEASE_PATHWAYS
from .validation import LIMITATIONS, RULE_VALIDATION

PEPTIDE_SIZE_WARNING = "the benchmark had no substrate of 40 residues or fewer, so the rule is untested on peptides this small"


def protease_of(rule: str) -> str:
    """'pepsin' for 'pepsin_ph1.3', 'chymotrypsin' for 'chymotrypsin_high'."""
    return rule.split("_")[0]


def is_validated(rule: str) -> bool:
    """True only when the benchmark passed the rule; a rule that was not benchmarked is not validated."""
    return RULE_VALIDATION.get(rule, "").startswith("validated")


def default_rule(protease: str) -> str:
    """The rule the default digestion uses for this protease: the enzyme of its single-enzyme pathway."""
    pathway = next(p for p in DEFAULT_SETTINGS.pathways if p.id == PROTEASE_PATHWAYS[protease])
    return pathway.stages[0].enzymes[0]


def list_proteases() -> list[dict]:
    """One row per protease: its rules (pH range, description, benchmark verdict), the rule the digestion uses and a warning when that rule is not validated."""
    rows = []
    for protease in PROTEASE_PATHWAYS:
        used = default_rule(protease)
        rows.append({
            "protease": protease, "default_pathway": PROTEASE_PATHWAYS[protease], "default_rule": used, "settings_version": DEFAULT_SETTINGS.name,
            "rules": [{"name": e.name, "ph_range": e.ph_range, "description": e.description, "source": e.source, "validation": RULE_VALIDATION.get(e.name, "not benchmarked"),
                       "validated": is_validated(e.name)} for e in BUILTIN_ENZYMES.values() if protease_of(e.name) == protease],
            "warning": "" if is_validated(used) else f"the rule the digestion uses ({used}) is not validated: {RULE_VALIDATION.get(used, 'not benchmarked')}",
        })
    return rows


def compare_proteases(sequence: str, proteases: list[str] | None = None) -> list[dict]:
    """Where each rule of each protease cuts one sequence, side by side: method, version, 1-based P1 cut positions, and warnings.

    Raises ValueError for an empty sequence, a non-standard residue or an unknown protease."""
    sequence = sequence.strip().upper()
    if not sequence or set(sequence) - STANDARD:
        raise ValueError("the sequence must be non-empty and use only the 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY)")
    unknown = [p for p in proteases or [] if p not in PROTEASE_PATHWAYS]
    if unknown:
        raise ValueError(f"Unknown protease {', '.join(unknown)}; available: {', '.join(PROTEASE_PATHWAYS)}")
    rows = []
    for e in BUILTIN_ENZYMES.values():
        protease = protease_of(e.name)
        if proteases and protease not in proteases:
            continue
        cuts = [i + 1 for i in e.sites(sequence)]
        warnings = [] if is_validated(e.name) else [RULE_VALIDATION.get(e.name, "not benchmarked")]
        if len(sequence) <= 40:
            warnings.append(PEPTIDE_SIZE_WARNING)
        rows.append({"protease": protease, "rule": e.name, "method": e.source, "settings_version": DEFAULT_SETTINGS.name, "ph_range": e.ph_range,
                     "used_by_digestion": e.name == default_rule(protease), "cut_positions": cuts, "n_cuts": len(cuts), "validated": is_validated(e.name),
                     "warnings": warnings, "note": LIMITATIONS[0]})
    return rows
