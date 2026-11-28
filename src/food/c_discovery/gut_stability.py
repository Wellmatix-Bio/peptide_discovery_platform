"""Gut stability (s06): where pepsin, trypsin and chymotrypsin cut a peptide. The Expasy PeptideCutter rules, no model."""
from __future__ import annotations

from food.b_digestion.rules import BUILTIN_ENZYMES

from . import settings


def cleavage_sites(sequence: str) -> dict[str, list[int]]:
    """{protease: 1-based P1 positions of its internal cuts}."""
    return {name: [i + 1 for i in BUILTIN_ENZYMES[rule].sites(sequence)] for name, rule in settings.GUT_ENZYMES.items()}
