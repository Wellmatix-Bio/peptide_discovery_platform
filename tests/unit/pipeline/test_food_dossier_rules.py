"""The dossier's statement of how many bonds the pepsin rule can reach, checked against the rule itself."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from food.b_digestion.rules import BUILTIN_ENZYMES  # noqa: E402
from food.shared.dossier import pepsin_reachable_bonds  # noqa: E402


def test_the_reach_formula_equals_the_most_bonds_the_rule_can_cut_at_every_length():
    for rule in ("pepsin_ph1.3", "pepsin_ph2.0"):
        for n in range(1, 41):
            assert len(BUILTIN_ENZYMES[rule].sites("F" * n)) == pepsin_reachable_bonds(n), (rule, n)
