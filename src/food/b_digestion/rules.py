"""Protease cleavage rules: the Expasy PeptideCutter tables, each written as one regex on the P1 residue."""
from __future__ import annotations

import re
from dataclasses import dataclass

# ============================== ENZYMES ==============================
# Protease cleavage rules.
# Rules are the Expasy PeptideCutter specificity tables as encoded in pyteomics (expasy_rules),
# written as one regular expression that matches the **P1 residue** (the residue
# before the scissile bond). A match at 0-based index ``i`` means the bond between
# residue ``i`` and ``i + 1`` is cleaved.
# 
# Look-behind / look-ahead groups encode the P2, P1', P2' context, so a site near a
# protein terminus that lacks the required context is not cleaved. This is the
# published behaviour of the rules, not a bug.
# 
# Predicted cleavage is a hypothesis: it says nothing about abundance, kinetics,
# stability or intestinal exposure.

@dataclass(frozen=True)
class Enzyme:
    name: str
    pattern: str                      # regex matching the P1 residue
    ph_range: tuple[float, float]     # pH range the rule was derived for / is used at
    source: str
    description: str

    @property
    def regex(self) -> "re.Pattern[str]":
        return re.compile(self.pattern)

    def sites(self, seq: str) -> list[int]:
        """0-based P1 indices; the bond after ``seq[i]`` is cleaved."""
        n = len(seq)
        return [m.start() for m in self.regex.finditer(seq) if 0 <= m.start() < n - 1]


_EXPASY = "Expasy PeptideCutter (Gasteiger 2005), https://web.expasy.org/peptide_cutter/"

BUILTIN_ENZYMES: dict[str, Enzyme] = {
    e.name: e
    for e in [
        Enzyme(
            "pepsin_ph1.3",
            r"((?<=[^HKR][^P])[^R](?=[FL][^P]))|((?<=[^HKR][^P])[FL](?=\w[^P]))",
            (1.0, 2.0), _EXPASY,
            "Pepsin, pH 1.3 rule: F/L at P1 or P1', not after H/K/R at P2, no P at P1'/P2'.",
        ),
        Enzyme(
            "pepsin_ph2.0",
            r"((?<=[^HKR][^P])[^R](?=[FLWY][^P]))|((?<=[^HKR][^P])[FLWY](?=\w[^P]))",
            (2.0, 4.0), _EXPASY,
            "Pepsin, pH > 2 rule: as pH 1.3 plus W and Y.",
        ),
        Enzyme(
            "trypsin",
            r"([KR](?=[^P]))|((?<=W)K(?=P))|((?<=M)R(?=P))",
            (6.5, 9.0), _EXPASY,
            "Trypsin: after K/R unless followed by P; exceptions WK-P and MR-P are cleaved.",
        ),
        Enzyme(
            "chymotrypsin_high",
            r"([FY](?=[^P]))|(W(?=[^MP]))",
            (6.5, 9.0), _EXPASY,
            "Chymotrypsin, high specificity: after F/Y (not before P), W (not before M/P).",
        ),
        Enzyme(
            "chymotrypsin_low",
            r"([FLY](?=[^P]))|(W(?=[^MP]))|(M(?=[^PY]))|(H(?=[^DMPW]))",
            (6.5, 9.0), _EXPASY,
            "Chymotrypsin, low specificity: adds L, M, H contexts.",
        ),
    ]
}


def build_registry(custom: dict | None = None) -> dict[str, Enzyme]:
    """Built-in enzymes plus user rules.

    ``custom`` maps name -> {"pattern": regex matching P1, "ph_range": [lo, hi],
    "source": str, "description": str}. A custom name may not shadow a built-in.
    """
    reg = dict(BUILTIN_ENZYMES)
    for name, spec in (custom or {}).items():
        if name in reg:
            raise ValueError(f"custom enzyme '{name}' shadows a built-in rule; choose another name")
        try:
            re.compile(spec["pattern"])
        except (re.error, KeyError) as exc:
            raise ValueError(f"custom enzyme '{name}': invalid or missing 'pattern' ({exc})") from exc
        lo, hi = spec.get("ph_range", [0.0, 14.0])
        reg[name] = Enzyme(name, spec["pattern"], (float(lo), float(hi)),
                           spec.get("source", "user-defined"), spec.get("description", ""))
    return reg
# ============================ END ENZYMES ============================
