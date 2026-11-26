"""Constraint filters for generated peptides. Each filter returns (passed, reason); nothing is dropped silently."""
from __future__ import annotations

from collections import Counter

from ..digestion.engine import digest_substrate
from ..digestion.settings import DEFAULT_SETTINGS
from ..protein_library.fasta import Substrate
from . import settings as G


def length_ok(seq: str) -> tuple[bool, str]:
    if not (G.MIN_LENGTH <= len(seq) <= G.MAX_LENGTH):
        return False, "length"
    return True, ""


def composition_ok(seq: str, phys: dict) -> tuple[bool, str]:
    if abs(phys["net_charge_pH7_4"]) > G.CHARGE_LIMIT:
        return False, "composition_charge"
    if max(Counter(seq).values()) / len(seq) > G.DOMINANT_RESIDUE_LIMIT:
        return False, "composition_dominant_residue"
    return True, ""


def synthesis_ok(phys: dict) -> tuple[bool, str]:
    liab = [x for x in phys["liabilities"].split(";") if x]
    blocking = [x for x in liab if x in G.SYNTHESIS_BLOCKING]
    if blocking:
        return False, "synthesis_" + blocking[0]
    if len(liab) > G.MAX_LIABILITIES:
        return False, "synthesis_too_many_liabilities"
    return True, ""


def novelty_ok(seq: str, digestion_seqs: set[str], evidence_seqs: set[str]) -> tuple[bool, str]:
    if seq in digestion_seqs:
        return False, "novelty_digestion_fragment"
    if seq in evidence_seqs:
        return False, "novelty_known_peptide"
    return True, ""


def guidance_ok(look: dict, template_pos_sim: float, template_neg_sim: float) -> tuple[bool, str]:
    """The edit must move toward the known positive it is steered by (strictly, by default) and not toward a known negative."""
    pos, neg = look["nearest_positive_sim"], look["nearest_negative_sim"]
    if pos < template_pos_sim - 1e-9:
        return False, "guidance_away_from_positive"
    if G.REQUIRE_STRICT_IMPROVEMENT and pos <= template_pos_sim + 1e-9:
        return False, "guidance_no_improvement"
    if neg > pos and neg > template_neg_sim + 1e-9:
        return False, "guidance_toward_negative"
    return True, ""


class Plausibility:
    """Is the edited peptide still a possible digestion product of its (edited) parent protein?

    The edit is applied to the parent sequence and the parent window around the fragment is digested with the same
    engine and pathways as WP B. The peptide has to appear at the same place. This is a rule-based check of cleavage
    boundaries and missed cleavages, not a prediction of abundance."""

    def __init__(self):
        self.settings = DEFAULT_SETTINGS
        self.reg = self.settings.registry()
        self.cache: dict[tuple, dict] = {}

    def check(self, parent: str, start0: int, template_len: int, edited: str) -> dict:
        """start0: 0-based start of the template in the parent; the template occupies parent[start0:start0+template_len]."""
        key = (parent[max(0, start0 - G.PLAUSIBILITY_PAD):start0], edited,
               parent[start0 + template_len:start0 + template_len + G.PLAUSIBILITY_PAD], start0 < G.PLAUSIBILITY_PAD,
               start0 + template_len + G.PLAUSIBILITY_PAD >= len(parent))
        if key in self.cache:
            return self.cache[key]
        ws = max(0, start0 - G.PLAUSIBILITY_PAD)
        window = parent[ws:start0] + edited + parent[start0 + template_len:start0 + template_len + G.PLAUSIBILITY_PAD]
        local_start = start0 - ws
        sub = Substrate(substrate_id="window", accession="window", precursor_start=1,
                                 precursor_end=len(window), sequence=window)
        pathways, limit, missed = [], False, None
        for pw in self.settings.pathways:
            origins, _ = digest_substrate(sub, pw, self.settings, self.reg)
            for o in origins:
                if o["sequence"] == edited and o["start_in_product"] - 1 == local_start:
                    pathways.append(pw.id)
                    limit = limit or o["is_limit_product"]
                    m = o["missed_cleavages_final"]
                    missed = m if missed is None else min(missed, m)
                    break
        res = {"plausible": bool(pathways), "pathways": ";".join(pathways), "is_limit_product": limit,
               "missed_cleavages_final": "" if missed is None else missed}
        self.cache[key] = res
        return res
