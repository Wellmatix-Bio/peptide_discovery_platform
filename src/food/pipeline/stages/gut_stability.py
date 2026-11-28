"""s06_food_gut_stability: where do pepsin, trypsin and chymotrypsin cut the peptide? The Expasy PeptideCutter rules, no model.

One rule per protease (`settings.GUT_ENZYMES`, rules in `digestion/rules.py`). Pepsin and trypsin are the rules the digestion stage uses; chymotrypsin is the low-specificity rule, which
finds 97% of observed cuts where the high-specificity rule finds 62% (docs/validation/food/protease_validation.md). A site is a hypothesis: it says nothing about
abundance, kinetics or intestinal exposure. Only internal bonds count; a rule never cuts after the last residue.
gut_survival = share of the peptide's internal bonds that none of the three proteases cuts.
"""
from __future__ import annotations

from pipeline.base import CandidateStage, RunContext
from food.c_discovery import settings
from food.c_discovery.gut_stability import cleavage_sites
from food.c_discovery.screening import screening
from schemas.candidate import Candidate
from schemas.run_config import StageConfig


class FoodGutStability(CandidateStage):
    name = "s06_food_gut_stability"
    requires = {"screening"}
    produces = {"cleavage"}

    def run(self, candidates: list[Candidate], config: StageConfig, ctx: RunContext) -> list[Candidate]:
        """Adds predictions["cleavage"] to every candidate."""
        for c in candidates:
            c.predictions["cleavage"] = {**screening.stability_scores(cleavage_sites(c.sequence), len(c.sequence)),
                                         "rules": dict(settings.GUT_ENZYMES), "method": "protease cleavage rules (Expasy PeptideCutter)",
                                         "statement": settings.UNCUT_BOND_FRACTION_STATEMENT}
        return candidates
