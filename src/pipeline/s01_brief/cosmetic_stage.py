# Stage 1 (cosmetics): product brief with functions, target depth, application area, product pH and peptide length.
from __future__ import annotations

from pipeline.base import RunContext, SetupStage, StageError
from schemas.brief import CosmeticBrief
from schemas.run_config import StageConfig


class CosmeticStage1(SetupStage):
    """Builds a CosmeticBrief, the one configuration the cosmetic stages read.

    Opt-in: it runs only when named in the run config, in place of `s01_therapeutic_product_brief`.
    """

    name = "s01_cosmetic_therapeutic_product_brief"

    def run(self, config: StageConfig, ctx: RunContext) -> CosmeticBrief:
        brief = config.params.get("brief")
        if brief is None:
            raise StageError(self.name, "brief is not specified in the config.")
        return CosmeticBrief(**brief)
