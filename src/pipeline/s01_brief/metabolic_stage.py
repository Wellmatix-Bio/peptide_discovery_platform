# Stage 1 (metabolism / weight control): product brief with receptor targets, half-life and hemolysis controls.
from __future__ import annotations

from pipeline.base import RunContext, SetupStage, StageError
from schemas.brief import MetabolicBrief
from schemas.run_config import StageConfig


class Stage1Metabolic(SetupStage):
    """Builds a MetabolicBrief, the one configuration Stages 4, 6, 8 and 11 read for the metabolic pipeline.

    Opt-in: it runs only when named in the run config, normally in place of `s01_therapeutic_product_brief`.
    """

    name = "s01_metabolic_therapeutic_product_brief"

    def run(self, config: StageConfig, ctx: RunContext) -> MetabolicBrief:
        brief = config.params.get("brief")
        if not brief:
            raise StageError(self.name, "brief is not specified in the config.")
        return MetabolicBrief(**brief)
