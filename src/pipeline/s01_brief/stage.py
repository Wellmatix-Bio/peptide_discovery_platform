# Stage 1: Product Brief - produces the machine-readable JSON project config / TPP.
from __future__ import annotations

from pipeline.base import RunContext, SetupStage, StageError
from schemas.brief import Brief
from schemas.run_config import StageConfig


class Stage1(SetupStage):
    name = "s01_therapeutic_product_brief"

    def run(self, config: StageConfig, ctx: RunContext) -> Brief:
        brief_path = config.params.get("brief_path")
        if not brief_path:
            raise StageError(self.name, "brief_path is not specified in the config.")
        return Brief.load(brief_path)
