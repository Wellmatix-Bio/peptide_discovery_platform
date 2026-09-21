# Smoke test: run s01 (product brief) and s02 (wound biology / targets)
# through PipelineRunner and print what each stage produced.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from runner import PipelineRunner
from schemas.run_config import RunConfig, StageConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

config = RunConfig(
    run_id="dfu_smoke_test",
    schema_version=1,
    seed=42,
    artifacts_dir=str(REPO_ROOT / "artifacts"),
    model_store=str(REPO_ROOT / "model_store"),
    knowledge_base_path=None,
    entry_stage="s01_therapeutic_product_brief",
    input=None,
    stages={
        "s01_therapeutic_product_brief": StageConfig(enabled=True),
        "s02_wound_biology_and_targets": StageConfig(
            enabled=True,
            params={"deficit_rules_path": str(REPO_ROOT / "configs" / "deficit_rules.config.yaml")},
        ),
        "s03_data_integration": StageConfig(enabled=False),
    },
)

runner = PipelineRunner(config)
runner.run()

print("\n=== s01 brief ===")
print(runner.ctx.brief.model_dump_json(indent=2))

print("\n=== s02 objectives ===")
print(runner.ctx.objectives.model_dump_json(indent=2))
