# Validated run configuration model.
from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel
import yaml
from common import storage


class StageConfig(BaseModel):
    enabled: bool = True
    params: dict[str, Any] = {}


class RunConfig(BaseModel):

    run_id: str
    schema_version: int
    seed: int = 42
    seed_candidates_path: str

    artifacts_dir: str
    model_store: str
    input: str | None = None

    use_feature_cache: bool = False

    entry_stage: str
    stages: dict[str, StageConfig]

    def for_stage(self, name: str) -> StageConfig:
        """Config for a single stage, defaulting to an enabled stage with no overrides."""
        return self.stages.get(name, StageConfig())

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Load a run config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        stages_path = Path("./configs/runs", data.get("run_id") + ".yaml")
        with open(stages_path, "r") as f:
            stages_data = yaml.safe_load(f)

        data["stages"] = stages_data.get("stages", {})
        return cls(**data)

    def snapshot(self, path: str | Path) -> None:
        """Write a snapshot of the run config to a YAML file."""
        storage.write_text(path, yaml.safe_dump(self.model_dump()))
