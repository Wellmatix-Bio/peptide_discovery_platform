# Validated run configuration model.
from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel
import yaml
from common import storage
from common.env import DEV_MODE

#: Internal candidate-file schema version. Not a client-facing knob -- there
#: is only ever one version in play at a time.
SCHEMA_VERSION = 1

# TODO: production run_id should be set by the caller (e.g. the Vertex job
# id) after job creation, not read from client config. Replace this
# placeholder once that wiring is in place.
_PLACEHOLDER_RUN_ID = "<your_run_id>"


class StageConfig(BaseModel):
    enabled: bool = True
    params: dict[str, Any] = {}


class RunConfig(BaseModel):

    #: Only read from client config in DEV_MODE; production runs must set
    #: this themselves after construction (see _PLACEHOLDER_RUN_ID above).
    run_id: str = _PLACEHOLDER_RUN_ID
    seed: int = 42
    seed_candidates_path: str

    artifacts_dir: str
    model_store: str

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

        if not DEV_MODE:
            data.pop("run_id", None)

        run_id = data.get("run_id", _PLACEHOLDER_RUN_ID)
        stages_path = Path("./configs/runs", run_id + ".yaml")
        with open(stages_path, "r") as f:
            stages_data = yaml.safe_load(f)

        data["stages"] = stages_data.get("stages", {})
        return cls(**data)

    def snapshot(self, path: str | Path) -> None:
        """Write a snapshot of the run config to a YAML file."""
        storage.write_text(path, yaml.safe_dump(self.model_dump()))
