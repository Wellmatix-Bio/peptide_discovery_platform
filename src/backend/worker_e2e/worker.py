"""The Vertex AI Custom Job entry point for the wound-healing pipeline.

This module is the IMAGE'S ENTRY POINT and nothing more: `backend.worker_e2e.worker` is what
worker_e2e/Dockerfile runs, and that path is kept so the image, its cloudbuild file and the tests
did not have to move. The run loop lives in backend.worker.worker, once, for all three
pipelines -- see its module docstring for what three copies of it cost.

Everything below binds PIPELINE and forwards. If you are looking for what a worker DOES, it is
not here.
"""

from __future__ import annotations

from backend.worker import worker as shared

# Re-exported so existing callers and tests keep resolving `worker.<name>` through this module,
# including `worker.storage` and `worker.time`, which tests monkeypatch.
from backend.worker.worker import (  # noqa: F401
    CompletedRunExists,
    completed_run_reason,
    storage,
    time,
    _parse_args,
)
from schemas.run_config import RunConfig

#: Which pipeline's stage names this image's configs use.
PIPELINE = "wound_healing"


def load_job_config(path: str, wait_seconds: float = 0) -> RunConfig:
    return shared.load_job_config(path, wait_seconds, pipeline=PIPELINE)


def run_job(config_path: str, wait_seconds: float = 0, overwrite: bool = False) -> list:
    return shared.run_job(config_path, wait_seconds, overwrite=overwrite, pipeline=PIPELINE)


def main(argv: list[str] | None = None) -> None:
    shared.main(argv, pipeline=PIPELINE)


if __name__ == "__main__":
    main()
