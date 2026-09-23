from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from google.api_core.exceptions import NotFound


from common import storage
from schemas.run_config import RunConfig
from schemas.e2e_config import validate_job_config


def load_job_config(path: str, wait_seconds: float = 0) -> RunConfig:
    """Read a self-contained JSON run config."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return validate_job_config(json.loads(storage.read_text(path)))
        except (FileNotFoundError, NotFound):
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(1.0, max(0, deadline - time.monotonic())))


def run_job(config_path: str, wait_seconds: float = 0) -> list:
    # Set environment before importing pipeline/model modules, which capture it.
    config = load_job_config(config_path, wait_seconds)
    # os.environ["DEV_MODE"] = "false"
    from common import env, model_sync

    model_sync._synced.clear()

    # API-submitted runs use the numeric Vertex job ID as the pipeline run_id.
    run_dir = storage.join(config.artifacts_dir, "runs", config.run_id)
    storage.ensure_dir(run_dir)
    status_path = storage.join(run_dir, "results.json")
    storage.write_text(
        status_path, json.dumps({"run_id": config.run_id, "status": "running"})
    )
    try:
        import numpy as np
        import torch
        from runner import PipelineRunner

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        runner = PipelineRunner(config)
        candidates = runner.run()
        ctx = runner.ctx
        storage.write_text(
            storage.join(run_dir, "run_context.json"),
            json.dumps(
                {
                    "brief": ctx.brief.model_dump(mode="json") if ctx.brief else None,
                    "objectives": (
                        ctx.objectives.model_dump(mode="json")
                        if ctx.objectives
                        else None
                    ),
                }
            ),
        )
        if config.use_feature_cache:
            ctx.feature_extractor.save(storage.join(run_dir, "feature_cache"))
        storage.write_text(
            status_path,
            json.dumps(
                {
                    "run_id": config.run_id,
                    "status": "success",
                    "n_final": len(candidates),
                }
            ),
        )
        return candidates
    except Exception as exc:
        # Preserve the original error if reporting the failure also fails.
        try:
            storage.write_text(
                status_path,
                json.dumps(
                    {
                        "run_id": config.run_id,
                        "status": "failed",
                        "error": str(exc),
                    }
                ),
            )
        except Exception:
            import logging

            logging.exception("Could not persist job failure status")
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-config", type=int, default=0,
                        help="Seconds to wait for the API to publish the job-ID config")
    parser.add_argument(
        "--config",
        required=True,
        help="Self-contained local or gs:// JSON run config",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_job(args.config, args.wait_for_config)


if __name__ == "__main__":
    main()
