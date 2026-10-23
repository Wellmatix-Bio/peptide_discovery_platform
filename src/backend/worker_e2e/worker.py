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


class CompletedRunExists(RuntimeError):
    """The run directory already holds a finished run, and this worker was about to overwrite it."""


def completed_run_reason(run_dir: str) -> str | None:
    """Why `run_dir` looks like a run that already finished, or None if it does not.

    Two independent signals, because either alone can be absent:
      - `candidates_final.json`, which only the end of a successful run writes.
      - `results.json` reading `status: success`, which is the same run's own verdict.

    Read-only. It must not create or modify anything, since the point is to leave an existing run
    exactly as it was found.
    """
    final_path = storage.join(run_dir, "candidates_final.json")
    if storage.exists(final_path):
        return f"{final_path} exists, so a run already finished here"

    status_path = storage.join(run_dir, "results.json")
    if storage.exists(status_path):
        try:
            status = json.loads(storage.read_text(status_path)).get("status")
        except Exception:  # noqa: BLE001 - unreadable status is not proof of a finished run
            return None
        if status == "success":
            return f"{status_path} reports status 'success', so a run already finished here"
    return None


def run_job(config_path: str, wait_seconds: float = 0, overwrite: bool = False) -> list:
    # Set environment before importing pipeline/model modules, which capture it.
    config = load_job_config(config_path, wait_seconds)
    # os.environ["DEV_MODE"] = "false"
    from common import env, model_sync

    model_sync._synced.clear()

    # API-submitted runs use the numeric Vertex job ID as the pipeline run_id.
    run_dir = storage.join(config.artifacts_dir, "runs", config.run_id)

    # REFUSE TO WRITE OVER A RUN THAT ALREADY FINISHED.
    #
    # A config carries both run_id and artifacts_dir, so anything that replays one writes into
    # that run's directory. On 2026-10-02 a worker started outside Vertex from an existing config
    # did exactly that: the "running" status written a few lines below replaced a completed run's
    # results.json, and its 11 final candidates became unreadable through the API even though
    # candidates_final.json was still sitting beside it. See docs/BASELINE.md.
    #
    # The check is BEFORE the first write and before the try block, deliberately. That block's
    # handler writes a "failed" status to this same path, so raising inside it would destroy the
    # very results this is protecting.
    #
    # A genuine Vertex retry is unaffected: it restarts a run that did not finish, so neither
    # signal is present. Overwriting on purpose stays possible, but has to be asked for.
    existing = completed_run_reason(run_dir)
    if existing and not overwrite:
        raise CompletedRunExists(
            f"Refusing to run: {existing}.\n"
            f"  run_id      {config.run_id}\n"
            f"  run_dir     {run_dir}\n"
            "Writing here would replace that run's results with this one's progress, and the\n"
            "finished candidates would no longer be readable through the API.\n"
            "Use a different run_id, or pass --overwrite if replacing it is what you want."
        )

    storage.ensure_dir(run_dir)
    status_path = storage.join(run_dir, "results.json")
    storage.write_text(
        status_path,
        json.dumps(
            {
                "run_id": config.run_id,
                "status": "running",
                "progress": 0.0,
                "stage": "initializing",
            }
        ),
    )
    try:
        import numpy as np
        import torch
        from runner import PipelineRunner

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        runner = PipelineRunner(config, status_path=status_path)
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
        ctx.feature_extractor.save(storage.join(run_dir, "feature_cache"))
        storage.write_text(
            status_path,
            json.dumps(
                {
                    "run_id": config.run_id,
                    "status": "success",
                    "n_final": len(candidates),
                    "progress": 1.0,
                    "stage": None,
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
    parser.add_argument(
        "--wait-for-config",
        type=int,
        default=0,
        help="Seconds to wait for the API to publish the job-ID config",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Self-contained local or gs:// JSON run config",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Run even though the run directory already holds a finished run, replacing its"
            " results. Off by default: a config replayed by accident would otherwise destroy them."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_job(args.config, args.wait_for_config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
