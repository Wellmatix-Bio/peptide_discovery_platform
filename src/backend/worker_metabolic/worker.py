"""Metabolic end-to-end worker: the e2e worker's run, guard and status handling with the metabolic job config."""

from __future__ import annotations

import argparse

from backend.worker_e2e.worker import _parse_args as _e2e_parse_args
from backend.worker_e2e.worker import load_job_config as _load_job_config
from backend.worker_e2e.worker import run_job as _run_job
from schemas.metabolic_config import validate_metabolic_job_config


def load_job_config(path: str, wait_seconds: float = 0):
    """Read a self-contained JSON run config and validate it as a metabolic job."""
    return _load_job_config(path, wait_seconds, validate_metabolic_job_config)


def run_job(config_path: str, wait_seconds: float = 0, overwrite: bool = False) -> list:
    return _run_job(
        config_path, wait_seconds, overwrite, validate=validate_metabolic_job_config
    )


def main(argv: list[str] | None = None) -> None:
    args: argparse.Namespace = _e2e_parse_args(argv)
    run_job(args.config, args.wait_for_config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
