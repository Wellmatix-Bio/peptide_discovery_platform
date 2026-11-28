"""Entry point of a food run: reads its config, loads the submitted proteins, runs the food stages with `FoodPipelineRunner`.

Keeps the run-directory contract of the other workers: `results.json` carries the worker's own status, and a finished run is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from google.api_core.exceptions import NotFound

from backend.worker.worker import CompletedRunExists, completed_run_reason
from common import storage
from food.pipeline.run_config import FoodRunConfig

log = logging.getLogger("worker_food")


def load_job_config(path: str, wait_seconds: float = 0) -> FoodRunConfig:
    """Read the JSON run config, waiting up to `wait_seconds` for the API to publish it."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return FoodRunConfig.model_validate_json(storage.read_text(path))
        except (FileNotFoundError, NotFound):
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(1.0, max(0, deadline - time.monotonic())))


def load_run_proteins(config: FoodRunConfig):
    """The submitted proteins with the run's food source, allergen status and protein-family cluster; QC issues and clustering notes come with them."""
    from food.a_proteins.families import assign_clusters
    from food.a_proteins.protein import load_proteins

    with tempfile.TemporaryDirectory() as tmp:  # the FASTA sits in the run folder, local or gs://; the reader wants a file
        fasta = Path(tmp) / "proteins.fasta"
        fasta.write_text(storage.read_text(config.fasta_path), encoding="utf-8")
        records = None
        if config.source_records_path:  # the library's records for the proteins it supplied or matched
            records = Path(tmp) / "source_records.csv"
            records.write_text(storage.read_text(config.source_records_path), encoding="utf-8")
        proteins, issues = load_proteins(fasta, records)
    # a record's own flag stands; the run's allergen status covers the proteins no record covers
    proteins = [replace(p, food_source=p.food_source or config.food_source, allergen_source=p.allergen_source if p.allergen_source is not None else config.allergen_source) for p in proteins]
    proteins, notes = assign_clusters(proteins)
    return proteins, issues, notes


def ranking_rules() -> dict:
    """The settings the ranking ran under, saved with the report so the reports can state them."""
    from food.c_discovery import settings

    return {"min_length": settings.MIN_LENGTH, "max_length": settings.MAX_LENGTH, "model_stage_pool": settings.MODEL_TOP, "final_count": settings.FINAL_COUNT,
            "max_per_cluster": settings.MAX_PER_CLUSTER, "max_per_parent_family": settings.MAX_PER_PARENT_CLUSTER, "min_exact_evidence": settings.MIN_EXACT_EVIDENCE,
            "tie_break": list(settings.TIEBREAK), "shortlist_size": settings.SHORTLIST_SIZE}


def write_ranking_report(runner, run_dir: str) -> None:
    """What the ranking stage knows beyond the ranked candidates (gate, run facts, reference controls), saved for the results route."""
    stage = next((s for s in runner.candidate_stages if s.name == "s07_food_ranking_decision"), None)
    if stage is not None:
        storage.write_text(storage.join(run_dir, "ranking_report.json"), json.dumps(
            {"gate": stage.gate, "facts": stage.facts, "controls": stage.controls, "digestion_warnings": getattr(stage, "digestion_warnings", []), "rules": ranking_rules()}, default=str))


def write_provenance_check(candidates, proteins, run_dir: str) -> list[dict]:
    """Checks every final candidate against the run's proteins and saves the outcome next to the results."""
    from food.shared.provenance import check_provenance

    issues = check_provenance(candidates, proteins)
    storage.write_text(storage.join(run_dir, "provenance_check.json"), json.dumps({"checked": len(candidates), "issues": issues}))
    return issues


def run_job(config_path: str, wait_seconds: float = 0, overwrite: bool = False) -> list:
    config = load_job_config(config_path, wait_seconds)
    run_dir = storage.join(config.artifacts_dir, "runs", config.run_id)
    existing = completed_run_reason(run_dir)  # before the first write: the failure handler below writes to the same results.json
    if existing and not overwrite:
        raise CompletedRunExists(f"Refusing to run: {existing}. Use a different run_id, or pass --overwrite to replace it.")

    storage.ensure_dir(run_dir)
    status_path = storage.join(run_dir, "results.json")
    storage.write_text(status_path, json.dumps({"run_id": config.run_id, "status": "running", "progress": 0.0, "stage": "initializing"}))
    try:
        import numpy as np
        import torch
        from food.pipeline.runner import FoodPipelineRunner

        proteins, issues, notes = load_run_proteins(config)
        storage.write_text(storage.join(run_dir, "protein_qc.json"), json.dumps([asdict(i) for i in issues]))
        storage.write_text(storage.join(run_dir, "protein_clusters.json"), json.dumps({
            "notes": notes, "clusters": {p.substrate_id: {"cluster": p.cluster, "source": p.cluster_source} for p in proteins}}))
        if not proteins:
            raise ValueError("no usable protein records in the submitted FASTA: " + "; ".join(f"{i.substrate_id}: {i.message}" for i in issues))
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        runner = FoodPipelineRunner(config, status_path=status_path)
        runner.ctx = replace(runner.ctx, proteins=proteins)
        candidates = runner.run()
        write_ranking_report(runner, run_dir)
        issues = write_provenance_check(candidates, proteins, run_dir)
        if issues:  # a peptide whose lineage cannot be reproduced must not reach a report
            raise RuntimeError(f"provenance check failed for {len({i['peptide_id'] for i in issues})} peptide(s); first: {issues[0]['check']}: {issues[0]['detail']}")
        storage.write_text(status_path, json.dumps({"run_id": config.run_id, "status": "success", "n_final": len(candidates), "progress": 1.0, "stage": "finished"}))
        return candidates
    except Exception as exc:
        try:
            storage.write_text(status_path, json.dumps({"run_id": config.run_id, "status": "failed", "error": str(exc)}))
        except Exception:  # keep the original error if reporting the failure also fails
            logging.exception("Could not persist job failure status")
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Self-contained local or gs:// JSON run config")
    parser.add_argument("--wait-for-config", type=int, default=0, help="Seconds to wait for the API to publish the config")
    parser.add_argument("--overwrite", action="store_true", help="Run even if the run directory already holds a finished run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_job(args.config, args.wait_for_config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
