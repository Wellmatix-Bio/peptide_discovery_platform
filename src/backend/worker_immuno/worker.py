"""The Vertex AI Custom Job entry point for the immunomodulation studio.

Unlike the three staged pipelines this takes a list of peptides and scores them directly, so it does
not go through PipelineRunner or backend.worker.worker. It does keep their run-directory contract:
`results.json` carries the worker's own status, and a finished run is never silently overwritten.

ImmunoWorker.run() scores each peptide with the six cytokine-induction predictors in model_store/.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from google.api_core.exceptions import NotFound

from backend.worker.worker import CompletedRunExists, completed_run_reason
from common import model_sync, storage
from schemas.immuno_config import CYTOKINE_MODELS, ImmunoRunConfig

log = logging.getLogger("worker_immuno")


def weights_source() -> str:
    """Where this process gets model weights, decided by DEV_MODE; raises if production has no real bucket to read from.

    DEV_MODE true: the local model_store/model_weights/ folders, nothing is downloaded.
    Otherwise: each model's folder is downloaded from VERTEX_MODEL_STORE (gs://...) the first time it is used
    (common/model_sync.py does the download; this only names it and refuses the unset placeholder early).
    """
    if model_sync.DEV_MODE:
        return "local weights in model_store/model_weights (DEV_MODE)"
    store = model_sync.VERTEX_MODEL_STORE
    if not storage.is_gcs_path(store):
        return f"local weights in model_store/model_weights (VERTEX_MODEL_STORE={store} is not a gs:// path)"
    if "TODO-bucket" in store:
        raise RuntimeError("VERTEX_MODEL_STORE is not set, so there is nowhere to load model weights from; set it to the gs:// model store.")
    return f"weights downloaded from {store}"


class ImmunoWorker:
    def __init__(self, config: ImmunoRunConfig, on_progress=None) -> None:
        self.config = config
        self.on_progress = on_progress  # called as on_progress(batches_done, batches_total, peptides_done) after each batch

    def _predictors(self) -> dict:
        import importlib

        return {
            cytokine: getattr(importlib.import_module(f"model_store.{package}"), cls)()
            for cytokine, (package, cls) in CYTOKINE_MODELS.items()
        }

    def _feature_extractor(self):
        from pipeline.feature_extractor import FeatureExtractor

        return FeatureExtractor()

    def run(self) -> list[dict]:
        """Score `config.peptides` on every cytokine; one {sequence, predictions} dict per peptide, in input order.

        predictions[cytokine] is the model's predict_batch row (probability, label, members, in_length_range)
        plus the `model` package that produced it.
        """
        log.info("model weights: %s", weights_source())
        peptides = self.config.peptides
        size = self.config.batch_size
        total = -(-len(peptides) // size)
        extractor = self._feature_extractor()  # shared, so ESM2 runs once per sequence for all six models
        predictors = self._predictors()  # loaded once, reused by every batch
        results = []
        for number in range(total):
            batch = peptides[number * size : (number + 1) * size]
            scored = [{"sequence": s, "predictions": {}} for s in batch]
            for cytokine, predictor in predictors.items():
                for result, row in zip(scored, predictor.predict_batch(batch, extractor)):
                    result["predictions"][cytokine] = {**row, "model": CYTOKINE_MODELS[cytokine][0]}
            results.extend(scored)
            log.info("batch %d of %d done (%d/%d peptides)", number + 1, total, len(results), len(peptides))
            if self.on_progress:
                self.on_progress(number + 1, total, len(results))
        return results


def load_job_config(path: str, wait_seconds: float = 0) -> ImmunoRunConfig:
    """Read a self-contained JSON run config, waiting up to `wait_seconds` for the API to publish it."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return ImmunoRunConfig.model_validate_json(storage.read_text(path))
        except (FileNotFoundError, NotFound):
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(1.0, max(0, deadline - time.monotonic())))


def run_job(config_path: str, wait_seconds: float = 0, overwrite: bool = False) -> list[dict]:
    config = load_job_config(config_path, wait_seconds)
    run_dir = storage.join(config.artifacts_dir, "runs", config.run_id)

    # Checked before the first write: the failure handler below writes to the same results.json.
    existing = completed_run_reason(run_dir)
    if existing and not overwrite:
        raise CompletedRunExists(
            f"Refusing to run: {existing}. Use a different run_id, or pass --overwrite to replace it."
        )

    storage.ensure_dir(run_dir)
    status_path = storage.join(run_dir, "results.json")
    storage.write_text(
        status_path,
        json.dumps({"run_id": config.run_id, "status": "running", "progress": 0.0, "stage": "initializing"}),
    )
    def report(done: int, total: int, peptides_done: int) -> None:
        # The status route reads `stage`, so the batch count reaches clients without a new API field.
        storage.write_text(
            status_path,
            json.dumps(
                {
                    "run_id": config.run_id,
                    "status": "running",
                    "progress": done / total,
                    "stage": f"batch {done} of {total} ({peptides_done}/{len(config.peptides)} peptides)",
                }
            ),
        )

    try:
        results = ImmunoWorker(config, on_progress=report).run()
        storage.write_text(storage.join(run_dir, "immuno_results.json"), json.dumps(results))
        storage.write_text(
            status_path,
            json.dumps({"run_id": config.run_id, "status": "success", "n_final": len(results), "progress": 1.0}),
        )
        return results
    except Exception as exc:
        # Keep the original error if reporting the failure also fails.
        try:
            storage.write_text(status_path, json.dumps({"run_id": config.run_id, "status": "failed", "error": str(exc)}))
        except Exception:
            import logging

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
