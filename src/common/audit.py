# Provenance log writer.
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditWriter:
    """Appends one JSONL record per setup stage, candidate stage, and
    failure to `path`, so a run's full provenance trail lives in one file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def record_setup(self, result: Any, *, run_id: str) -> None:
        self._append(
            {
                "run_id": run_id,
                "timestamp": _now(),
                "kind": "setup_stage",
                "stage": result.stage,
                "duration_s": round(result.duration_s, 3),
                "warnings": result.warnings,
            }
        )

    def record_stage(self, result: Any, *, run_id: str) -> None:
        self._append(
            {
                "run_id": run_id,
                "timestamp": _now(),
                "kind": "candidate_stage",
                "stage": result.stage,
                "n_in": result.n_in,
                "n_out": result.n_out,
                "n_removed": result.n_removed,
                "duration_s": round(result.duration_s, 3),
                "models_used": [
                    dataclasses.asdict(ref) for ref in result.models_used
                ],
                "warnings": result.warnings,
            }
        )

    def record_failure(self, stage: str, exc: Exception, *, run_id: str) -> None:
        self._append(
            {
                "run_id": run_id,
                "timestamp": _now(),
                "kind": "failure",
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
