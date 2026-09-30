# Human-readable end-of-run funnel summary.
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from common import storage

if TYPE_CHECKING:
    from pipeline.base import CandidateStageResult

BAR_WIDTH = 30


def _bar(fraction: float) -> str:
    filled = round(fraction * BAR_WIDTH)
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def _format_stats(
    *,
    run_id: str,
    n_start: int,
    n_final: int,
    duration_s: float,
    stage_results: list["CandidateStageResult"],
) -> str:
    lines: list[str] = []
    lines.append(f"Run: {run_id}")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Duration: {duration_s:.1f}s")
    lines.append("")

    survival_pct = (n_final / n_start * 100) if n_start else 0.0
    lines.append(
        f"Candidates: {n_start} in -> {n_final} out ({survival_pct:.1f}% survived)"
    )
    lines.append("")

    header = f"{'Stage':<32} {'In':>6} {'Out':>6} {'Removed':>8} {'Time':>8}  Funnel"
    lines.append(header)
    lines.append("-" * len(header))

    for result in stage_results:
        fraction = (result.n_out / n_start) if n_start else 0.0
        lines.append(
            f"{result.stage:<32} {result.n_in:>6} {result.n_out:>6} "
            f"{result.n_removed:>8} {result.duration_s:>7.1f}s  {_bar(fraction)}"
        )
        for warning in result.warnings:
            lines.append(f"    ! {warning}")

    lines.append("")
    return "\n".join(lines) + "\n"


def write_run_stats(
    run_dir: str | Path,
    *,
    run_id: str,
    n_start: int,
    n_final: int,
    duration_s: float,
    stage_results: list["CandidateStageResult"],
) -> str | Path:
    """Write a plain-text funnel summary to `run_dir/stats_{run_id}.txt`."""
    path = storage.join(run_dir, f"stats_{run_id}.txt")

    content = _format_stats(
        run_id=run_id,
        n_start=n_start,
        n_final=n_final,
        duration_s=duration_s,
        stage_results=stage_results,
    )
    storage.write_text(path, content)
    return path if storage.is_gcs_path(path) else Path(path)
