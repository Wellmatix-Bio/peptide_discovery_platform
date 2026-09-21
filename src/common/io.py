# Serialise / deserialise candidates.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.candidate import Candidate


class BoundaryWriter:
    """Writes each stage's surviving candidates to
    `run_dir/candidates/<stage_name>.jsonl`, one JSON object per line —
    a full per-stage snapshot for debugging and provenance."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.candidates_dir = self.run_dir / "candidates"

    def write(
        self, stage_name: str, candidates: list[Candidate], *, run_id: str
    ) -> None:
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        path = self.candidates_dir / f"{stage_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for candidate in candidates:
                f.write(candidate.model_dump_json() + "\n")


def load_candidates_fasta(path: str | Path, *, schema_version: int) -> list[Candidate]:
    """
    Load candidates from a FASTA file. The fasta format is:
    >candidate_id_1
    SEQUENCE_1
    >candidate_id_2
    SEQUENCE_2
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    candidates: list[Candidate] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Header line, extract candidate ID
                candidate_id = line[1:].strip()
                candidates.append(Candidate(id=candidate_id))
            else:
                # Sequence line, add sequence to the last candidate
                if candidates:
                    candidates[-1].sequence = line
    return candidates


def _rank_sort_key(candidate: Candidate) -> tuple[int, int]:
    """Ranked candidates first (by rank ascending), then everything else
    (rejected / insufficient_evidence / never reached Stage 11) in their
    existing order."""
    ranking = candidate.predictions.get("ranking")
    rank = ranking.get("rank") if ranking else None
    if rank is None:
        return (1, 0)
    return (0, rank)


def write_final_candidates(run_dir: str | Path, candidates: list[Candidate]) -> Path:
    """Write the run's final candidates to `run_dir/candidates_final.json`,
    sorted by Stage 11 rank when present (unranked candidates keep their
    incoming order, appended after all ranked ones)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "candidates_final.json"

    ordered = sorted(enumerate(candidates), key=lambda pair: (*_rank_sort_key(pair[1]), pair[0]))
    payload = [candidate.model_dump(mode="json") for _, candidate in ordered]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path
