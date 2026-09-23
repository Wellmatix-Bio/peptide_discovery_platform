# Serialise / deserialise candidates.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import storage
from schemas.candidate import Candidate


class BoundaryWriter:
    """Writes each stage's surviving candidates to
    `run_dir/candidates/<stage_name>.jsonl`, one JSON object per line —
    a full per-stage snapshot for debugging and provenance."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        self.candidates_dir = storage.join(run_dir, "candidates")

    def write(
        self, stage_name: str, candidates: list[Candidate], *, run_id: str
    ) -> None:
        storage.ensure_dir(self.candidates_dir)
        path = storage.join(self.candidates_dir, f"{stage_name}.jsonl")
        body = "".join(c.model_dump_json() + "\n" for c in candidates)
        storage.write_text(path, body)


def load_candidates_fasta(path: str | Path, *, schema_version: int) -> list[Candidate]:
    """
    Load candidates from a FASTA file (local path or gs://... URI). The
    fasta format is:
    >candidate_id_1
    SEQUENCE_1
    >candidate_id_2
    SEQUENCE_2
    """
    if not storage.exists(path):
        raise FileNotFoundError(f"FASTA file not found: {path}")
    candidates: list[Candidate] = []
    for line in storage.read_text(path).splitlines():
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


def load_candidates_jsonl(path: str, *, schema_version: int) -> list[Candidate]:
    """Read-side counterpart to BoundaryWriter.write."""
    if not storage.exists(path):
        raise FileNotFoundError(f"Candidates JSONL not found: {path}")
    body = storage.read_text(path)
    return [
        Candidate.model_validate_json(line)
        for line in body.splitlines()
        if line.strip()
    ]


def _rank_sort_key(candidate: Candidate) -> tuple[int, int]:
    """Ranked candidates first (by rank ascending), then everything else
    (rejected / insufficient_evidence / never reached Stage 11) in their
    existing order."""
    ranking = candidate.predictions.get("ranking")
    rank = ranking.get("rank") if ranking else None
    if rank is None:
        return (1, 0)
    return (0, rank)


def write_final_candidates(run_dir: str | Path, candidates: list[Candidate]) -> str | Path:
    """Write the run's final candidates to `run_dir/candidates_final.json`,
    sorted by Stage 11 rank when present (unranked candidates keep their
    incoming order, appended after all ranked ones)."""
    path = storage.join(run_dir, "candidates_final.json")

    ordered = sorted(enumerate(candidates), key=lambda pair: (*_rank_sort_key(pair[1]), pair[0]))
    payload = [candidate.model_dump(mode="json") for _, candidate in ordered]

    storage.write_text(path, json.dumps(payload, indent=2))
    return path if storage.is_gcs_path(path) else Path(path)
