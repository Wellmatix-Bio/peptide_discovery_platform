"""Read-side counterpart to `BoundaryWriter` for food candidates; the shared loader builds the base `Candidate` and would drop the food fields."""

from __future__ import annotations

from common import storage
from schemas.food import FoodCandidate


def load_food_candidates_jsonl(path: str) -> list[FoodCandidate]:
    if not storage.exists(path):
        raise FileNotFoundError(f"Candidates JSONL not found: {path}")
    return [FoodCandidate.model_validate_json(line) for line in storage.read_text(path).splitlines() if line.strip()]
