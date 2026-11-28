"""The run context of a food run: the platform's `RunContext` plus the protein library."""
from __future__ import annotations

from dataclasses import dataclass

from pipeline.base import RunContext
from food.a_proteins.protein import Protein


@dataclass(frozen=True)
class FoodRunContext(RunContext):
    proteins: list[Protein] | None = None
