# Stage 1 product brief / Target Product Profile.
from __future__ import annotations
import json
from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class Level(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"


class SafetyConstraints(BaseModel):
    hemolysis: Level
    human_cell_cytotoxicity: Level


class Brief(BaseModel):
    indication: str
    wound_context: List[str] = Field(default_factory=list)
    desired_functions: List[str] = Field(default_factory=list)
    pathogens: List[str] = Field(default_factory=list)
    delivery_system: str
    max_length: int = Field(gt=0, le=100)
    release_target_hours: float = Field(gt=0)
    safety_constraints: SafetyConstraints
    target_population: List[str] = Field(default_factory=list)
    production_constraints: Optional[List[str]] = Field(default=None, description="Constraints related to manufacturing, scalability, and cost.")

    @classmethod
    def load(cls, path: str | Path) -> "Brief":
        """Load a product brief from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Brief file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)