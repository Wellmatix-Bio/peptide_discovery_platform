# Stage 1 product brief / Target Product Profile.
from __future__ import annotations
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, model_validator


class Brief(BaseModel):
    wound_context: List[str] = Field(default_factory=list)
    desired_functions: List[str] = Field(default_factory=list)
    pathogens: List[str] = Field(default_factory=list)
    min_length: int = Field(gt=0, le=100)
    max_length: int = Field(gt=0, le=100)

    @model_validator(mode="after")
    def _check_length_bounds(self) -> "Brief":
        if self.min_length > self.max_length:
            raise ValueError("min_length must be <= max_length")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Brief":
        """Load a product brief from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Brief file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)