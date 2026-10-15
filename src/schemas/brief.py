# Stage 1 product brief / Target Product Profile.
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


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


MetabolicReceptor = Literal["GLP1R", "GIPR", "GCGR"]
_RECEPTOR_ALIASES = {"GLP1R": "GLP1R", "GLP-1R": "GLP1R", "GIPR": "GIPR", "GCGR": "GCGR"}


class MetabolicBrief(Brief):
    """Stage 1 brief for metabolism / weight control; the single configuration Stages 4, 6, 8 and 11 read.

    The wound fields stay on the base class, empty. Lengths are held to the range the metabolic models cover.
    """

    min_length: int = Field(ge=5, le=50)
    max_length: int = Field(ge=5, le=50)
    #: Receptors the peptide should activate: Stage 4 GA fitness, Stage 6 screening and Stage 11 ranking.
    target_receptors: List[MetabolicReceptor] = Field(min_length=1)
    #: Stage 8 rejects below this predicted human-plasma half-life; Stage 11 scores relative to it.
    min_plasma_half_life_minutes: float = Field(default=60.0, gt=0)
    #: How many peptides Route B (untagged ProtGPT2) samples in Stage 4.
    route_b_n_peptides: int = Field(default=100, ge=1, le=10000)
    #: Turns hemolysis filtering (Stage 8) and its safety score (Stage 11) on or off.
    hemolysis_screening: bool = True

    @field_validator("target_receptors", mode="before")
    @classmethod
    def _normalize_receptors(cls, value):
        if not isinstance(value, (list, tuple)):
            return value
        names = []
        for item in value:
            key = str(item).strip().upper()
            if key not in _RECEPTOR_ALIASES:
                raise ValueError(f"unknown receptor {item!r}; expected GLP-1R, GIPR or GCGR")
            if _RECEPTOR_ALIASES[key] not in names:
                names.append(_RECEPTOR_ALIASES[key])
        return names
