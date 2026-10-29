# Stage 1 product brief / Target Product Profile.
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Literal
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


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


class MetabolicBrief(Brief):
    """Stage 1 brief for metabolism / weight control; the single configuration Stages 4, 6, 8 and 11 read.

    The wound fields stay on the base class, empty. Lengths are held to the range the metabolic models cover.
    """

    min_length: int = Field(ge=5, le=50)
    max_length: int = Field(ge=5, le=50)
    #: Receptors the peptide should activate: Stage 4 GA fitness, Stage 6 screening and Stage 11 ranking.
    target_receptors: List[MetabolicReceptor] = Field(min_length=1)

    @field_validator("target_receptors")
    @classmethod
    def _drop_duplicate_receptors(cls, value):
        return list(dict.fromkeys(value))


CosmeticFunction = Literal["collagen_ecm", "antioxidant", "pigmentation"]
TargetDepth = Literal["outer_layer", "epidermis", "dermis"]
ApplicationArea = Literal["face", "eye_area", "body"]
#: Skin layer a primary function acts in, used when `target_depth` is not given.
DEFAULT_DEPTH_BY_FUNCTION: dict[str, TargetDepth] = {
    "collagen_ecm": "dermis",
    "antioxidant": "epidermis",
    "pigmentation": "epidermis",
}


class CosmeticFunctions(BaseModel):
    primary: CosmeticFunction = "collagen_ecm"
    secondary: List[CosmeticFunction] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _check_secondary(self) -> "CosmeticFunctions":
        if len(set(self.secondary)) != len(self.secondary):
            raise ValueError("functions.secondary must not repeat a function")
        if self.primary in self.secondary:
            raise ValueError("functions.secondary must not include the primary function")
        return self


class CosmeticApplication(BaseModel):
    area: ApplicationArea = "face"
    sensitive_skin: bool = False


class CosmeticProduct(BaseModel):
    ph: float = Field(default=5.5, ge=4.0, le=7.5)


class CosmeticPeptide(BaseModel):
    min_length: int = Field(default=6, ge=6, le=20)
    max_length: int = Field(default=12, ge=2, le=20)

    @model_validator(mode="after")
    def _check_length_bounds(self) -> "CosmeticPeptide":
        if self.min_length > self.max_length:
            raise ValueError("peptide.min_length must be <= peptide.max_length")
        return self


class CosmeticBrief(BaseModel):
    """Stage 1 brief for cosmetics and skin beauty; the single configuration the cosmetic stages read.

    Every field has a default, so an empty brief is valid. `target_depth` is filled from the primary function when
    omitted, and `safety_level` is derived, never supplied.
    """

    functions: CosmeticFunctions = Field(default_factory=CosmeticFunctions)
    target_depth: TargetDepth | None = None
    application: CosmeticApplication = Field(default_factory=CosmeticApplication)
    product: CosmeticProduct = Field(default_factory=CosmeticProduct)
    peptide: CosmeticPeptide = Field(default_factory=CosmeticPeptide)
    @model_validator(mode="after")
    def _default_target_depth(self) -> "CosmeticBrief":
        if self.target_depth is None:
            self.target_depth = DEFAULT_DEPTH_BY_FUNCTION[self.functions.primary]
        return self

    @computed_field
    @property
    def safety_level(self) -> Literal["strict", "standard"]:
        """`strict` for the eye area or sensitive skin, `standard` otherwise."""
        if self.application.area == "eye_area" or self.application.sensitive_skin:
            return "strict"
        return "standard"

    @classmethod
    def load(cls, path: str | Path) -> "CosmeticBrief":
        """Load a cosmetic brief from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Brief file not found: {path}")
        with open(path, "r") as f:
            return cls(**json.load(f))
