# The candidate peptide object passed between stages.
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class Candidate(BaseModel):
    id: str
    sequence: Optional[str] = None
    predictions: dict[str, Any] = Field(default_factory=dict)

    def present_fields(self) -> set[str]:
        """Field names populated on this candidate, for precondition checks."""
        fields = {name for name, value in self.__dict__.items() if value is not None}
        fields |= set(self.predictions.keys())
        return fields
