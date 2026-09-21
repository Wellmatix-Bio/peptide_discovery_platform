# Known-peptide knowledge base used for reference-guided generation.
from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel


class KnowledgeBase(BaseModel):
    """No-op stub. Real implementation should hold standardized known-peptide records."""

    source_path: str

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeBase:
        return cls(source_path=str(path))
