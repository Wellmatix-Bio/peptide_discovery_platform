# Model name, version, checksum.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRef:
    name: str
    version: str
    checksum: str | None = None


class ModelRegistry:
    """No-op stub. Real implementation should load model metadata from `store_path`."""

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
