from abc import ABC, abstractmethod
from typing import Any

from src.schemas.candidate import Candidate


class Stage4Route(ABC):
    """Abstract base class for Stage 4 routing strategies (A/B/C/D), mirroring `CandidateStage.run()`; routes are not stages and get no audit entry of their own."""

    _config: dict[str, Any] = {}

    def __init__(self, config: dict[str, Any] = {}) -> None:
        self._config: dict[str, Any] = config

    @property
    def config(self) -> dict[str, Any]:
        """The run config for this route. Set by `Stage4.run()`."""
        return self._config

    @abstractmethod
    def run(self) -> list[Candidate]:
        """Execute the routing logic. Return newly generated candidates."""
