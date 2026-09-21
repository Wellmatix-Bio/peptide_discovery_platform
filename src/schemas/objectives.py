# Biological objective vector across the five target peptide functions.
from pydantic import BaseModel


class DeficitScore(BaseModel):
    severity: float
    drivers: list[str] = []


class ObjectiveVector(BaseModel):
    deficits: dict[str, DeficitScore]

    def to_vector(self) -> list[float]:
        """Convert the deficits to a vector of severity scores."""
        # placeholder implementation; replace with actual logic.
        return [score.severity for score in self.deficits.values()]