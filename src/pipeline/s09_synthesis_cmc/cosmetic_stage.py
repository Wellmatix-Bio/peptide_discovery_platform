# Stage 9 (cosmetics): synthesis-feasibility gate on the sequence, plus synthesis rule flags that never reject.
from __future__ import annotations

from common.logging import get_logger
from common.model_registry import ModelRef
from pipeline.base import CandidateStage, RunContext, StageError
from pipeline.s09_synthesis_cmc.stage import build_model
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage9CosmeticParams

logger = get_logger(__name__)


def synthesis_flags(sequence: str) -> dict[str, list[int]]:
    """Rule flags with the 0-based positions that trigger them; a flag is absent when it does not apply."""
    flags: dict[str, list[int]] = {}
    aspartimide = [i for i in range(len(sequence) - 1) if sequence[i : i + 2] == "DG"]
    if aspartimide:
        flags["aspartimide_risk"] = aspartimide
    oxidation = [i for i, aa in enumerate(sequence) if aa in "MW"]
    if oxidation:
        flags["met_trp_cleavage_oxidation"] = oxidation
    return flags


class CosmeticStage9(CandidateStage):
    """Opt-in; replaces the wound-healing synthesis stage for a cosmetic run."""

    name = "s09_cosmetic_synthesis_cmc"
    produces = {"synthesis_feasibility"}

    def models_used(self) -> list[ModelRef]:
        return [ModelRef(name="synthesis_feasibility_predictor", version="v1")]

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        params = Stage9CosmeticParams.model_validate(config.params)
        if any(not c.sequence for c in candidates):
            raise StageError(self.name, "received a candidate with no sequence")
        if not candidates:
            return []
        fe = ctx.feature_extractor
        fe.get_esm2_embedding_batch([c.sequence for c in candidates])
        model = build_model()

        survivors: list[Candidate] = []
        for candidate in candidates:
            probability = model.predict_proba(candidate.sequence, fe)
            flags = synthesis_flags(candidate.sequence)
            candidate.predictions["synthesis_feasibility"] = {
                "probability": probability,
                "min": params.synthesis_feasibility_min,
                "pass": probability >= params.synthesis_feasibility_min,
                "flags": flags,
            }
            if probability < params.synthesis_feasibility_min:
                logger.info(
                    "s09_cosmetic.reject",
                    extra={"candidate_id": candidate.id, "probability": probability},
                )
                continue
            survivors.append(candidate)
        logger.info(
            "s09_cosmetic.done", extra={"n_in": len(candidates), "n_out": len(survivors)}
        )
        return survivors
