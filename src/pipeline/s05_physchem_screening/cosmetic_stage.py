# Stage 5 (cosmetics): physicochemical screening in order B (properties) -> C (rule rejects) -> D (rule flags) -> E (models).
from __future__ import annotations

from typing import Any

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext, StageError
from pipeline.feature_extractor import PROPY_AVAILABLE, PROPY_UNAVAILABLE_REASON
from pipeline.s05_physchem_screening.cosmetic_physchem import (
    Thresholds,
    compute_properties,
    has_lipid,
    rule_flags,
    rule_rejects,
    seed_reference,
)
from pipeline.s05_physchem_screening.stage import (
    AGGREGATION_MIN_LENGTH,
    _get_aggregation_model,
    _get_solubility_model,
)
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage5CosmeticParams

logger = get_logger(__name__)

def _seed_set(candidates: list[Candidate], given: list[str] | None) -> list[dict[str, Any]]:
    """Seed peptides for the percentile bounds: the stage param if given, else the set Stage 4 attached to its candidates."""
    if given:
        return [{"sequence": s, "modifications": {}} for s in dict.fromkeys(given)]
    seeds: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for seed in candidate.predictions.get("seed_set", []):
            seeds.setdefault(seed["sequence"], seed)
    return list(seeds.values())


class CosmeticStage5(CandidateStage):
    """Cheapest-first screen; a candidate rejected at one step is not carried to the next. Opt-in."""

    name = "s05_cosmetic_physchem_screening"
    produces = {"sequence", "hydrophobic_moment", "net_charge", "molecular_weight"}

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        params = Stage5CosmeticParams.model_validate(config.params)
        ph = params.product_ph or getattr(getattr(ctx.brief, "product", None), "ph", None)
        if ph is None:
            raise StageError(self.name, "set product_ph in the stage params or the cosmetic brief.")
        if any(not c.sequence for c in candidates):
            raise StageError(self.name, "received a candidate with no sequence")
        thresholds = Thresholds.from_params(params)
        solvent = params.solubility_solvent
        aggregation_flag_min = params.aggregation_flag_min
        solubility_threshold = params.solubility_threshold

        seeds = _seed_set(candidates, params.seed_sequences)
        reference = seed_reference(seeds, ph, thresholds)
        if reference is None:
            logger.warning(
                "s05_cosmetic.no_seed_reference",
                extra={"n_seeds": len(seeds)},
            )

        # B and C, then D, for every candidate; E only for those still standing.
        standing: list[tuple[Candidate, dict, dict, dict]] = []
        rejected: dict[str, int] = {}
        for candidate in candidates:
            sequence = candidate.sequence
            mods = candidate.predictions.get("modifications") or {}
            props = compute_properties(sequence, ph, mods)
            reasons = rule_rejects(sequence, props, reference, thresholds)
            if reasons:
                for reason in reasons:
                    rejected[reason] = rejected.get(reason, 0) + 1
                logger.info("s05_cosmetic.reject", extra={"candidate_id": candidate.id, "reasons": reasons})
                continue
            flags = rule_flags(sequence, props, mods, thresholds)
            standing.append((candidate, mods, props, flags))

        if standing:
            # Warm the shared ESM2 cache once for the solubility model.
            ctx.feature_extractor.get_esm2_embedding_batch([c.sequence for c, *_ in standing])

        survivors: list[Candidate] = []
        for candidate, mods, props, flags in standing:
            sequence = candidate.sequence
            not_screened: list[str] = []

            if len(sequence) < AGGREGATION_MIN_LENGTH:
                aggregation = {"score": None, "status": "skipped_too_short"}
                not_screened.append("aggregation")
            elif not PROPY_AVAILABLE:
                aggregation = {"score": None, "status": "unavailable", "reason": PROPY_UNAVAILABLE_REASON}
                not_screened.append("aggregation")
            else:
                score = _get_aggregation_model().predict_aggregation(sequence, ctx.feature_extractor)
                aggregation = {"score": score, "status": "ok", "flag_min": aggregation_flag_min}
                if score > aggregation_flag_min:
                    flags["aggregation"] = score

            score = _get_solubility_model().predict_proba(sequence, solvent, ctx.feature_extractor)
            soluble = int(score >= solubility_threshold)
            solubility = {
                "score": score,
                "solvent": solvent,
                "threshold": solubility_threshold,
                "decision": soluble,
            }
            if not soluble:
                if has_lipid(mods):
                    # The model sees the sequence only, not the lipid tail.
                    flags["solubility_unreliable"] = score
                else:
                    rejected["solubility_insoluble"] = rejected.get("solubility_insoluble", 0) + 1
                    logger.info(
                        "s05_cosmetic.reject",
                        extra={"candidate_id": candidate.id, "reasons": ["solubility_insoluble"]},
                    )
                    continue

            candidate.predictions.update(
                {
                    **props,
                    "length": len(sequence),
                    "aggregation_tendency": aggregation,
                    "solubility": solubility,
                    "physchem_flags": flags,
                    "physchem_flag_count": len(flags),
                    "screening_verdict": {
                        "overall": "proceed",
                        "flags": sorted(flags),
                        "not_screened": not_screened,
                        "seed_reference": reference,
                    },
                }
            )
            survivors.append(candidate)

        logger.info(
            "s05_cosmetic.done",
            extra={"n_in": len(candidates), "n_out": len(survivors), "rejected": rejected},
        )
        return survivors
