# Stage 8 (cosmetics): irritation rule and recorded (not enforced) penetration rule, safety and stability models, then skin-protease stability on survivors.
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from common.logging import get_logger
from common.model_registry import ModelRef
from pipeline.base import RunContext, StageError
from pipeline.s05_physchem_screening.cosmetic_physchem import has_lipid
from pipeline.s08_safety_developability.cosmetic_safety import (
    REJECT_TIERS,
    irritation_cutoffs,
    irritation_score,
    irritation_tier,
    penetration,
    resolve_cell_type,
    skin_proteases,
)
from pipeline.s08_safety_developability.metabolic_stage import (
    HALF_LIFE_MAX_LENGTH,
    HALF_LIFE_MIN_LENGTH,
    _get_half_life_model,
)
from pipeline.s08_safety_developability.stage import (
    Stage8,
    _get_cleavage_model,
    _get_cytotoxicity_model,
    _get_hemolysis_v2_model,
)
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage8CosmeticParams

from model_store.allergenicity_predictor_v1 import AllergenicityPredictor  # noqa: E402
from model_store.allergenicity_predictor_v1.predictor import (  # noqa: E402
    MAX_LENGTH as ALLERGENICITY_MAX_LENGTH,
    TRAINED_MIN_LENGTH as ALLERGENICITY_TRAINED_MIN_LENGTH,
)
from model_store.half_life_predictor_v1 import HalfLifeContext  # noqa: E402
from model_store.skin_sensitization_predictor_v1 import (  # noqa: E402
    DECISION_THRESHOLD as SKIN_SENSITIZATION_THRESHOLD,
    SkinSensitizationPredictor,
)

logger = get_logger(__name__)

STANDARD_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")

_allergenicity_model: AllergenicityPredictor | None = None


def _get_allergenicity_model() -> AllergenicityPredictor:
    global _allergenicity_model
    if _allergenicity_model is None:
        _allergenicity_model = AllergenicityPredictor()
    return _allergenicity_model


_skin_sensitization_model: SkinSensitizationPredictor | None = None


def _get_skin_sensitization_model() -> SkinSensitizationPredictor:
    global _skin_sensitization_model
    if _skin_sensitization_model is None:
        _skin_sensitization_model = SkinSensitizationPredictor()
    return _skin_sensitization_model


class CosmeticStage8(Stage8):
    """Opt-in; replaces the wound-healing safety stage for a cosmetic run. Reuses Stage 8's cleavage-stability calculation."""

    name = "s08_cosmetic_safety_developability"
    requires = {"structure", "hydrophobic_moment", "net_charge", "molecular_weight"}
    produces = {"cleavage_stability", "half_life", "irritation"}

    def models_used(self) -> list[ModelRef]:
        return [
            ModelRef(name="cytotoxicity_predictor", version="v1"),
            ModelRef(name="skin_sensitization_predictor", version=_get_skin_sensitization_model().version),
            ModelRef(name="hemolysis_predictor", version="v2"),
            ModelRef(name="allergenicity_model", version=_get_allergenicity_model().version),
            ModelRef(name="half_life_predictor", version="v1"),
            ModelRef(name="cleavage_site_predictor", version="v2"),
        ]

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        params = Stage8CosmeticParams.model_validate(config.params)
        depth = params.target_depth or getattr(ctx.brief, "target_depth", None)
        level = params.safety_level or getattr(ctx.brief, "safety_level", None)
        ph = params.product_ph or getattr(getattr(ctx.brief, "product", None), "ph", None)
        missing = [n for n, v in (("target_depth", depth), ("safety_level", level), ("product_ph", ph)) if v is None]
        if missing:
            raise StageError(self.name, f"set {missing} in the stage params or the cosmetic brief.")
        if any(not c.sequence for c in candidates):
            raise StageError(self.name, "received a candidate with no sequence")

        fe = ctx.feature_extractor
        rejected: dict[str, list[str]] = {}
        started = datetime.now(timezone.utc).isoformat()

        def reject(candidate: Candidate, reason: str) -> None:
            rejected.setdefault(reason, []).append(candidate.id)
            logger.info("s08_cosmetic.reject", extra={"candidate_id": candidate.id, "reason": reason})

        # A. Rule checks.
        standing = self._rule_checks(candidates, params, depth, level, ph, reject)

        # B. Safety and stability models, cheapest first; later models only see candidates still standing.
        standing = self._skin_sensitization(standing, params, reject)
        standing = self._hemolysis(standing, params, fe, reject)
        standing = self._half_life(standing, params, fe, reject)
        standing = self._allergenicity(standing, params, fe, reject)
        standing = self._cytotoxicity(standing, params, fe, reject)

        # C. Skin-protease stability, last, survivors only.
        survivors = self._cleavage(standing, params, depth, reject)

        for candidate in survivors:
            self._finish(candidate, level, started)
        logger.info(
            "s08_cosmetic.done",
            extra={
                "n_in": len(candidates),
                "n_out": len(survivors),
                "rejected": {reason: len(ids) for reason, ids in rejected.items()},
            },
        )
        return survivors

    # A. Rule checks.

    def _rule_checks(self, candidates, params, depth, level, ph, reject) -> list[Candidate]:
        seed_set = next((c.predictions["seed_set"] for c in candidates if c.predictions.get("seed_set")), [])
        cutoffs = irritation_cutoffs(
            seed_set, ph, params.irritation_percentile_medium, params.irritation_percentile_high
        )
        if cutoffs is None:
            logger.warning("s08_cosmetic.no_irritation_reference", extra={"n_seeds": len(seed_set)})
        reject_tiers = REJECT_TIERS[level]

        standing: list[Candidate] = []
        for candidate in candidates:
            p = candidate.predictions
            mods = p.get("modifications")
            score = irritation_score(p["hydrophobic_moment"], p["net_charge"])
            if cutoffs is None:
                irritation = {"score": score, "tier": None, "status": "not_screened", "safety_level": level}
            else:
                tier = irritation_tier(score, cutoffs, has_lipid(mods))
                irritation = {
                    "score": score,
                    "tier": tier,
                    "lipid_tier_shift": has_lipid(mods),
                    "cutoffs": cutoffs,
                    "safety_level": level,
                    "status": "ok",
                }
            p["irritation"] = irritation
            p["penetration"] = penetration(depth, p["molecular_weight"], mods, params.penetration_epidermis_mw_max)
            if irritation["tier"] in reject_tiers:
                reject(candidate, f"irritation_{irritation['tier']}")
            # Penetration filter off until a penetration model exists (docs/TODO.md); the rule result is still recorded.
            # elif not p["penetration"]["passed"]:
            #     reject(candidate, "penetration")
            else:
                standing.append(candidate)
        return standing

    # B. Models.

    def _skin_sensitization(self, candidates, params, reject) -> list[Candidate]:
        """Sensitizer at the model's fixed 0.55; `reject_skin_sensitizers` decides whether that rejects or only flags."""
        if not candidates:
            return []
        results = _get_skin_sensitization_model().predict_batch(
            [c.sequence for c in candidates], [c.predictions.get("modifications") for c in candidates]
        )
        standing = []
        for candidate, result in zip(candidates, results):
            sensitizer = result["p_sensitizer"] >= SKIN_SENSITIZATION_THRESHOLD
            candidate.predictions["skin_sensitization"] = {
                "p_sensitizer": result["p_sensitizer"],
                "members": result["members"],
                "threshold": SKIN_SENSITIZATION_THRESHOLD,
                "sensitizer": sensitizer,
                "rejects": params.reject_skin_sensitizers,
                "status": "ok",
            }
            if sensitizer and params.reject_skin_sensitizers:
                reject(candidate, "skin_sensitization")
            else:
                standing.append(candidate)
        return standing

    def _hemolysis(self, candidates, params, fe, reject) -> list[Candidate]:
        if not candidates:
            return []
        probabilities = _get_hemolysis_v2_model().predict_hemolytic_proba_batch(
            [c.sequence for c in candidates], fe
        )
        standing = []
        for candidate, probability in zip(candidates, probabilities):
            candidate.predictions["hemolysis"] = {
                "p_hemolytic": probability,
                "reject_min": params.hemolysis_reject_min,
                "status": "ok",
            }
            if probability >= params.hemolysis_reject_min:
                reject(candidate, "hemolysis")
            else:
                standing.append(candidate)
        return standing

    def _half_life(self, candidates, params, fe, reject) -> list[Candidate]:
        if not candidates:
            return []
        scorable = [
            c for c in candidates
            if HALF_LIFE_MIN_LENGTH <= len(c.sequence) <= HALF_LIFE_MAX_LENGTH
            and set(c.sequence) <= STANDARD_RESIDUES
        ]
        values = {}
        if scorable:
            logs = _get_half_life_model().predict_log10_half_life_batch(
                [c.sequence for c in scorable], fe, HalfLifeContext()
            )
            values = {c.id: v for c, v in zip(scorable, logs)}
        standing = []
        for candidate in candidates:
            value = values.get(candidate.id)
            if value is None:
                candidate.predictions["half_life"] = {
                    "log10_seconds": None,
                    "status": "skipped_unsupported_sequence",
                    "reason": f"half-life model covers {HALF_LIFE_MIN_LENGTH}-{HALF_LIFE_MAX_LENGTH} standard residues",
                }
                standing.append(candidate)
                continue
            candidate.predictions["half_life"] = {
                "log10_seconds": float(value),
                "reject_min": params.half_life_reject_min,
                "context": "human plasma, in vitro, unmodified",
                "status": "ok",
            }
            if value < params.half_life_reject_min:
                reject(candidate, "half_life")
            else:
                standing.append(candidate)
        return standing

    def _allergenicity(self, candidates, params, fe, reject) -> list[Candidate]:
        if not candidates:
            return []
        model = _get_allergenicity_model()
        # The model saw only full-length proteins, so shorter sequences are not scored: its output there is noise.
        scorable = [
            c for c in candidates
            if params.allergenicity_min_length <= len(c.sequence) <= ALLERGENICITY_MAX_LENGTH
        ]
        results = {}
        if scorable:
            scored = model.predict_batch([c.sequence for c in scorable], fe)
            results = {c.id: r for c, r in zip(scorable, scored)}
        standing = []
        for candidate in candidates:
            result = results.get(candidate.id)
            if result is None:
                candidate.predictions["allergenicity"] = {
                    "p_allergen": None,
                    "status": "outside_scored_length_range",
                    "min_length": params.allergenicity_min_length,
                    "trained_min_length": ALLERGENICITY_TRAINED_MIN_LENGTH,
                    "model_version": model.version,
                }
                standing.append(candidate)
                continue
            candidate.predictions["allergenicity"] = {
                "p_allergen": result["p_allergen"],
                "members": result["members"],
                "reject_min": params.allergenicity_reject_min,
                "status": "ok",
                "model_version": model.version,
            }
            if result["p_allergen"] >= params.allergenicity_reject_min:
                reject(candidate, "allergenicity")
            else:
                standing.append(candidate)
        return standing

    def _cytotoxicity(self, candidates, params, fe, reject) -> list[Candidate]:
        if not candidates:
            return []
        model = _get_cytotoxicity_model()
        model._load()
        resolved = [resolve_cell_type(c, sorted(model.cell_type_to_id)) for c in params.cytotoxicity_cell_types]
        for item in resolved:
            if not item["exact"]:
                logger.warning("s08_cosmetic.cell_type_substituted", extra=item)
        standing = []
        for candidate in candidates:
            per_cell = {}
            for item in resolved:
                per_cell[item["used"]] = {
                    "p_cytotoxic": model.predict_cytotoxicity(candidate.sequence, fe, cell_type=item["used"]),
                    "requested": item["requested"],
                    "exact": item["exact"],
                }
            worst = max(v["p_cytotoxic"] for v in per_cell.values())
            candidate.predictions["cytotoxicity"] = {
                "by_cell_type": per_cell,
                "max_p_cytotoxic": worst,
                "reject_min": params.cytotoxicity_reject_min,
                "status": "ok",
            }
            if worst >= params.cytotoxicity_reject_min:
                reject(candidate, "cytotoxicity")
            else:
                standing.append(candidate)
        return standing

    # C. Skin-protease stability.

    def _wound_relevant_accessions(self, model, config_params: dict) -> list[str]:
        """The enzymes chosen for this depth that the panel actually has."""
        available = {row["enzyme_uniprot"] for row in model.available_enzymes()}
        return sorted(set(config_params["protease_accessions"]) & available)

    def _cleavage(self, candidates, params, depth, reject) -> list[Candidate]:
        if not candidates:
            return []
        model = _get_cleavage_model()
        requested = skin_proteases(depth, params.skin_protease_accessions, params.dermis_extra_protease_accessions)
        available = {row["enzyme_uniprot"] for row in model.available_enzymes()}
        missing = [a for a in requested if a not in available]
        if missing:
            logger.warning("s08_cosmetic.enzymes_missing_from_panel", extra={"accessions": missing})
        results = self.compute_cleavage_stability_batch(
            candidates,
            model,
            {"protease_accessions": requested, "cleavage_decay_rate": params.cleavage_decay_rate},
        )
        standing = []
        for candidate, result in zip(candidates, results):
            result["enzymes_requested"] = requested
            result["enzymes_missing_from_panel"] = missing
            result["reject_min"] = params.cleavage_stability_reject_min
            candidate.predictions["cleavage_stability"] = result
            score = result["score"]
            if score is None:
                # Not screened is never a pass: carried as a flag.
                result["status"] = "no_enzymes_in_panel"
                standing.append(candidate)
            elif score < params.cleavage_stability_reject_min:
                reject(candidate, "cleavage_stability")
            else:
                standing.append(candidate)
        return standing

    # E. Decision and the record carried to Stage 9 / 11.

    def _finish(self, candidate: Candidate, level: str, started: str) -> None:
        p = candidate.predictions
        flags = []
        if p["cleavage_stability"]["score"] is None:
            flags.append("cleavage_not_screened")
        if p["skin_sensitization"]["sensitizer"]:
            flags.append("skin_sensitizer")
        if p["allergenicity"]["status"] != "ok":
            flags.append("allergenicity_not_screened")
        if p["half_life"]["status"] != "ok":
            flags.append("half_life_not_screened")
        if p["irritation"]["status"] == "not_screened":
            flags.append("irritation_not_screened")
        p["safety_flags"] = flags
        p["safety_penalties"] = {
            "physchem_flag_count": p.get("physchem_flag_count", 0),
            "cleavage_stability": p["cleavage_stability"]["score"],
            "half_life_log10_seconds": p["half_life"]["log10_seconds"],
            "irritation_tier": p["irritation"]["tier"],
            "flags": flags,
        }
        p["structure_ref"] = {
            "source": "s07_structure_mechanism",
            "location": "predictions.structure.pdb_str",
            "n_residues": len(candidate.sequence),
        }
        p["safety_verdict"] = {
            "overall": "proceed",
            "safety_level": level,
            "reject_reason": None,
            "model_versions": {ref.name: ref.version for ref in self.models_used()},
            "timestamp": started,
        }
