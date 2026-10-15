# Stage 8 (metabolism / weight control): safety and developability with incretin-relevant proteases and plasma half-life.
from __future__ import annotations

import sys
from pathlib import Path

from tqdm import tqdm

from common.gpu import release_stage_models
from common.logging import get_logger
from common.model_registry import ModelRef
from pipeline.base import RunContext
from pipeline.s08_safety_developability.stage import Stage8, build_models
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage8MetabolicParams

logger = get_logger(__name__)

MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"
if str(MODEL_STORE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_STORE_DIR))

from model_store.half_life_predictor_v1 import HalfLifeContext, HalfLifePredictor  # noqa: E402

#: Single source of the thresholds, so the run-config schema and the stage cannot drift apart.
DEFAULT_THRESHOLDS = Stage8MetabolicParams().model_dump()

# Incretin-degrading proteases: DPP-4, neprilysin (NEP) and prolyl carboxypeptidase (PPCE).
METABOLIC_PROTEASE_ACCESSIONS = {"P27487", "P08473", "P48147"}

# The half-life model's trained length range.
HALF_LIFE_MIN_LENGTH = 5
HALF_LIFE_MAX_LENGTH = 50

_half_life_model: HalfLifePredictor | None = None


def _get_half_life_model() -> HalfLifePredictor:
    global _half_life_model
    if _half_life_model is None:
        _half_life_model = HalfLifePredictor()
    return _half_life_model


class Stage8Metabolic(Stage8):
    """Stage 8 with the metabolism thresholds.

    Differences from Stage 8: hemolysis is optional (a null cutoff turns it off), the protease panel is DPP-4 /
    NEP / PPCE, solubility is flagged below 0.173 in ultrapure water, and predicted plasma half-life below the
    client's minimum rejects. Cytotoxicity (DRAMP_aggregate) and aggregation are unchanged. Opt-in: it runs
    only when named in the run config, normally in place of `s08_safety_developability`.
    """

    name = "s08_metabolic_safety_developability"
    produces = {
        "hemolysis",
        "cytotoxicity",
        "solubility",
        "aggregation_tendency",
        "cleavage_stability",
        "plasma_half_life",
    }

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        """Scores the metabolism safety panel, then rejects candidates failing any hard threshold."""
        params = {**DEFAULT_THRESHOLDS, **config.params}
        # A metabolic brief is the single configuration: it overrides the hemolysis switch and half-life minimum.
        brief = ctx.brief
        if getattr(brief, "hemolysis_screening", True) is False:
            params["hemolysis_phc50_reject_max"] = None
        if getattr(brief, "min_plasma_half_life_minutes", None) is not None:
            params["plasma_half_life_min_minutes"] = brief.min_plasma_half_life_minutes
        models = build_models(params)
        half_life_model = _get_half_life_model()

        sequences = [candidate.sequence for candidate in candidates]
        if not all(sequences):
            raise ValueError("Stage 8 received a candidate with no sequence")

        feature_extractor = ctx.feature_extractor
        # Warm the shared ESM2 cache once for the whole stage.
        feature_extractor.get_esm2_embedding_batch(sequences)

        hemolysis_enabled = params["hemolysis_phc50_reject_max"] is not None
        hemolysis_results = (
            self.compute_hemolysis_batch(sequences, models.hemolysis, feature_extractor)
            if hemolysis_enabled
            else [{"phc50": None, "status": "disabled"}] * len(sequences)
        )
        half_life_results = self.compute_half_life_batch(
            sequences, half_life_model, feature_extractor
        )

        survivors: list[Candidate] = []
        for candidate, hemolysis_result, half_life_result in zip(
            tqdm(candidates, desc="Stage 8 (metabolic)"),
            hemolysis_results,
            half_life_results,
        ):
            sequence = candidate.sequence
            candidate.predictions.update(
                {
                    "hemolysis": hemolysis_result,
                    "cytotoxicity": self.compute_cytotoxicity(
                        sequence,
                        models.cytotoxicity,
                        params["cytotoxicity_cell_type"],
                        feature_extractor,
                    ),
                    "solubility": self.compute_solubility(
                        sequence,
                        models.solubility,
                        params["solubility_solvent"],
                        feature_extractor,
                    ),
                    "aggregation_tendency": self.compute_aggregation(
                        sequence, models.aggregation, feature_extractor
                    ),
                    "plasma_half_life": half_life_result,
                }
            )
            candidate.predictions["cleavage_stability"] = (
                self.compute_cleavage_stability(candidate, models.cleavage, params)
            )

            verdict = self.compute_safety_verdict(candidate.predictions, params)
            candidate.predictions["safety_verdict"] = verdict

            if verdict["overall"] == "reject":
                logger.info(
                    "s08_metabolic.reject",
                    extra={
                        "candidate_id": candidate.id,
                        "properties": verdict["properties"],
                    },
                )
                continue
            survivors.append(candidate)

        return survivors

    def models_used(self) -> list[ModelRef]:
        return super().models_used() + [
            ModelRef(name="half_life_predictor", version="v1")
        ]

    def release_models(self) -> None:
        super().release_models()
        release_stage_models(globals(), stage_name=self.name)

    def compute_half_life_batch(
        self, sequences: list[str], model: HalfLifePredictor, feature_extractor
    ) -> list[dict]:
        """Plasma half-life in minutes (human plasma, in vitro, unmodified); out-of-range sequences are not scored."""
        results: list[dict] = [
            {
                "minutes": None,
                "status": "skipped_unsupported_sequence",
                "reason": (
                    f"half-life model covers {HALF_LIFE_MIN_LENGTH}-{HALF_LIFE_MAX_LENGTH} standard"
                    " residues"
                ),
            }
        ] * len(sequences)
        scorable = [
            i
            for i, s in enumerate(sequences)
            if HALF_LIFE_MIN_LENGTH <= len(s) <= HALF_LIFE_MAX_LENGTH
            and set(s) <= set("ACDEFGHIKLMNPQRSTVWY")
        ]
        if scorable:
            seconds = model.predict_half_life_seconds_batch(
                [sequences[i] for i in scorable],
                feature_extractor,
                HalfLifeContext(),
            )
            for i, value in zip(scorable, seconds):
                results[i] = {
                    "minutes": value / 60.0,
                    "matrix": "human plasma",
                    "status": "ok",
                }
        return results

    def _wound_relevant_accessions(self, model, config_params: dict) -> list[str]:
        """Restricts the enzyme panel to the metabolism proteases; overridable via config.params["protease_accessions"]."""
        override = config_params.get("protease_accessions")
        allowlist = set(override) if override else METABOLIC_PROTEASE_ACCESSIONS
        available = {row["enzyme_uniprot"] for row in model.available_enzymes()}
        return sorted(allowlist & available)

    def compute_safety_verdict(self, predictions: dict, config_params: dict) -> dict:
        """Parent verdict with hemolysis made optional and plasma half-life added as a hard reject."""
        t = {**DEFAULT_THRESHOLDS, **config_params}
        hemolysis_cutoff = t["hemolysis_phc50_reject_max"]

        # The parent reads hemolysis unconditionally; hand it a cutoff that cannot trip, then overwrite below.
        verdict = super().compute_safety_verdict(
            {
                **predictions,
                "hemolysis": {"phc50": predictions["hemolysis"]["phc50"] or 0.0},
            },
            {**t, "hemolysis_phc50_reject_max": hemolysis_cutoff or 0.0},
        )
        properties = verdict["properties"]

        if hemolysis_cutoff is None:
            properties["hemolysis"] = "disabled"

        half_life = predictions["plasma_half_life"]["minutes"]
        if half_life is None:
            properties["plasma_half_life"] = "not_screened"
        elif half_life < t["plasma_half_life_min_minutes"]:
            properties["plasma_half_life"] = "reject"
        else:
            properties["plasma_half_life"] = "pass"

        return self._roll_up(properties)

    @staticmethod
    def _roll_up(properties: dict[str, str]) -> dict:
        if "reject" in properties.values():
            overall = "reject"
        elif "flag" in properties.values() or "not_screened" in properties.values():
            # An unrun screen makes the verdict a flag, never a pass or a reject.
            overall = "flag"
        else:
            overall = "pass"

        verdict = {"properties": properties, "overall": overall}
        unscreened = sorted(k for k, v in properties.items() if v == "not_screened")
        if unscreened:
            verdict["not_screened"] = unscreened
            verdict["not_screened_note"] = (
                "These checks did not run, so this candidate was not cleared by them. Not checked"
                " is not the same as passed."
            )
        return verdict
