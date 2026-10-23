# Stage 8 (metabolism / weight control): safety and developability with incretin-relevant proteases and plasma half-life.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.gpu import release_stage_models
from common.logging import get_logger
from common.model_registry import ModelRef
from pipeline.base import RunContext
from pipeline.s08_safety_developability.stage import (
    Stage8,
    Stage8Models,
    _get_aggregation_model,
    _get_cytotoxicity_model,
    _get_solubility_model,
)
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage8MetabolicParams

logger = get_logger(__name__)

MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"
if str(MODEL_STORE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_STORE_DIR))

from model_store.cleavage_site_predictor_v2 import CleavageSitePredictor  # noqa: E402
from model_store.half_life_predictor_v1 import HalfLifeContext, HalfLifePredictor  # noqa: E402
from model_store.hemolysis_predictor_v2 import HemolysisPredictor  # noqa: E402

#: Single source of the thresholds, so the run-config schema and the stage cannot drift apart.
DEFAULT_THRESHOLDS = Stage8MetabolicParams().model_dump()

# Incretin-degrading proteases DPP-4, NEP and PPCE (prolyl carboxypeptidase), all in cleavage_site_predictor_v2's panel.
METABOLIC_PROTEASE_ACCESSIONS = {"P27487", "P08473", "P42785"}

STANDARD_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")

# The half-life model's trained length range.
HALF_LIFE_MIN_LENGTH = 5
HALF_LIFE_MAX_LENGTH = 50

_half_life_model: HalfLifePredictor | None = None
_hemolysis_v2_model: HemolysisPredictor | None = None
_cleavage_model: CleavageSitePredictor | None = None


def _get_half_life_model() -> HalfLifePredictor:
    global _half_life_model
    if _half_life_model is None:
        _half_life_model = HalfLifePredictor()
    return _half_life_model


def _get_hemolysis_v2_model() -> HemolysisPredictor:
    """hemolysis_predictor_v2: the classifier output filters, the regression output ranks."""
    global _hemolysis_v2_model
    if _hemolysis_v2_model is None:
        _hemolysis_v2_model = HemolysisPredictor()
    return _hemolysis_v2_model


def _get_cleavage_model() -> CleavageSitePredictor:
    """cleavage_site_predictor_v2, the only cleavage model this stage uses."""
    global _cleavage_model
    if _cleavage_model is None:
        _cleavage_model = CleavageSitePredictor()
    return _cleavage_model


def build_models() -> Stage8Models:
    """Stage 8's model bundle with the v2 hemolysis and cleavage models."""
    return Stage8Models(
        hemolysis=_get_hemolysis_v2_model(),
        cytotoxicity=_get_cytotoxicity_model(),
        solubility=_get_solubility_model(),
        aggregation=_get_aggregation_model(),
        cleavage=_get_cleavage_model(),
    )


class Stage8Metabolic(Stage8):
    """Stage 8 with the metabolism thresholds.

    Differences from Stage 8: hemolysis comes from hemolysis_predictor_v2 and is optional (the classifier's
    P(hemolytic) filters, the regression's pHC50 is kept for ranking; a null cutoff turns the filter off), cleavage
    stability comes from cleavage_site_predictor_v2 against DPP-4, NEP and PPCE, solubility is flagged below 0.173
    in ultrapure water, and predicted plasma half-life below the client's minimum rejects. Cytotoxicity (DRAMP_aggregate) and aggregation are unchanged. Opt-in: it runs
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
        hemolysis_enabled = params["hemolysis_screening"]
        if not hemolysis_enabled:
            params["hemolysis_probability_reject_max"] = None
        models = build_models()
        half_life_model = _get_half_life_model()

        sequences = [candidate.sequence for candidate in candidates]
        if not all(sequences):
            raise ValueError("Stage 8 received a candidate with no sequence")

        feature_extractor = ctx.feature_extractor
        # Warm the shared ESM2 cache once for the whole stage.
        feature_extractor.get_esm2_embedding_batch(sequences)

        hemolysis_results = (
            self.compute_hemolysis_batch(sequences, models.hemolysis, feature_extractor)
            if hemolysis_enabled
            else [{"phc50": None, "p_hemolytic": None, "status": "disabled"}] * len(sequences)
        )
        half_life_results = self.compute_half_life_batch(
            sequences, half_life_model, feature_extractor
        )

        cleavage_results = self.compute_cleavage_stability_batch(
            candidates, models.cleavage, params
        )

        survivors: list[Candidate] = []
        for candidate, hemolysis_result, half_life_result, cleavage_result in zip(
            tqdm(candidates, desc="Stage 8 (metabolic)"),
            hemolysis_results,
            half_life_results,
            cleavage_results,
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
            candidate.predictions["cleavage_stability"] = cleavage_result

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
        return [
            ref
            for ref in super().models_used()
            if ref.name not in ("hemolysis_predictor", "cleavage_site_predictor")
        ] + [
            ModelRef(name="hemolysis_predictor", version="v2"),
            ModelRef(name="cleavage_site_predictor", version="v2"),
            ModelRef(name="half_life_predictor", version="v1"),
        ]

    def release_models(self) -> None:
        super().release_models()
        release_stage_models(globals(), stage_name=self.name)

    def compute_hemolysis_batch(
        self, sequences: list[str], model: HemolysisPredictor, feature_extractor
    ) -> list[dict]:
        """pHC50 (regression, for ranking) and P(hemolytic) (classifier, for filtering) from hemolysis_predictor_v2; non-standard sequences are not scored."""
        results: list[dict] = [
            {
                "phc50": None,
                "p_hemolytic": None,
                "status": "skipped_unsupported_sequence",
                "reason": "hemolysis model covers the 20 standard residues",
            }
        ] * len(sequences)
        scorable = [i for i, s in enumerate(sequences) if s and set(s) <= STANDARD_RESIDUES]
        if scorable:
            predictions = model.predict_batch([sequences[i] for i in scorable], feature_extractor)
            for i, p in zip(scorable, predictions):
                results[i] = {
                    "phc50": p["phc50"],
                    "p_hemolytic": p["p_hemolytic"],
                    "hc50_um": p["hc50_um"],
                    "model": "hemolysis_predictor_v2",
                    "status": "ok",
                }
        return results

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

    def _get_distance_matrix(self, candidate: Candidate):
        """Stage 7's Ca distance matrix when there is one; otherwise None, and the cleavage model folds the peptide itself."""
        structure = candidate.predictions.get("structure")
        if structure and "ca_distance_matrix" in structure:
            return np.asarray(structure["ca_distance_matrix"], dtype=np.float32)
        return None

    def compute_safety_verdict(self, predictions: dict, config_params: dict) -> dict:
        """Parent verdict with hemolysis made optional and plasma half-life added as a hard reject."""
        t = {**DEFAULT_THRESHOLDS, **config_params}
        hemolysis_cutoff = t["hemolysis_probability_reject_max"]

        # The parent reads a pHC50 cutoff unconditionally; hand it values that cannot trip, then set hemolysis below.
        verdict = super().compute_safety_verdict(
            {
                **predictions,
                "hemolysis": {"phc50": 0.0},
            },
            {**t, "hemolysis_phc50_reject_max": float("inf")},
        )
        properties = verdict["properties"]

        # Filtering uses the classifier's P(hemolytic); the regression pHC50 is not a gate.
        p_hemolytic = predictions["hemolysis"].get("p_hemolytic")
        if hemolysis_cutoff is None:
            properties["hemolysis"] = "disabled"
        elif p_hemolytic is None:
            properties["hemolysis"] = "not_screened"
        else:
            properties["hemolysis"] = "reject" if p_hemolytic > hemolysis_cutoff else "pass"

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
