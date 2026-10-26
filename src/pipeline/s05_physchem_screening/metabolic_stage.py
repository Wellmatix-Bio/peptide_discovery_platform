# Stage 5 (metabolism / weight control): sequence and physicochemical screening with thresholds for incretin-like peptides.
from __future__ import annotations

from pipeline.feature_extractor import PROPY_AVAILABLE, PROPY_UNAVAILABLE_REASON
from pipeline.s05_physchem_screening.stage import (
    AGGREGATION_MIN_LENGTH,
    Stage5,
    _get_aggregation_model,
)
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage5MetabolicParams

#: Single source of the thresholds, so the run-config schema and the stage cannot drift apart.
DEFAULT_THRESHOLDS = Stage5MetabolicParams().model_dump()


class Stage5Metabolic(Stage5):
    """Same property computations as Stage 5, with the metabolism thresholds.

    Hard filters (candidate removed): molecular weight, mean Eisenberg hydrophobicity, and length outside the
    brief's range. Everything else flags and keeps the candidate. Opt-in: it runs only when named in the
    run config, normally in place of `s05_physchem_screening`.
    """

    name = "s05_metabolic_physchem_screening"

    _min_length: int = 0

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx,
    ) -> list[Candidate]:
        if ctx.brief is None:
            raise ValueError("Stage 5 screening requires the Stage 1 product brief")
        self._min_length = ctx.brief.min_length
        # Parent defaults pH to 8 and its solvent differently; pin both to this stage's defaults.
        params = {
            "ph": DEFAULT_THRESHOLDS["ph"],
            "solubility_solvent": DEFAULT_THRESHOLDS["solubility_solvent"],
            **config.params,
        }
        return super().run(
            candidates, config.model_copy(update={"params": params}), ctx
        )

    def compute_aggregation_tendency(self, sequence: str, feature_extractor) -> dict:
        """Aggregation-propensity probability from aggregation_predictor_v1; reports not-run states instead of a score."""
        if len(sequence) < AGGREGATION_MIN_LENGTH:
            return {
                "score": None,
                "status": "skipped_too_short",
                "reason": (
                    f"sequence is shorter than {AGGREGATION_MIN_LENGTH} residues, below which the"
                    " model's QSO/SOCN/PAAC/APAAC lag features are undefined"
                ),
            }
        if not PROPY_AVAILABLE:
            return {
                "score": None,
                "status": "unavailable",
                "reason": PROPY_UNAVAILABLE_REASON,
            }
        try:
            score = _get_aggregation_model().predict_aggregation(
                sequence, feature_extractor
            )
        except ZeroDivisionError as exc:
            # propy's autocorrelation features divide by the residue-property variance, which is zero for homopolymers.
            return {
                "score": None,
                "status": "error",
                "reason": f"aggregation features undefined for this sequence ({exc})",
            }
        return {"score": score, "status": "ok"}

    def compute_screening_verdict(
        self, sequence: str, predictions: dict, config_params: dict, max_length: int
    ) -> dict:
        """Per-property verdict (pass / flag / reject / not_screened) and an overall reject / flag / pass."""
        t = {
            **DEFAULT_THRESHOLDS,
            **{k: v for k, v in config_params.items() if k in DEFAULT_THRESHOLDS},
        }
        properties: dict[str, str] = {}

        length = predictions["length"]
        properties["length"] = (
            "reject" if length < self._min_length or length > max_length else "pass"
        )

        weight = predictions["molecular_weight"]
        if weight > t["molecular_weight_reject_max"]:
            properties["molecular_weight"] = "reject"
        elif weight > t["molecular_weight_flag_max"]:
            properties["molecular_weight"] = "flag"
        else:
            properties["molecular_weight"] = "pass"

        properties["hydrophobicity"] = (
            "reject"
            if predictions["hydrophobicity"] > t["hydrophobicity_reject_max"]
            else "pass"
        )

        charge = predictions["net_charge"]
        properties["net_charge"] = (
            "flag"
            if charge < t["net_charge_flag_min"] or charge > t["net_charge_flag_max"]
            else "pass"
        )

        properties["isoelectric_point"] = (
            "flag"
            if abs(predictions["isoelectric_point"] - t["ph"])
            < t["isoelectric_point_flag_ph_window"]
            else "pass"
        )

        properties["hydrophobic_moment"] = (
            "flag"
            if predictions["hydrophobic_moment"] > t["hydrophobic_moment_flag_max"]
            else "pass"
        )
        properties["amphipathicity"] = (
            "flag"
            if predictions["amphipathicity"] > t["amphipathicity_flag_max"]
            else "pass"
        )
        properties["instability_index"] = (
            "flag"
            if predictions["instability_index"] > t["instability_index_flag_max"]
            else "pass"
        )

        # A None score means the model did not run; it is recorded as not screened, which rolls up as a flag.
        solubility_score = predictions["solubility"]["score"]
        if solubility_score is None:
            properties["solubility"] = "not_screened"
        else:
            properties["solubility"] = (
                "flag" if solubility_score < t["solubility_flag_min"] else "pass"
            )

        aggregation_score = predictions["aggregation_tendency"]["score"]
        if aggregation_score is None:
            properties["aggregation_tendency"] = "not_screened"
        else:
            properties["aggregation_tendency"] = (
                "flag" if aggregation_score > t["aggregation_tendency_flag_max"] else "pass"
            )

        # Flagged when s4pred's mean confidence is below the ambiguous bound (which includes unstable).
        secondary_structure = predictions["secondary_structure_consistency"]
        if not secondary_structure.get("available", True):
            properties["secondary_structure"] = "not_screened"
        else:
            properties["secondary_structure"] = (
                "flag" if secondary_structure["stability"] != "stable" else "pass"
            )

        properties["disulfide_complexity"] = (
            "flag"
            if predictions["disulfide_complexity"]["category"] in ("flag", "high")
            else "pass"
        )
        properties["oxidation_risk"] = (
            "flag"
            if predictions["oxidation_risk"]["count"] > t["oxidation_count_flag_max"]
            else "pass"
        )
        properties["deamidation_risk"] = (
            "flag"
            if predictions["deamidation_risk"]["count"] > t["deamidation_count_flag_max"]
            else "pass"
        )

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
