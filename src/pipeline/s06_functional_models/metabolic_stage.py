# Stage 6 (metabolism / weight control): GLP-1R, GIPR and GCGR agonism screening.
from __future__ import annotations

import math
import sys
from pathlib import Path

from tqdm import tqdm

from common.gpu import release_stage_models
from common.logging import get_logger
from common.model_registry import ModelRef
from pipeline.base import CandidateStage, RunContext
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage6MetabolicParams

logger = get_logger(__name__)

MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"
if str(MODEL_STORE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_STORE_DIR))

from model_store.binding_affinity_predictor_v1 import BindingAffinityPredictor  # noqa: E402
from model_store.functional_potency_predictor_v1 import FunctionalPotencyPredictor  # noqa: E402

RECEPTORS = ("GLP1R", "GIPR", "GCGR")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
SCORE_EPSILON = 1e-6

_potency_model: FunctionalPotencyPredictor | None = None
_binding_model: BindingAffinityPredictor | None = None
_agonism_classifier = None
_agonism_classifier_loaded = False


def _get_potency_model() -> FunctionalPotencyPredictor:
    global _potency_model
    if _potency_model is None:
        _potency_model = FunctionalPotencyPredictor()
    return _potency_model


def _get_binding_model() -> BindingAffinityPredictor:
    global _binding_model
    if _binding_model is None:
        _binding_model = BindingAffinityPredictor()
    return _binding_model


def _get_agonism_classifier():
    """The 3-receptor agonism classifier if model_store ships one, else None (the stub); API: `predict_proba_batch(seqs, feature_extractor) -> {receptor: [p or None, ...]}`."""
    global _agonism_classifier, _agonism_classifier_loaded
    if not _agonism_classifier_loaded:
        _agonism_classifier_loaded = True
        try:
            from model_store.agonism_classifier_v1 import AgonismClassifier  # type: ignore

            _agonism_classifier = AgonismClassifier()
        except ImportError:
            _agonism_classifier = None
    return _agonism_classifier


def pec50_to_ec50_nm(pec50: float) -> float:
    """pEC50 = -log10(EC50 [M]) -> EC50 in nM."""
    return 10.0 ** (9.0 - pec50)


def potency_score(ec50_nm: float, ec50_max_nm: float, ec50_floor_nm: float) -> float:
    """Linear in pEC50 between the gate (score 0) and the floor (score 1), clipped; higher is more potent."""
    lo, hi = 9.0 - math.log10(ec50_max_nm), 9.0 - math.log10(ec50_floor_nm)
    if hi <= lo:
        raise ValueError("ec50_floor_nm must be below ec50_max_nm")
    return float(min(1.0, max(0.0, (9.0 - math.log10(ec50_nm) - lo) / (hi - lo))))


def geometric_mean(scores: list[float]) -> float:
    """F_activity: (prod S_i) ** (1 / n) over the selected receptors, each floored at SCORE_EPSILON."""
    return math.exp(sum(math.log(max(s, SCORE_EPSILON)) for s in scores) / len(scores))


def evaluate_candidate(
    predictions: dict, receptors: list[str], params: dict
) -> dict:
    """Receptor gates for one candidate: `rejected` on any failed gate, `insufficient_evidence` on an unusable required prediction, else `passed`."""
    failed, missing, evaluated = [], [], {}
    for r in receptors:
        ec50 = predictions["ec50_nm"].get(r)
        threshold = params["ec50_max_nm"].get(r)
        usable = ec50 is not None and math.isfinite(ec50) and threshold is not None
        passed = (ec50 <= threshold) if usable else None
        evaluated[f"ec50:{r}"] = {
            "value": ec50,
            "threshold": threshold,
            "direction": "lower_is_better",
            "passed": passed,
            "clamped_to_floor": predictions["ec50_clamped"].get(r, False),
        }
        if not usable:
            missing.append(f"ec50:{r}")
        elif not passed:
            failed.append(f"high_ec50:{r}")

        prob = (predictions.get("agonism_probability") or {}).get(r)
        minimum = params["agonism_probability_min"].get(r)
        usable = prob is not None and math.isfinite(prob) and minimum is not None
        passed = (prob >= minimum) if usable else None
        evaluated[f"agonism_probability:{r}"] = {
            "value": prob,
            "threshold": minimum,
            "direction": "higher_is_better",
            "passed": passed,
        }
        if not usable:
            missing.append(f"agonism_probability:{r}")
        elif not passed:
            failed.append(f"low_agonism_probability:{r}")

    status = "rejected" if failed else "insufficient_evidence" if missing else "passed"
    return {
        "status": status,
        "passed": not failed,
        "required_receptors": list(receptors),
        "failed_requirements": failed,
        "missing_required_predictions": missing,
        "evaluated_predictions": evaluated,
    }


class Stage6Metabolic(CandidateStage):
    """Replaces the wound-healing functional gates (AMP, MIC/MBIC, migration, angiogenesis) with receptor agonism screening.

    Opt-in: it runs only when named in the run config, normally in place of `s06_functional_models`.
    """

    name = "s06_metabolic_functional_screening"
    produces = {"sequence"}

    def release_models(self) -> None:
        release_stage_models(globals(), stage_name=self.name)

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        params = Stage6MetabolicParams.model_validate(config.params).model_dump()
        brief_receptors = getattr(ctx.brief, "target_receptors", None) if ctx.brief else None
        receptors = [r for r in RECEPTORS if r in (brief_receptors or params["target_receptors"])]
        if not receptors:
            raise ValueError("Stage 6 (metabolic) needs at least one target receptor")
        floor = params["ec50_floor_nm"]

        sequences = [c.sequence for c in candidates if c.sequence]
        if len(sequences) != len(candidates):
            raise ValueError("Stage 6 received a candidate with no sequence")
        if not candidates:
            return []
        fe = ctx.feature_extractor
        # The potency and binding models score only the 20 standard uppercase residues; anything else is
        # unscorable, which is missing evidence for that candidate rather than a failure of the stage.
        scorable = [i for i, seq in enumerate(sequences) if set(seq) <= STANDARD_AA]
        scorable_sequences = [sequences[i] for i in scorable]
        n = len(sequences)
        pec50 = {r: [None] * n for r in RECEPTORS}
        pkd = {r: [None] * n for r in RECEPTORS}
        if scorable_sequences:
            fe.get_esm2_embedding_batch(scorable_sequences)  # one batched forward pass shared by both models
            potency_out = _get_potency_model().predict_pec50_all_batch(scorable_sequences, fe)
            binding_out = _get_binding_model().predict_pkd_all_batch(scorable_sequences, fe)
            for r in RECEPTORS:
                for j, i in enumerate(scorable):
                    pec50[r][i] = float(potency_out[r][j])
                    pkd[r][i] = float(binding_out[r][j])
        classifier = _get_agonism_classifier()
        probs = classifier.predict_proba_batch(sequences, fe) if classifier else None

        survivors = []
        for i, candidate in enumerate(tqdm(candidates, desc="Stage 6 (metabolic)")):
            ec50, clamped, probability = {}, {}, {}
            for r in RECEPTORS:
                if pec50[r][i] is None:
                    ec50[r], clamped[r] = None, False
                else:
                    raw = pec50_to_ec50_nm(pec50[r][i])
                    ec50[r], clamped[r] = max(raw, floor), raw < floor
                value = probs[r][i] if probs and r in probs else None
                probability[r] = float(value) if value is not None else None
            candidate.predictions.update(
                {
                    "pec50": {r: pec50[r][i] for r in RECEPTORS},
                    "ec50_nm": ec50,
                    "ec50_clamped": clamped,
                    "agonism_probability": probability,
                    "binding_pkd": {r: pkd[r][i] for r in RECEPTORS},
                }
            )
            result = evaluate_candidate(candidate.predictions, receptors, params)
            usable = all(ec50[r] is not None for r in receptors)
            candidate.predictions["metabolic_activity_score"] = (
                geometric_mean(
                    [potency_score(ec50[r], params["ec50_max_nm"][r], floor) for r in receptors]
                )
                if usable
                else None
            )
            candidate.predictions["stage6_filter"] = {"candidate_id": candidate.id, **result}
            if result["passed"]:
                survivors.append(candidate)
            else:
                logger.info("s06m.filter", extra=candidate.predictions["stage6_filter"])
        return survivors

    def models_used(self) -> list[ModelRef]:
        used = [
            ModelRef(name="functional_potency_predictor", version="v1"),
            ModelRef(name="binding_affinity_predictor", version="v1"),
        ]
        if _get_agonism_classifier() is not None:
            used.append(ModelRef(name="agonism_classifier", version="v1"))
        return used
