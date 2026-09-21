# Stage 6: Functional AI Models - antimicrobial, migration, angiogenesis, immunomodulation, collagen.
from __future__ import annotations
import sys
import math
from pathlib import Path

from common.gpu import release_stage_models
from common.logging import get_logger
from common.model_registry import ModelRef
from pipeline.base import CandidateStage, RunContext
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from tqdm import tqdm

logger = get_logger(__name__)

MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"
if str(MODEL_STORE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_STORE_DIR))

from model_store.amp_classifier_v1 import AMPClassifier  # noqa: E402
from model_store.mic_predictor_v1 import MICPredictorEnsemble  # noqa: E402
from model_store.mic_predictor_v1.predictor import (
    SUPPORTED_ORGANISMS as MIC_SUPPORTED_ORGANISMS,
)  # noqa: E402
from model_store.mbic_predictor_v1 import MBICPredictor  # noqa: E402
from model_store.proliferation_migration_predictor_v1 import (
    ProliferationMigrationPredictor,
)  # noqa: E402
from model_store.angiogenic_activity_predictor_v1 import (
    AngiogenicActivityPredictor,
)  # noqa: E402
from model_store.anti_inflammatory_predictor_v1 import (
    AntiInflammatoryPredictor,
)  # noqa: E402

# MBIC's SVR was trained on this species vocabulary (model_card.json); an
# out-of-vocabulary species silently becomes an all-zero one-hot, so only
# score species this model actually knows about.
MBIC_SUPPORTED_SPECIES = {
    "Acinetobacter baumannii",
    "Candida albicans",
    "Candida tropicalis",
    "Cutibacterium acnes",
    "Enterococcus faecium",
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Salmonella enterica",
    "Staphylococcus aureus",
    "Staphylococcus epidermidis",
    "Streptococcus mutans",
    "Streptococcus sanguinis",
}

_amp_model: AMPClassifier | None = None
_mic_model: MICPredictorEnsemble | None = None
_mbic_model: MBICPredictor | None = None
_proliferation_migration_model: ProliferationMigrationPredictor | None = None
_angiogenic_model: AngiogenicActivityPredictor | None = None
_anti_inflammatory_model: AntiInflammatoryPredictor | None = None


def _get_amp_model() -> AMPClassifier:
    global _amp_model
    if _amp_model is None:
        _amp_model = AMPClassifier()
    return _amp_model


def _get_mic_model() -> MICPredictorEnsemble:
    global _mic_model
    if _mic_model is None:
        _mic_model = MICPredictorEnsemble()
    return _mic_model


def _get_mbic_model() -> MBICPredictor:
    global _mbic_model
    if _mbic_model is None:
        _mbic_model = MBICPredictor()
    return _mbic_model


def _get_proliferation_migration_model() -> ProliferationMigrationPredictor:
    global _proliferation_migration_model
    if _proliferation_migration_model is None:
        _proliferation_migration_model = ProliferationMigrationPredictor()
    return _proliferation_migration_model


def _get_angiogenic_model() -> AngiogenicActivityPredictor:
    global _angiogenic_model
    if _angiogenic_model is None:
        _angiogenic_model = AngiogenicActivityPredictor()
    return _angiogenic_model


def _get_anti_inflammatory_model() -> AntiInflammatoryPredictor:
    global _anti_inflammatory_model
    if _anti_inflammatory_model is None:
        _anti_inflammatory_model = AntiInflammatoryPredictor()
    return _anti_inflammatory_model


# Software defaults; override through config.params["stage6_thresholds"].
STAGE6_THRESHOLDS = {
    "min_amp_probability": 0.70,
    "min_proliferation_migration": 0.65,
    "min_angiogenic_activity": 0.60,
    "min_anti_inflammatory_probability": 0.65,
    "max_mic": {"MRSA": 16.0, "Pseudomonas_aeruginosa": 16.0},
    "max_mbic": {"MRSA": 32.0, "Pseudomonas_aeruginosa": 32.0},
}

# Threshold keys whose value is itself a dict (per-pathogen) — overriding one
# of these must merge per-pathogen, not replace the whole dict, or an
# override of e.g. just MRSA would silently drop every other pathogen's
# default threshold.
STAGE6_NESTED_THRESHOLD_KEYS = {"max_mic", "max_mbic"}


def merge_stage6_thresholds(overrides: dict) -> dict:
    """STAGE6_THRESHOLDS with `overrides` applied. Per-pathogen keys
    (max_mic/max_mbic) are merged one pathogen at a time; every other key is
    a plain top-level replace."""
    merged = dict(STAGE6_THRESHOLDS)
    for key, value in overrides.items():
        if key in STAGE6_NESTED_THRESHOLD_KEYS and isinstance(value, dict):
            merged[key] = {**merged.get(key, {}), **value}
        else:
            merged[key] = value
    return merged


def filter_predictions(
    predictions: dict,
    desired_functions: list[str],
    pathogens: list[str],
    thresholds: dict,
) -> dict:
    """Check only requested functions against the existing model outputs."""
    required = set(desired_functions)
    failed, missing, evaluated = [], [], {}

    def check(key, value, threshold, reason, *, lower=False, metadata=None):
        metadata = metadata or {}
        if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
            raise ValueError(f"Invalid Stage 6 threshold for {key}")
        usable = (
            value is not None
            and math.isfinite(value)
            and threshold is not None
            and metadata.get("applicability_domain") is not False
            and metadata.get("status", "ok") == "ok"
        )
        passed = (
            (value <= threshold if lower else value >= threshold) if usable else None
        )
        evaluated[key] = {
            "value": value,
            "threshold": threshold,
            "direction": "lower_is_better" if lower else "higher_is_better",
            "passed": passed,
        }
        if not usable:
            missing.append(key)
        elif not passed:
            failed.append(reason)

    for functions, key, field, reason in (
        ({"antimicrobial"}, "amp_probability", None, "low_amp_probability"),
        (
            {
                "keratinocyte_migration",
                "fibroblast_migration",
                "wound_closure",
                "proliferation",
            },
            "proliferation_migration",
            "migration",
            "low_proliferation_migration",
        ),
        (
            {"angiogenesis"},
            "angiogenic_activity",
            "angiogenic",
            "low_angiogenic_activity",
        ),
        (
            {"anti_inflammatory"},
            "anti_inflammatory_probability",
            None,
            "low_anti_inflammatory_activity",
        ),
    ):
        if required & functions:
            output = predictions.get(key)
            metadata = output if isinstance(output, dict) else {}
            value = output.get(field or "value") if isinstance(output, dict) else output
            check(key, value, thresholds.get(f"min_{key}"), reason, metadata=metadata)

    # MIC/MBIC checks disabled: TODO re-enable once mic_predictor_v1 covers
    # the brief's actual pathogens (MRSA gap)
    # and mbic thresholds cover every brief pathogen.
    per_pathogen_checks: list[tuple[str, str, str]] = [
        ("antimicrobial", "mic", "log_mic_um"),
        ("antibiofilm", "mbic", "pmbic"),
    ]
    for function, key, field in per_pathogen_checks:
        if function not in required:
            continue
        output = predictions.get(key) or {}
        values = output.get(field) or {}
        if not pathogens:
            missing.append(f"{key}:required_pathogens")
        for pathogen in dict.fromkeys(pathogens):
            # Normalize spelling only; never substitute S. aureus for MRSA.
            raw = values.get(pathogen, values.get(pathogen.replace("_", " ")))
            value = None
            if raw is not None:
                try:
                    value = 10.0 ** (raw if key == "mic" else 6.0 - raw)
                except OverflowError:
                    pass
            check(
                f"{key}:{pathogen}",
                value,
                thresholds.get(f"max_{key}", {}).get(pathogen),
                f"high_{key}:{pathogen}",
                lower=True,
                metadata=output,
            )

    status = "rejected" if failed else "insufficient_evidence" if missing else "passed"
    return {
        "status": status,
        "passed": status == "passed",
        "required_functions": list(desired_functions),
        "failed_requirements": failed,
        "missing_required_predictions": missing,
        "evaluated_predictions": evaluated,
    }


class Stage6(CandidateStage):
    name = "s06_functional_models"
    produces = {"sequence"}

    def release_models(self) -> None:
        release_stage_models(globals(), stage_name=self.name)

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        """Compute the existing predictions, then filter required Stage 1 functions."""
        if ctx.brief is None:
            raise ValueError("Stage 6 filtering requires the Stage 1 product brief")
        pathogens = ctx.brief.pathogens
        thresholds = merge_stage6_thresholds(config.params.get("stage6_thresholds", {}))
        survivors = []

        for candidate in tqdm(candidates, desc="Stage 6"):
            sequence = candidate.sequence
            candidate.predictions.update(
                {
                    "amp_probability": self.compute_amp_probability(sequence),
                    "mic": self.compute_mic(sequence, pathogens),
                    "mbic": self.compute_mbic(sequence, pathogens),
                    "proliferation_migration": self.compute_proliferation_migration(
                        sequence
                    ),
                    "angiogenic_activity": self.compute_angiogenic_activity(sequence),
                    "anti_inflammatory_probability": self.compute_anti_inflammatory_probability(
                        sequence
                    ),
                }
            )

            result = filter_predictions(
                candidate.predictions,
                ctx.brief.desired_functions,
                pathogens,
                thresholds,
            )
            candidate.predictions["stage6_filter"] = {
                "candidate_id": candidate.id,
                **result,
            }
            if result["passed"]:
                survivors.append(candidate)
            else:
                logger.info("s06.filter", extra=candidate.predictions["stage6_filter"])

        return survivors

    def models_used(self) -> list[ModelRef]:
        return [
            ModelRef(name="amp_classifier", version="v1"),
            ModelRef(name="mic_predictor", version="v1"),
            ModelRef(name="mbic_predictor", version="v1"),
            ModelRef(name="proliferation_migration_predictor", version="v1"),
            ModelRef(name="angiogenic_activity_predictor", version="v1"),
            ModelRef(name="anti_inflammatory_predictor", version="v1"),
        ]

    # ------------------------------------------------------------------
    # Individual model computations — one method each.
    # ------------------------------------------------------------------

    def compute_amp_probability(self, sequence: str) -> float:
        """P(antimicrobial peptide) from amp_classifier_v1 (5-fold XGBoost +
        meta-model ensemble over ESM2 + modlAMP descriptors). Supports the
        antimicrobial-action target function."""
        model = _get_amp_model()
        return model.predict_proba(sequence)

    def compute_mic(self, sequence: str, pathogens: list[str]) -> dict:
        """log10(MIC, uM) per pathogen from mic_predictor_v1 (BiLSTM+CNN+RF
        ensemble), restricted to the model's 3 ATCC reference organisms.
        `pathogens` outside that vocabulary are skipped, not guessed at; if
        none of the brief's pathogens match, scores all 3 supported organisms
        so the prediction isn't silently empty."""
        model = _get_mic_model()
        organisms = [
            p for p in pathogens if p in MIC_SUPPORTED_ORGANISMS
        ] or MIC_SUPPORTED_ORGANISMS
        per_organism = {
            organism: model.predict_log_mic(sequence, organism)
            for organism in organisms
        }
        return {"log_mic_um": per_organism, "status": "ok"}

    def compute_mbic(self, sequence: str, pathogens: list[str]) -> dict:
        """pMBIC (biofilm-inhibition potency) per pathogen from mbic_predictor_v1
        (SVR over ESM2 PCA + modlAMP physchem + species one-hot). Same
        vocabulary-restriction/fallback behavior as compute_mic."""
        model = _get_mbic_model()
        species_list = [p for p in pathogens if p in MBIC_SUPPORTED_SPECIES] or sorted(
            MBIC_SUPPORTED_SPECIES
        )
        per_species = {
            species: model.predict_pmbic(sequence, species) for species in species_list
        }
        return {"pmbic": per_species, "status": "ok"}

    def compute_proliferation_migration(self, sequence: str) -> dict:
        """Migration-dominant / proliferation-dominant probabilities from
        proliferation_migration_predictor_v1 (two VotingClassifier ensembles
        over hand-built physchem descriptors). Supports the cell
        proliferation/migration target function."""
        model = _get_proliferation_migration_model()
        return model.predict(sequence)

    def compute_angiogenic_activity(self, sequence: str) -> dict:
        """Angiogenic-dominant probability from angiogenic_activity_predictor_v1
        (SVM+RF+MLP ensemble over physchem + ESM2-PCA features). Supports the
        angiogenesis target function."""
        model = _get_angiogenic_model()
        return model.predict(sequence)

    def compute_anti_inflammatory_probability(self, sequence: str) -> float:
        """P(anti-inflammatory peptide) from anti_inflammatory_predictor_v1
        (DAC-AIPs variational-autoencoder + contrastive-learning classifier).
        Supports the immunomodulation target function."""
        model = _get_anti_inflammatory_model()
        return model.predict_proba(sequence)
