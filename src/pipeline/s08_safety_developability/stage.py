# Stage 8: Safety and Developability - hemolysis, cytotoxicity, aggregation, cleavage stability.
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.gpu import release_stage_models
from common.logging import get_logger
from common.model_registry import ModelRef
from pipeline.base import CandidateStage, RunContext
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

logger = get_logger(__name__)

MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"
if str(MODEL_STORE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_STORE_DIR))

from model_store.hemolysis_predictor_v1 import ModifiedHemolyticPredictor  # noqa: E402
from model_store.cytotoxicity_predictor_v1 import CytotoxicityClassifier  # noqa: E402
from model_store.solubility_predictor_v1 import SolubilityPredictor  # noqa: E402
from model_store.aggregation_predictor_v1 import AggregationPredictor  # noqa: E402
from model_store.cleavage_site_predictor_v1 import CleavageSitePredictor  # noqa: E402

# Hard reject thresholds; any one failing rejects the candidate (high pHC50 = hemolytic at low concentration = worse).
HEMOLYSIS_PHC50_REJECT_MAX = 4.0
CYTOTOXICITY_REJECT_MAX = 0.5  # P(cytotoxic)
AGGREGATION_REJECT_MAX = 0.7  # P(aggregation-prone)
SOLUBILITY_FLAG_MIN = 0.3  # P(soluble) below this is flagged, not rejected

# Cleavage-stability score (see compute_cleavage_stability) below this rejects.
CLEAVAGE_STABILITY_REJECT_MIN = 0.3

# Wound-relevant protease selection (step 4): a hand-reviewed accession allowlist against the bundled 118-enzyme catalog
WOUND_PROTEASE_ACCESSIONS = {
    # MMP-9 / gelatinase B (EC 3.4.24.35)
    "O18733",
    "P41245",
    "P41246",
    "P50282",
    "P52176",
    # MMP-2 / gelatinase A (EC 3.4.24.24)
    "P33434",
    "P50757",
    "Q90611",
    # MMP-1 / interstitial collagenase (EC 3.4.24.7)
    "P13943",
    "P28053",
    "Q11133",
    "Q9XSZ5",
    # MMP-3 / stromelysin-1 (EC 3.4.24.17)
    "P03957",
    "P28862",
    "P28863",
    "Q28397",
    "Q6Y4Q5",
    # MMP-8 / neutrophil collagenase (EC 3.4.24.34)
    "O70138",
    "O88766",
    # MMP-12 / macrophage elastase (EC 3.4.24.65)
    "P34960",
    "P79227",
    "Q63341",
    # Neutrophil elastase (EC 3.4.21.37), not the pancreatic Elastase-1/-2 entries.
    "Q3UP87",
    # Cathepsin G, neutrophil serine protease released with NETs (EC 3.4.21.20)
    "P17977",
    "P28293",
    # Plasmin itself (EC 3.4.21.7), not the plasminogen activators.
    "O18783",
    "P06867",
    "P12545",
    "P20918",
    "Q01177",
    "Q29485",
    "Q5R8X6",
    # S. epidermidis extracellular elastase (M04); no Pseudomonas protease is in the panel.
    "P0C0Q4",
}

# Cleavage stability score = exp(-CLEAVAGE_DECAY_RATE * weighted_site_count).
CLEAVAGE_DECAY_RATE = 0.5

# Fallback used only when Stage 7 is disabled and its structure predictions are absent.
IDEALIZED_CA_STEP_ANGSTROM = 3.8  # extended-chain Ca-Ca spacing along backbone


def _get_hemolysis_v1_model() -> ModifiedHemolyticPredictor:
    global _hemolysis_v1_model
    if _hemolysis_v1_model is None:
        _hemolysis_v1_model = ModifiedHemolyticPredictor()
    return _hemolysis_v1_model


def _get_cytotoxicity_model() -> CytotoxicityClassifier:
    global _cytotoxicity_model
    if _cytotoxicity_model is None:
        _cytotoxicity_model = CytotoxicityClassifier()
    return _cytotoxicity_model


def _get_solubility_model() -> SolubilityPredictor:
    global _solubility_model
    if _solubility_model is None:
        _solubility_model = SolubilityPredictor()
    return _solubility_model


def _get_aggregation_model() -> AggregationPredictor:
    global _aggregation_model
    if _aggregation_model is None:
        _aggregation_model = AggregationPredictor()
    return _aggregation_model


def _get_cleavage_model() -> CleavageSitePredictor:
    global _cleavage_model
    if _cleavage_model is None:
        _cleavage_model = CleavageSitePredictor()
    return _cleavage_model


_hemolysis_v1_model: ModifiedHemolyticPredictor | None = None
_cytotoxicity_model: CytotoxicityClassifier | None = None
_solubility_model: SolubilityPredictor | None = None
_aggregation_model: AggregationPredictor | None = None
_cleavage_model: CleavageSitePredictor | None = None


@dataclass(frozen=True)
class Stage8Models:
    """Bundle returned by build_models(): every model this stage depends on, lazy-loaded on first use of each."""

    hemolysis: ModifiedHemolyticPredictor
    cytotoxicity: CytotoxicityClassifier
    solubility: SolubilityPredictor
    aggregation: AggregationPredictor
    cleavage: CleavageSitePredictor


def build_models(config_params: dict) -> Stage8Models:
    """Factory: import and initialize every Stage 8 model (each lazy-loads its weights on first predict())."""
    return Stage8Models(
        hemolysis=_get_hemolysis_v1_model(),
        cytotoxicity=_get_cytotoxicity_model(),
        solubility=_get_solubility_model(),
        aggregation=_get_aggregation_model(),
        cleavage=_get_cleavage_model(),
    )


class Stage8(CandidateStage):
    name = "s08_safety_developability"
    produces = {
        "hemolysis",
        "cytotoxicity",
        "solubility",
        "aggregation_tendency",
        "cleavage_stability",
    }

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        """Scores hemolysis, cytotoxicity, solubility, aggregation and cleavage stability, then rejects candidates failing any hard threshold."""
        models = build_models(config.params)
        solvent = config.params.get("solubility_solvent", "Ultrapure water")
        # DRAMP_aggregate -> maps to mammalian cell
        cell_type = config.params.get("cytotoxicity_cell_type", "DRAMP_aggregate")

        # Batch hemolysis so the model loads once for the stage.
        sequences: list[str] = [
            candidate.sequence for candidate in candidates if candidate.sequence
        ]
        if len(sequences) != len(candidates):
            raise ValueError("Stage 8 received a candidate with no sequence")

        feature_extractor = ctx.feature_extractor
        # Warm the shared ESM2 cache once for the whole stage.
        ctx.feature_extractor.get_esm2_embedding_batch(sequences)

        hemolysis_results = self.compute_hemolysis_batch(
            sequences, models.hemolysis, feature_extractor
        )

        survivors: list[Candidate] = []
        for candidate, hemolysis_result in zip(
            tqdm(candidates, desc="Stage 8"), hemolysis_results
        ):
            sequence = candidate.sequence
            candidate.predictions.update(
                {
                    "hemolysis": hemolysis_result,
                    "cytotoxicity": self.compute_cytotoxicity(
                        sequence, models.cytotoxicity, cell_type, feature_extractor
                    ),
                    "solubility": self.compute_solubility(
                        sequence, models.solubility, solvent, feature_extractor
                    ),
                    "aggregation_tendency": self.compute_aggregation(
                        sequence, models.aggregation, feature_extractor
                    ),
                }
            )
            candidate.predictions["cleavage_stability"] = (
                self.compute_cleavage_stability(
                    candidate, models.cleavage, config.params
                )
            )

            verdict = self.compute_safety_verdict(candidate.predictions, config.params)
            candidate.predictions["safety_verdict"] = verdict

            if verdict["overall"] == "reject":
                logger.info(
                    "s08.reject",
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
            ModelRef(
                name="hemolysis_predictor",
                version="v1",
            ),
            ModelRef(name="cytotoxicity_predictor", version="v1"),
            ModelRef(name="solubility_predictor", version="v1"),
            ModelRef(name="aggregation_predictor", version="v1"),
            ModelRef(name="cleavage_site_predictor", version="v1"),
        ]

    def release_models(self) -> None:
        release_stage_models(globals(), stage_name=self.name)

    # Individual model computations.

    def compute_hemolysis_batch(
        self,
        sequences: list[str],
        model: ModifiedHemolyticPredictor,
        feature_extractor=None,
    ) -> list[dict]:
        """Predict pHC50 for all candidates in one v1 batch (higher = more hemolytic)."""
        if not sequences:
            return []
        phc50_values = model.predict_phc50_batch(sequences, feature_extractor)
        return [{"phc50": phc50, "status": "ok"} for phc50 in phc50_values]

    def compute_cytotoxicity(
        self,
        sequence: str,
        model: CytotoxicityClassifier,
        cell_type: str,
        feature_extractor,
    ) -> dict:
        """P(mammalian-cell cytotoxic) from cytotoxicity_predictor_v1."""
        score = model.predict_cytotoxicity(
            sequence, feature_extractor, cell_type=cell_type
        )
        return {"score": score, "cell_type": cell_type, "status": "ok"}

    def compute_solubility(
        self,
        sequence: str,
        model: SolubilityPredictor,
        solvent: str,
        feature_extractor,
    ) -> dict:
        """P(soluble) from solubility_predictor_v1, same model/solvent convention as Stage 5."""
        score = model.predict_proba(sequence, solvent, feature_extractor)
        return {"score": score, "solvent": solvent, "status": "ok"}

    def compute_aggregation(
        self, sequence: str, model: AggregationPredictor, feature_extractor
    ) -> dict:
        """P(aggregation-prone) from aggregation_predictor_v1."""
        if len(sequence) < 5:
            return {"score": None, "status": "skipped_too_short"}
        score = model.predict_aggregation(sequence, feature_extractor)
        return {"score": score, "status": "ok"}

    # Cleavage-site stability (steps 3-6).

    def compute_cleavage_stability(
        self, candidate: Candidate, model: CleavageSitePredictor, config_params: dict
    ) -> dict:
        """Protease-cleavage stability score in (0, 1] (1.0 = no predicted sites); None if no allowlisted enzyme is in the panel."""
        sequence = candidate.sequence
        distance_matrix = self._get_distance_matrix(candidate)
        exposure_by_position = self._get_exposure_by_position(candidate, len(sequence))

        wound_accessions = self._wound_relevant_accessions(model, config_params)
        if not wound_accessions:
            return {
                "score": None,
                "status": "no_wound_relevant_enzymes_in_panel",
                "sites": [],
            }

        weighted_sites = []
        for accession in wound_accessions:
            result = model.predict_one(sequence, distance_matrix, accession)
            for position in result["cleavage_site_positions"]:
                probability = result["per_residue_probability"][position]
                exposure = exposure_by_position[position]
                weighted_sites.append(
                    {
                        "position": position,
                        "enzyme_accession": accession,
                        "enzyme_name": result["enzyme_name"],
                        "probability": probability,
                        "exposure": exposure,
                        "weighted_severity": probability * exposure,
                    }
                )

        decay_rate = config_params.get("cleavage_decay_rate", CLEAVAGE_DECAY_RATE)
        total_severity = sum(site["weighted_severity"] for site in weighted_sites)
        score = float(np.exp(-decay_rate * total_severity))

        return {
            "score": score,
            "status": "ok",
            "n_wound_enzymes_screened": len(wound_accessions),
            "n_sites": len(weighted_sites),
            "sites": weighted_sites,
        }

    def _wound_relevant_accessions(
        self, model: CleavageSitePredictor, config_params: dict
    ) -> list[str]:
        """Restricts the enzyme panel to WOUND_PROTEASE_ACCESSIONS; overridable via config.params["wound_protease_accessions"]."""
        override = config_params.get("wound_protease_accessions")
        allowlist = set(override) if override else WOUND_PROTEASE_ACCESSIONS

        available = {row["enzyme_uniprot"] for row in model.available_enzymes()}
        return sorted(allowlist & available)

    def _get_distance_matrix(self, candidate: Candidate) -> np.ndarray:
        """Ca-Ca distance matrix from Stage 7, falling back to an idealized straight chain when it did not run."""
        structure = candidate.predictions.get("structure")
        if structure and "ca_distance_matrix" in structure:
            return np.asarray(structure["ca_distance_matrix"], dtype=np.float32)

        n = len(candidate.sequence)
        positions = np.arange(n) * IDEALIZED_CA_STEP_ANGSTROM
        return np.abs(positions[:, None] - positions[None, :]).astype(np.float32)

    def _get_exposure_by_position(
        self, candidate: Candidate, length: int
    ) -> list[float]:
        """Per-residue exposure weight in [0, 1] from Stage 7, defaulting to 1.0 when it did not run."""
        structure = candidate.predictions.get("structure")
        if structure and "exposure_by_position" in structure:
            exposure = structure["exposure_by_position"]
            if len(exposure) != length:
                raise ValueError(
                    f"structure.exposure_by_position length {len(exposure)} != sequence length {length}"
                )
            return list(exposure)

        return [1.0] * length

    # Verdict.

    def compute_safety_verdict(self, predictions: dict, config_params: dict) -> dict:
        """Applies the hard-reject / soft-flag thresholds: "reject" if any rejects, else "flag" if any flags, else "pass"."""
        t = {
            "hemolysis_phc50_reject_max": config_params.get(
                "hemolysis_phc50_reject_max", HEMOLYSIS_PHC50_REJECT_MAX
            ),
            "cytotoxicity_reject_max": config_params.get(
                "cytotoxicity_reject_max", CYTOTOXICITY_REJECT_MAX
            ),
            "aggregation_reject_max": config_params.get(
                "aggregation_reject_max", AGGREGATION_REJECT_MAX
            ),
            "solubility_flag_min": config_params.get(
                "solubility_flag_min", SOLUBILITY_FLAG_MIN
            ),
            "cleavage_stability_reject_min": config_params.get(
                "cleavage_stability_reject_min", CLEAVAGE_STABILITY_REJECT_MIN
            ),
        }
        properties: dict[str, str] = {}

        # Higher pHC50 means hemolysis at a lower concentration (worse).
        properties["hemolysis"] = (
            "reject"
            if predictions["hemolysis"]["phc50"] > t["hemolysis_phc50_reject_max"]
            else "pass"
        )

        properties["cytotoxicity"] = (
            "reject"
            if predictions["cytotoxicity"]["score"] > t["cytotoxicity_reject_max"]
            else "pass"
        )

        # A screen that did not run is not a pass; "not_screened" rolls up as a flag.
        aggregation = predictions["aggregation_tendency"]
        aggregation_score = aggregation["score"]
        if aggregation_score is None:
            properties["aggregation_tendency"] = "not_screened"
        elif aggregation_score > t["aggregation_reject_max"]:
            properties["aggregation_tendency"] = "reject"
        else:
            properties["aggregation_tendency"] = "pass"

        # Same rule as aggregation_tendency: a null score means the model did not run.
        solubility_score = predictions["solubility"]["score"]
        if solubility_score is None:
            properties["solubility"] = "not_screened"
        elif solubility_score < t["solubility_flag_min"]:
            properties["solubility"] = "flag"
        else:
            properties["solubility"] = "pass"

        cleavage_score = predictions["cleavage_stability"]["score"]
        if cleavage_score is None:
            properties["cleavage_stability"] = "not_screened"
        elif cleavage_score < t["cleavage_stability_reject_min"]:
            properties["cleavage_stability"] = "reject"
        else:
            properties["cleavage_stability"] = "pass"

        if "reject" in properties.values():
            overall = "reject"
        elif "flag" in properties.values() or "not_screened" in properties.values():
            # An unrun screen makes the verdict a flag, never a pass or a reject.
            overall = "flag"
        else:
            overall = "pass"

        unscreened = sorted(k for k, v in properties.items() if v == "not_screened")
        verdict = {"properties": properties, "overall": overall}
        if unscreened:
            verdict["not_screened"] = unscreened
            verdict["not_screened_note"] = (
                "These checks did not run, so this candidate was not cleared by them. Not checked"
                " is not the same as passed."
            )
        return verdict
