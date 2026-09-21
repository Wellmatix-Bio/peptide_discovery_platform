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

from model_store.hemolysis_predictor_v2 import HemoPI2PHC50Predictor  # noqa: E402
from model_store.hemolysis_predictor_v1 import ReplicatedHemoPI2Predictor  # noqa: E402
from model_store.cytotoxicity_predictor_v1 import CytotoxicityClassifier  # noqa: E402
from model_store.solubility_predictor_v1 import SolubilityPredictor  # noqa: E402
from model_store.aggregation_predictor_v1 import AggregationPredictor  # noqa: E402
from model_store.cleavage_site_predictor_v1 import CleavageSitePredictor  # noqa: E402

# Reject thresholds. Same "hard threshold overrides score" convention as the
# rest of the pipeline (CLAUDE.md) -- any one of these failing rejects the
# candidate at Stage 8 regardless of its other scores.
# pHC50 = -log10(HC50 in M); HIGH pHC50 = hemolytic at LOW concentration = worse
# (see routeA.py's _hemolysis_safety_factor, which has always used this direction
# correctly: "-> 1 as pHC50 falls (safer), -> 0 as pHC50 rises (more hemolytic)").
# 4.0 = HemoPI2's own HC50 < 100 uM "Hemolytic" cutoff, converted: 6 - log10(100) = 4.0.
HEMOLYSIS_PHC50_REJECT_MAX = 4.0
CYTOTOXICITY_REJECT_MAX = 0.5  # P(cytotoxic)
AGGREGATION_REJECT_MAX = 0.7  # P(aggregation-prone)
SOLUBILITY_FLAG_MIN = 0.3  # P(soluble) below this is flagged, not rejected

# Cleavage-stability score (see compute_cleavage_stability) below this rejects.
CLEAVAGE_STABILITY_REJECT_MIN = 0.3

# Wound-relevant protease selection (step 4): a hand-reviewed accession
# allowlist against the bundled 118-enzyme catalog
# (model_store/cleavage_site_predictor_v1/enzymes/enzyme_catalog.csv), not a
# keyword filter -- protein names in that catalog are inconsistent enough
# ("Elastase-1 precursor" covers both neutrophil-adjacent and purely
# digestive/pancreatic orthologs; "u-/t-Plasminogen activator" entries
# activate plasmin but aren't the wound-fluid protease itself) that name
# matching alone pulls in irrelevant enzymes.
#
# The panel is missing real human MMP-1/-2/-3/-8/-9/-12 and human neutrophil
# elastase entirely (README.md's "Known coverage gap") -- their structures
# failed the source project's alignment check upstream of this migration.
# Non-human orthologs of the same functional families stand in as
# directional evidence, grouped by EC number below.
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
    # Neutrophil elastase, true neutrophil serine protease (EC 3.4.21.37) --
    # NOT the "Elastase-1/-2 precursor" entries, which are pancreatic.
    "Q3UP87",
    # Cathepsin G, neutrophil serine protease released with NETs (EC 3.4.21.20)
    "P17977",
    "P28293",
    # Plasmin/plasminogen itself, wound-fluid fibrinolysis (EC 3.4.21.7) --
    # NOT the uPA/tPA/salivary plasminogen-activator entries (EC
    # 3.4.21.68/73), which activate plasmin but aren't the protease itself.
    "O18783",
    "P06867",
    "P12545",
    "P20918",
    "Q01177",
    "Q29485",
    "Q5R8X6",
    # Bacterial secreted protease: S. epidermidis extracellular elastase
    # (M04 family). No Pseudomonas-secreted protease is in this panel.
    "P0C0Q4",
}

# Cleavage stability score = exp(-CLEAVAGE_DECAY_RATE * weighted_site_count).
# Chosen so a single fully-exposed, high-confidence cut site (weight ~1.0)
# costs a moderate stability penalty, and sites compound geometrically.
CLEAVAGE_DECAY_RATE = 0.5

# Interim substitute for Stage 7 structural exposure/distance data, which
# does not exist yet. TODO(stage-7): once Stage 7 populates
# candidate.predictions["structure"] with real Ca coordinates and per-residue
# exposure, delete this stub and read that instead.
IDEALIZED_CA_STEP_ANGSTROM = 3.8  # extended-chain Ca-Ca spacing along backbone


def _get_hemolysis_v2_model() -> HemoPI2PHC50Predictor:
    global _hemolysis_v2_model
    if _hemolysis_v2_model is None:
        _hemolysis_v2_model = HemoPI2PHC50Predictor()
    return _hemolysis_v2_model


def _get_hemolysis_v1_model() -> ReplicatedHemoPI2Predictor:
    global _hemolysis_v1_model
    if _hemolysis_v1_model is None:
        _hemolysis_v1_model = ReplicatedHemoPI2Predictor()
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


_hemolysis_v2_model: HemoPI2PHC50Predictor | None = None
_hemolysis_v1_model: ReplicatedHemoPI2Predictor | None = None
_cytotoxicity_model: CytotoxicityClassifier | None = None
_solubility_model: SolubilityPredictor | None = None
_aggregation_model: AggregationPredictor | None = None
_cleavage_model: CleavageSitePredictor | None = None


@dataclass(frozen=True)
class Stage8Models:
    """Bundle returned by build_models(): every model this stage depends on, lazy-loaded on first use of each."""

    hemolysis: HemoPI2PHC50Predictor | ReplicatedHemoPI2Predictor
    hemolysis_version: str
    cytotoxicity: CytotoxicityClassifier
    solubility: SolubilityPredictor
    aggregation: AggregationPredictor
    cleavage: CleavageSitePredictor


def build_models(config_params: dict) -> Stage8Models:
    """Factory: import and initialize every Stage 8 model.

    Each predictor lazy-loads its own weights on first predict() call, so
    construction here is cheap; this just fixes the one place that knows how
    to build each of them.
    """
    hemolysis_version = config_params.get("hemolysis_predictor_version", "v1")
    if hemolysis_version not in ("v1", "v2"):
        raise ValueError(
            f"unknown hemolysis_predictor_version {hemolysis_version!r}; expected 'v1' or 'v2'"
        )
    hemolysis_model = (
        _get_hemolysis_v2_model()
        if hemolysis_version == "v2"
        else _get_hemolysis_v1_model()
    )
    return Stage8Models(
        hemolysis=hemolysis_model,
        hemolysis_version=hemolysis_version,
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
        """Scores hemolysis, cytotoxicity, solubility, aggregation, and
        cleavage stability, then rejects candidates that fail any hard
        safety/developability threshold (CLAUDE.md: hard thresholds override
        score, applied throughout -- not as a late add-on)."""
        models = build_models(config.params)
        self._hemolysis_version_used = models.hemolysis_version
        solvent = config.params.get("solubility_solvent", "Ultrapure water")
        # DRAMP_aggregate -> maps to mammalian cell
        cell_type = config.params.get("cytotoxicity_cell_type", "DRAMP_aggregate")

        # Hemolysis is computed up front for every candidate in one call --
        # v2 (HemoPI2) is a single batched subprocess instead of one process
        # per sequence; v1 (in-process sklearn reproduction) batches
        # internally too (one predict_phc50_batch call), so both versions pay
        # one model-load cost for the whole stage rather than per candidate.
        sequences: list[str] = [
            candidate.sequence for candidate in candidates if candidate.sequence
        ]
        if len(sequences) != len(candidates):
            raise ValueError("Stage 8 received a candidate with no sequence")
        hemolysis_results = self.compute_hemolysis_batch(sequences, models.hemolysis)

        survivors: list[Candidate] = []
        for candidate, hemolysis_result in zip(
            tqdm(candidates, desc="Stage 8"), hemolysis_results
        ):
            sequence = candidate.sequence
            candidate.predictions.update(
                {
                    "hemolysis": hemolysis_result,
                    "cytotoxicity": self.compute_cytotoxicity(
                        sequence, models.cytotoxicity, cell_type
                    ),
                    "solubility": self.compute_solubility(
                        sequence, models.solubility, solvent
                    ),
                    "aggregation_tendency": self.compute_aggregation(
                        sequence, models.aggregation
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
                version=getattr(self, "_hemolysis_version_used", "v1"),
            ),
            ModelRef(name="cytotoxicity_predictor", version="v1"),
            ModelRef(name="solubility_predictor", version="v1"),
            ModelRef(name="aggregation_predictor", version="v1"),
            ModelRef(name="cleavage_site_predictor", version="v1"),
        ]

    def release_models(self) -> None:
        release_stage_models(globals(), stage_name=self.name)

    # ------------------------------------------------------------------
    # Individual model computations.
    # ------------------------------------------------------------------

    def compute_hemolysis_batch(
        self,
        sequences: list[str],
        model: HemoPI2PHC50Predictor | ReplicatedHemoPI2Predictor,
    ) -> list[dict]:
        """pHC50 for every candidate in one call (v1's in-process batched
        predict, or v2/HemoPI2's single batched subprocess) instead of one
        call per sequence. HIGH pHC50 = hemolytic at LOW concentration =
        worse, same convention for both predictor versions (see
        hemolysis_predictor_v1/v2's READMEs for their HC50(uM) -> pHC50
        conversions)."""
        if not sequences:
            return []
        phc50_values = model.predict_phc50_batch(sequences)
        return [{"phc50": phc50, "status": "ok"} for phc50 in phc50_values]

    def compute_cytotoxicity(
        self, sequence: str, model: CytotoxicityClassifier, cell_type: str
    ) -> dict:
        """P(mammalian-cell cytotoxic) from cytotoxicity_predictor_v1."""
        score = model.predict_cytotoxicity(sequence, cell_type=cell_type)
        return {"score": score, "cell_type": cell_type, "status": "ok"}

    def compute_solubility(
        self, sequence: str, model: SolubilityPredictor, solvent: str
    ) -> dict:
        """P(soluble) from solubility_predictor_v1, same model/solvent convention as Stage 5."""
        score = model.predict_proba(sequence, solvent)
        return {"score": score, "solvent": solvent, "status": "ok"}

    def compute_aggregation(self, sequence: str, model: AggregationPredictor) -> dict:
        """P(aggregation-prone) from aggregation_predictor_v1."""
        if len(sequence) < 5:
            return {"score": None, "status": "skipped_too_short"}
        score = model.predict_aggregation(sequence)
        return {"score": score, "status": "ok"}

    # ------------------------------------------------------------------
    # Cleavage-site stability (steps 3-6).
    # ------------------------------------------------------------------

    def compute_cleavage_stability(
        self, candidate: Candidate, model: CleavageSitePredictor, config_params: dict
    ) -> dict:
        """Per-protease cleavage-site prediction (predict_one()/predict() return
        per-enzyme results, not a single pooled risk number -- see
        cleavage_site_predictor_v1/README.md's usage example), filtered to
        wound-relevant proteases, weighted by structural exposure, and
        collapsed into one stability score used as a reject/flag gate.
        """
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
        """Restricts the bundled enzyme panel to WOUND_PROTEASE_ACCESSIONS
        (MMPs active in wound exudate, neutrophil elastase/cathepsin G,
        plasmin, and the one bundled bacterial secreted protease), intersected
        with what the panel actually contains. Overridable via
        config.params["wound_protease_accessions"] for callers that want a
        different explicit list."""
        override = config_params.get("wound_protease_accessions")
        allowlist = set(override) if override else WOUND_PROTEASE_ACCESSIONS

        available = {row["enzyme_uniprot"] for row in model.available_enzymes()}
        return sorted(allowlist & available)

    def _get_distance_matrix(self, candidate: Candidate) -> np.ndarray:
        """Ca-Ca distance matrix for the candidate, from Stage 7's structure
        prediction. TODO(stage-7): Stage 7 is not yet implemented, so this
        falls back to an idealized extended-chain placeholder (Ca spacing
        IDEALIZED_CA_STEP_ANGSTROM along a straight line) -- structurally
        wrong (real peptides fold), but keeps this stage runnable end-to-end
        until Stage 7 lands. Replace with candidate.predictions["structure"]'s
        real coordinates once available.
        """
        structure = candidate.predictions.get("structure")
        if structure and "ca_distance_matrix" in structure:
            return np.asarray(structure["ca_distance_matrix"], dtype=np.float32)

        n = len(candidate.sequence)
        positions = np.arange(n) * IDEALIZED_CA_STEP_ANGSTROM
        return np.abs(positions[:, None] - positions[None, :]).astype(np.float32)

    def _get_exposure_by_position(
        self, candidate: Candidate, length: int
    ) -> list[float]:
        """Per-residue exposure/flexibility weight in [0, 1] (1.0 = fully
        exposed, sites here count fully; 0.0 = fully buried, sites here are
        ignored), from Stage 7. TODO(stage-7): Stage 7 is not yet
        implemented, so this defaults every position to neutral weight 1.0
        (no up/down-weighting) until real exposure data is available.
        """
        structure = candidate.predictions.get("structure")
        if structure and "exposure_by_position" in structure:
            exposure = structure["exposure_by_position"]
            if len(exposure) != length:
                raise ValueError(
                    f"structure.exposure_by_position length {len(exposure)} != sequence length {length}"
                )
            return list(exposure)

        return [1.0] * length

    # ------------------------------------------------------------------
    # Verdict.
    # ------------------------------------------------------------------

    def compute_safety_verdict(self, predictions: dict, config_params: dict) -> dict:
        """Applies Stage 8's hard-reject / soft-flag thresholds. Overall is
        "reject" if any property rejects, else "flag" if any flags, else "pass" --
        same convention as Stage 5's compute_screening_verdict."""
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

        # HIGH pHC50 = hemolytic at LOW concentration = worse (see
        # routeA.py's _hemolysis_safety_factor: "-> 1 as pHC50 falls (safer),
        # -> 0 as pHC50 rises (more hemolytic)"). The check here was
        # previously inverted (rejected on phc50 < min, i.e. rejected the
        # *safe* candidates) and briefly disabled entirely while that bug
        # was tracked down; both v1 and v2 predict pHC50 in this same
        # direction and are trained/validated against HemoPI2's own
        # benchmark, so this gate is trustworthy for either version.
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

        aggregation_score = predictions["aggregation_tendency"]["score"]
        properties["aggregation_tendency"] = (
            "reject"
            if aggregation_score is not None
            and aggregation_score > t["aggregation_reject_max"]
            else "pass"
        )

        solubility_score = predictions["solubility"]["score"]
        properties["solubility"] = (
            "flag"
            if solubility_score is not None
            and solubility_score < t["solubility_flag_min"]
            else "pass"
        )

        cleavage_score = predictions["cleavage_stability"]["score"]
        properties["cleavage_stability"] = (
            "reject"
            if cleavage_score is not None
            and cleavage_score < t["cleavage_stability_reject_min"]
            else "pass"
        )

        if "reject" in properties.values():
            overall = "reject"
        elif "flag" in properties.values():
            overall = "flag"
        else:
            overall = "pass"

        return {"properties": properties, "overall": overall}
