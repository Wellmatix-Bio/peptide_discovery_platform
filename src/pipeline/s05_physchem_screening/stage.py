# Stage 5: Sequence and Physicochemical Screening.
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
from tqdm import tqdm

from common.gpu import release_stage_models
from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

logger = get_logger(__name__)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

# s4pred is vendored (not pip-installed), so its package dir must be on sys.path before import.
_S4PRED_DIR = Path(__file__).parent / "s4pred"
if str(_S4PRED_DIR) not in sys.path:
    sys.path.insert(0, str(_S4PRED_DIR))

from .s4pred.network import S4PRED  # noqa: E402
from .s4pred.utilities import aas2int  # noqa: E402

MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"
if str(MODEL_STORE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_STORE_DIR))

from model_store.solubility_predictor_v1 import SolubilityPredictor  # noqa: E402
from model_store.aggregation_predictor_v1 import AggregationPredictor  # noqa: E402

# Default solvent for the solubility model when no delivery-vehicle-specific solvent is
# configured: closest bundled match to physiological/wound-fluid conditions (aqueous, pH 7.4).
DEFAULT_SOLUBILITY_SOLVENT = "0.1 M PBS"

# AggregationPredictor requires len(sequence) >= 5 (QSO/SOCN/PAAC/APAAC lag requirement).
AGGREGATION_MIN_LENGTH = 5

_solubility_model: SolubilityPredictor | None = None
_aggregation_model: AggregationPredictor | None = None


def _get_solubility_model(use_feature_cache: bool = False) -> SolubilityPredictor:
    global _solubility_model
    # Rebuild if the cached instance's mode doesn't match what's asked for
    # now, not just when it's unset -- otherwise a stale instance built with
    # the opposite use_feature_cache silently ignores this call's flag and
    # crashes on a mismatched feature_extractor.
    if (
        _solubility_model is None
        or _solubility_model.use_feature_cache != use_feature_cache
    ):
        _solubility_model = SolubilityPredictor(use_feature_cache=use_feature_cache)
    return _solubility_model


def _get_aggregation_model(use_feature_cache: bool = False) -> AggregationPredictor:
    global _aggregation_model
    if (
        _aggregation_model is None
        or _aggregation_model.use_feature_cache != use_feature_cache
    ):
        _aggregation_model = AggregationPredictor(use_feature_cache=use_feature_cache)
    return _aggregation_model


_SS_CLASSES = {0: "C", 1: "H", 2: "E"}  # coil, helix, strand -- s4pred's output order

# Dominant fold each desired function mechanistically depends on; used to flag
# a fold prediction that contradicts the candidate's intended mechanism.
FUNCTION_EXPECTED_FOLD = {
    "antimicrobial action": {"H"},  # amphipathic helix drives membrane disruption
    "cell proliferation/migration": {"H", "C"},
    "angiogenesis": {"H", "C"},
    "immunomodulation": {"H", "C", "E"},
    "collagen synthesis": {"C", "E"},  # extended/coil, not a folded globular domain
}

# Default filter thresholds (screening table). Override per-run via
# StageConfig.params — every key here is read with config.params.get(key, default).
DEFAULT_THRESHOLDS = {
    "molecular_weight_reject_max": 5000.0,  # Da; >4-5 kDa flagged for delivery/synthesis
    "net_charge_flag_min": -5.0,
    "net_charge_flag_max": 9.0,
    "hydrophobic_fraction_reject_max": 0.6,  # fraction of residues on the hydrophobic side of Eisenberg 0
    "hydrophobic_moment_flag_max": 0.5,  # Eisenberg-scale moment; high = strong membrane disruption
    "instability_index_flag_max": 40.0,  # Guruprasad et al. 1990 cutoff
    "solubility_flag_min": 0.4,  # P(soluble) below this is flagged, not rejected -- Stage 5 is a soft screen
    "aggregation_tendency_flag_max": 0.6,  # P(aggregation-prone) above this is flagged
    "amphipathicity_flag_max": 0.8,
    "isoelectric_point_flag_ph_window": 0.5,  # |pI - pH| below this is flagged (near-neutral solubility risk)
    "ss_confidence_unstable_max": 0.5,  # below this mean top-class confidence, s4pred's call is "unstable"
    "ss_confidence_ambiguous_max": 0.7,  # below this (and above unstable_max), s4pred's call is "ambiguous"
    "risk_category_low_max": 0,  # motif count <= this -> "low" risk
    "risk_category_medium_max": 2,  # motif count <= this -> "medium" risk, else "high"
}

_s4pred_model: S4PRED | None = None


def _get_s4pred_model() -> S4PRED:
    """Lazily load and cache the s4pred network (5 ensembled weight files) on first use."""
    global _s4pred_model
    if _s4pred_model is None:
        model = S4PRED()
        weights_dir = _S4PRED_DIR / "weights"
        for i, sub_model in enumerate(
            [model.model_1, model.model_2, model.model_3, model.model_4, model.model_5],
            start=1,
        ):
            sub_model.load_state_dict(
                torch.load(weights_dir / f"weights_{i}.pt", map_location="cpu")
            )
        model.eval()
        model.requires_grad_(False)
        _s4pred_model = model
    return _s4pred_model


# Deamidation-prone Asn dipeptide motifs (fastest-reacting): Asn followed by Gly or Ser.
DEAMIDATION_MOTIFS = ("NG", "NS")

# Oxidation-prone residues: Met (thioether), Cys (thiol), Trp (indole ring).
OXIDATION_RESIDUES = ("M", "C", "W")

# Guruprasad et al. 1990 DIWV table used by ExPASy ProtParam; unlisted dipeptides default to 1.0.

with open(Path(__file__).with_name("diwv.json"), "r") as f:
    DIWV = json.load(f)


class Stage5(CandidateStage):
    name = "s05_physchem_screening"
    produces = {"sequence"}

    def release_models(self) -> None:
        release_stage_models(globals(), stage_name=self.name)

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        """Screens candidates on sequence/physicochemical properties, rejecting hard failures and flagging soft risks (see compute_screening_verdict)."""
        ph = config.params.get("ph", 8)
        solvent = config.params.get("solubility_solvent", "Ultrapure water")
        desired_functions = ctx.brief.desired_functions if ctx.brief else []

        use_feature_cache = ctx.use_feature_cache
        feature_extractor = ctx.feature_extractor if use_feature_cache else None
        if use_feature_cache:
            # Warm the shared ESM2 cache once for the whole stage instead of
            # one embedding per candidate inside compute_solubility below.
            sequences = [
                candidate.sequence for candidate in candidates if candidate.sequence
            ]
            ctx.feature_extractor.get_esm2_embedding_batch(sequences)

        survivors: list[Candidate] = []
        for candidate in tqdm(candidates, desc="Stage 5"):
            sequence = candidate.sequence
            candidate.predictions.update(
                {
                    "molecular_weight": self.compute_molecular_weight(sequence),
                    "length": self.compute_length(sequence),
                    "net_charge": self.compute_net_charge(sequence, ph=ph),
                    "isoelectric_point": self.compute_isoelectric_point(sequence),
                    "hydrophobicity": self.compute_hydrophobicity(sequence),
                    "hydrophobic_fraction": self.compute_hydrophobic_fraction(sequence),
                    "hydrophobic_moment": self.compute_hydrophobic_moment(sequence),
                    "amphipathicity": self.compute_amphipathicity(sequence),
                    "instability_index": self.compute_instability_index(sequence),
                    "deamidation_risk": self.compute_deamidation_risk(
                        sequence, config.params
                    ),
                    "oxidation_risk": self.compute_oxidation_risk(
                        sequence, config.params
                    ),
                    "disulfide_complexity": self.compute_disulfide_complexity(sequence),
                    "secondary_structure_consistency": self.compute_secondary_structure_consistency(
                        sequence, desired_functions, config.params
                    ),
                    "solubility": self.compute_solubility(
                        sequence,
                        feature_extractor,
                        use_feature_cache,
                        solvent=solvent,
                    ),
                    "aggregation_tendency": self.compute_aggregation_tendency(
                        sequence, feature_extractor, use_feature_cache
                    ),
                }
            )
            verdict = self.compute_screening_verdict(
                sequence, candidate.predictions, config.params
            )
            candidate.predictions["screening_verdict"] = verdict

            if verdict["overall"] == "reject":
                failed = {k: v for k, v in verdict["properties"].items() if v != "pass"}
                logger.info(
                    "s05.reject",
                    extra={"candidate_id": candidate.id, "failed": failed},
                )
                continue
            survivors.append(candidate)

        return survivors

    # ------------------------------------------------------------------
    # Individual attribute computations — one method each.
    # ------------------------------------------------------------------

    def compute_molecular_weight(self, sequence: str) -> float:
        """Monoisotopic-free average molecular weight in Daltons (modlAMP GlobalDescriptor)."""
        gd = GlobalDescriptor([sequence])
        gd.calculate_MW(amide=False)
        return float(gd.descriptor[0][0])

    def compute_length(self, sequence: str) -> int:
        """Residue count."""
        return len(sequence)

    def compute_net_charge(self, sequence: str, ph: float = 7.4) -> float:
        """Net charge at the given pH (default physiological 7.4)."""
        gd = GlobalDescriptor([sequence])
        gd.calculate_charge(ph=ph, amide=False)
        return float(gd.descriptor[0][0])

    def compute_isoelectric_point(self, sequence: str) -> float:
        """pH at which the peptide carries no net charge."""
        gd = GlobalDescriptor([sequence])
        gd.isoelectric_point()
        return float(gd.descriptor[0][0])

    def compute_hydrophobicity(self, sequence: str) -> float:
        """Mean hydrophobicity on the Eisenberg consensus scale."""
        pd_global = PeptideDescriptor([sequence], "eisenberg")
        pd_global.calculate_global()
        return float(pd_global.descriptor[0][0])

    def compute_hydrophobic_fraction(self, sequence: str) -> float:
        """Fraction of residues with a positive (hydrophobic-side) Eisenberg value."""
        pd_scale = PeptideDescriptor([sequence], "eisenberg")
        pd_scale.calculate_global()
        scale_values = pd_scale.scale
        hydrophobic = sum(1 for aa in sequence if scale_values.get(aa, [0])[0] > 0)
        return hydrophobic / len(sequence) if sequence else 0.0

    def compute_hydrophobic_moment(self, sequence: str) -> float:
        """Eisenberg-scale hydrophobic moment, assuming alpha-helical periodicity (100 deg/residue)."""
        pd_moment = PeptideDescriptor([sequence], "eisenberg")
        pd_moment.calculate_moment()
        return float(pd_moment.descriptor[0][0])

    def compute_amphipathicity(self, sequence: str) -> float:
        """Amphipathicity index: hydrophobic moment normalized to [0, 1] by the max moment theoretically achievable for this sequence."""
        pd_scale = PeptideDescriptor([sequence], "eisenberg")
        pd_scale.calculate_global()
        scale_values = pd_scale.scale  # {residue: [eisenberg value]}

        moment = self.compute_hydrophobic_moment(sequence)
        # Per-residue mean ceiling, matching calculate_moment()'s per-residue-mean output.
        residue_values = [
            abs(scale_values[aa][0]) for aa in sequence if aa in scale_values
        ]
        if not residue_values:
            return 0.0
        max_possible_moment = sum(residue_values) / len(sequence)
        if max_possible_moment == 0:
            return 0.0
        return max(0.0, min(1.0, moment / max_possible_moment))

    def compute_instability_index(self, sequence: str) -> float:
        """Guruprasad et al. 1990 instability index; II > 40 is conventionally unstable."""
        length = len(sequence)
        if length < 2:
            return 0.0
        total = sum(DIWV.get(sequence[i : i + 2], 1.0) for i in range(length - 1))
        return (10.0 / length) * total

    def compute_deamidation_risk(self, sequence: str, config_params: dict) -> dict:
        """Motif-based deamidation risk: counts NG/NS dipeptide occurrences (fastest-reacting Asn hotspots)."""
        positions = [
            i
            for i in range(len(sequence) - 1)
            if sequence[i : i + 2] in DEAMIDATION_MOTIFS
        ]
        return {
            "count": len(positions),
            "positions": positions,
            "motifs_found": [sequence[i : i + 2] for i in positions],
            "risk_category": self._risk_category(len(positions), config_params),
        }

    def compute_oxidation_risk(self, sequence: str, config_params: dict) -> dict:
        """Motif-based oxidation risk: counts Met/Cys/Trp residue occurrences."""
        positions = [i for i, aa in enumerate(sequence) if aa in OXIDATION_RESIDUES]
        return {
            "count": len(positions),
            "positions": positions,
            "motifs_found": [sequence[i] for i in positions],
            "risk_category": self._risk_category(len(positions), config_params),
        }

    def compute_disulfide_complexity(self, sequence: str) -> dict:
        """Rule-based disulfide complexity from Cys count; odd counts are flagged (not gated) since free thiols can be intentional."""
        positions = [i for i, aa in enumerate(sequence) if aa == "C"]
        count = len(positions)

        if count == 0:
            category = "none"
        elif count % 2 == 1:
            category = "flag"
        elif count == 2:
            category = "low"
        elif count == 4:
            category = "medium"
        else:
            category = "high"

        return {
            "category": category,
            "count": count,
            "positions": positions,
        }

    def compute_secondary_structure_consistency(
        self, sequence: str, desired_functions: list[str], config_params: dict
    ) -> dict:
        """s4pred secondary-structure screen: flags low-confidence/ambiguous folds and folds inconsistent with the candidate's intended mechanism."""
        t = {
            **DEFAULT_THRESHOLDS,
            **{k: v for k, v in config_params.items() if k in DEFAULT_THRESHOLDS},
        }
        ss, confidence = self._predict_secondary_structure(sequence)
        mean_confidence = float(confidence.mean()) if len(confidence) else 0.0

        if mean_confidence < t["ss_confidence_unstable_max"]:
            stability = "unstable"
        elif mean_confidence < t["ss_confidence_ambiguous_max"]:
            stability = "ambiguous"
        else:
            stability = "stable"

        expected = {
            cls
            for fn in desired_functions
            for cls in FUNCTION_EXPECTED_FOLD.get(fn, set())
        }
        predicted_classes = set(ss)
        mechanism_consistent = not expected or bool(expected & predicted_classes)

        flag = stability != "stable" or not mechanism_consistent

        return {
            "predicted_ss": ss,
            "mean_confidence": mean_confidence,
            "stability": stability,
            "expected_fold_classes": sorted(expected),
            "mechanism_consistent": mechanism_consistent,
            "flag": flag,
        }

    def _predict_secondary_structure(self, sequence: str) -> tuple[str, "torch.Tensor"]:
        """Runs vendored s4pred; returns the per-residue C/H/E string and per-residue top-class confidence."""
        model = _get_s4pred_model()
        encoded = torch.tensor([aas2int(sequence)])
        with torch.no_grad():
            log_probs = model(
                encoded
            )  # GRUnet.forward squeezes the batch dim -> [L, 3]
            probs = log_probs.exp()
            probs = probs / probs.sum(-1, keepdim=True)
            top_class = probs.argmax(-1)
            confidence = probs.gather(-1, top_class.unsqueeze(-1)).squeeze(-1)
        ss = "".join(_SS_CLASSES[i] for i in top_class.tolist())
        return ss, confidence

    def compute_solubility(
        self,
        sequence: str,
        feature_extractor,
        use_feature_cache: bool = False,
        solvent: str = DEFAULT_SOLUBILITY_SOLVENT,
    ) -> dict:
        """P(soluble) from solubility_predictor_v1 (XGBoost over ESM2 + solvent descriptors).
        `solvent` defaults to a physiological/wound-fluid-like aqueous buffer since the brief's
        delivery_system is free text, not one of the model's 7 trained-on lab solvents.
        """
        model = _get_solubility_model(use_feature_cache)
        score = model.predict_proba(sequence, solvent, feature_extractor)
        return {"score": score, "solvent": solvent, "status": "ok"}

    def compute_aggregation_tendency(
        self, sequence: str, feature_extractor, use_feature_cache: bool = False
    ) -> dict:
        """Aggregation-propensity probability from aggregation_predictor_v1 (XGBoost over
        AAindex1/biopython/propy descriptors). Below AGGREGATION_MIN_LENGTH the model's
        feature extraction (QSO/SOCN/PAAC/APAAC lag) is undefined, so it's skipped."""
        if len(sequence) < AGGREGATION_MIN_LENGTH:
            return {
                "score": None,
                "status": "skipped_too_short",
            }
        model = _get_aggregation_model(use_feature_cache)
        score = model.predict_aggregation(sequence, feature_extractor)
        return {"score": score, "status": "ok"}

    def compute_screening_verdict(
        self, sequence: str, predictions: dict, config_params: dict
    ) -> dict:
        """Applies the Stage 5 filter table's thresholds to this candidate's predictions.

        Returns a per-property reject/flag/pass verdict plus an overall status:
        "reject" if any property rejects, else "flag" if any property flags, else "pass".
        Thresholds default to DEFAULT_THRESHOLDS and are overridable via config params.
        """
        t = {
            **DEFAULT_THRESHOLDS,
            **{k: v for k, v in config_params.items() if k in DEFAULT_THRESHOLDS},
        }
        properties: dict[str, str] = {}

        properties["molecular_weight"] = (
            "reject"
            if predictions["molecular_weight"] > t["molecular_weight_reject_max"]
            else "pass"
        )

        max_length = config_params.get("max_length")
        properties["length"] = (
            "flag" if max_length and predictions["length"] > max_length else "pass"
        )

        charge = predictions["net_charge"]
        properties["net_charge"] = (
            "flag"
            if charge < t["net_charge_flag_min"] or charge > t["net_charge_flag_max"]
            else "pass"
        )

        ph = config_params.get("ph", 7.4)
        pi = predictions["isoelectric_point"]
        properties["isoelectric_point"] = (
            "flag" if abs(pi - ph) < t["isoelectric_point_flag_ph_window"] else "pass"
        )

        properties["hydrophobicity"] = (
            "reject"
            if predictions["hydrophobic_fraction"]
            > t["hydrophobic_fraction_reject_max"]
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

        solubility_score = predictions["solubility"]["score"]
        properties["solubility"] = (
            "flag"
            if solubility_score is not None
            and solubility_score < t["solubility_flag_min"]
            else "pass"
        )

        aggregation_score = predictions["aggregation_tendency"]["score"]
        properties["aggregation_tendency"] = (
            "flag"
            if aggregation_score is not None
            and aggregation_score > t["aggregation_tendency_flag_max"]
            else "pass"
        )

        properties["secondary_structure"] = (
            "flag" if predictions["secondary_structure_consistency"]["flag"] else "pass"
        )

        properties["disulfide_complexity"] = (
            "flag"
            if predictions["disulfide_complexity"]["category"] in ("flag", "high")
            else "pass"
        )

        properties["instability_index"] = (
            "flag"
            if predictions["instability_index"] > t["instability_index_flag_max"]
            else "pass"
        )

        properties["oxidation_risk"] = (
            "flag"
            if predictions["oxidation_risk"]["risk_category"] == "high"
            else "pass"
        )

        properties["deamidation_risk"] = (
            "flag"
            if predictions["deamidation_risk"]["risk_category"] == "high"
            else "pass"
        )

        if "reject" in properties.values():
            overall = "reject"
        elif "flag" in properties.values():
            overall = "flag"
        else:
            overall = "pass"

        return {"properties": properties, "overall": overall}

    @staticmethod
    def _risk_category(count: int, config_params: dict) -> str:
        t = {
            **DEFAULT_THRESHOLDS,
            **{k: v for k, v in config_params.items() if k in DEFAULT_THRESHOLDS},
        }
        if count <= t["risk_category_low_max"]:
            return "low"
        if count <= t["risk_category_medium_max"]:
            return "medium"
        return "high"
