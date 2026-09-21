# Stage 9: Synthesis / CMC Feasibility.
from __future__ import annotations
import sys
from pathlib import Path

from tqdm import tqdm

from common.logging import get_logger
from common.model_registry import ModelRef
from pipeline.base import CandidateStage, RunContext
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

logger = get_logger(__name__)

MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"
if str(MODEL_STORE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_STORE_DIR))

from model_store.synthesis_feasibility_predictor_v1 import SynthesisFeasibilityEnsemble  # noqa: E402

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

# Per-coupling-step SPPS yield; purity compounds multiplicatively over (length - 1) couplings.
BASE_STEP_YIELD = 0.99

ARGININE_YIELD_PENALTY = 0.03  # Pbf-protected Arg: slow, sterically hindered coupling
BETA_BRANCHED_RUN_YIELD_PENALTY = 0.04  # per residue, once >=3 in a row
DG_NG_MOTIF_YIELD_PENALTY = 0.05  # Asp-Gly/Asn-Gly: succinimide rearrangement risk
OXIDATION_RESIDUE_YIELD_PENALTY = 0.01  # Met/Trp: real cost is cleavage-step oxidation

BETA_BRANCHED_RESIDUES = set("VIT")  # valine, isoleucine, threonine
BETA_BRANCHED_RUN_MIN_LENGTH = 3
OXIDATION_PRONE_RESIDUES = set("MW")
DG_NG_MOTIFS = ("DG", "NG")

DISULFIDE_STEPS_PER_PAIR = 2  # oxidation/folding step + verification step
CYCLIZATION_STEPS = 3
NON_STANDARD_RESIDUE_STEPS_EACH = 1

# >=3 disulfide pairs requires directed/orthogonal-protection folding, a specialist job.
DISULFIDE_PAIRS_REQUIRING_DIRECTED_FOLDING = 3

# Non-standard building blocks, from Candidate.predictions["modifications"] (not inferable from sequence alone).
NON_STANDARD_BUILDING_BLOCKS = {
    "d_amino_acid": {"cost_multiplier": 1.5, "routine_vendor_availability": False},
    "lipid_tail": {"cost_multiplier": 2.0, "routine_vendor_availability": False},
    "peg_chain": {"cost_multiplier": 1.6, "routine_vendor_availability": False},
    "cyclic": {"cost_multiplier": 1.8, "routine_vendor_availability": False},
}

DIFFICULTY_MEDIUM_MIN_SCORE = 2
DIFFICULTY_HIGH_MIN_SCORE = 5

# Hard gate: difficulty classes rejected outright regardless of other scores
# (CLAUDE.md: hard thresholds override score). Everything else here --
# yield, purification, cost, storage, scale-up -- is a soft penalty carried
# through to Stage 11, never a rejection reason on its own.
REJECTED_DIFFICULTY_CLASSES = {"high"}

FINAL_PURITY_BASE_RECOVERY = 0.95  # fraction of crude purity retained through purification, routine case
FINAL_PURITY_DISULFIDE_RECOVERY_PENALTY = 0.05  # per pair beyond the first
FINAL_PURITY_DG_NG_RECOVERY_PENALTY = 0.08  # rearrangement isomer co-elutes
FINAL_PURITY_NON_STANDARD_RECOVERY_PENALTY = 0.05  # per non-standard building block


def build_model() -> SynthesisFeasibilityEnsemble:
    """Factory: import and initialize the synthesis-feasibility ML ensemble (supplementary weak prior, not a gate)."""
    return SynthesisFeasibilityEnsemble()


class Stage9(CandidateStage):
    name = "s09_synthesis_cmc"
    produces = {"synthesis_feasibility"}

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        """Rule-based synthesis/CMC feasibility assessment per candidate,
        then rejects candidates that are effectively unsynthesizable (hard
        gate on difficulty_class). Yield, purification, cost, storage, and
        scale-up risk are preserved as soft-penalty scores for Stage 11,
        never rejection reasons on their own."""
        model = build_model()

        survivors: list[Candidate] = []
        for candidate in tqdm(candidates, desc="Stage 9"):
            feasibility = self.assess_synthesis_feasibility(candidate, model, config.params)
            candidate.predictions["synthesis_feasibility"] = feasibility

            verdict = self.compute_synthesis_verdict(feasibility, config.params)
            candidate.predictions["synthesis_verdict"] = verdict

            if verdict["overall"] == "reject":
                logger.info(
                    "s09.reject",
                    extra={"candidate_id": candidate.id, "properties": verdict["properties"]},
                )
                continue
            survivors.append(candidate)

        return survivors

    def compute_synthesis_verdict(self, feasibility: dict, config_params: dict) -> dict:
        """Hard gate on difficulty_class only. Everything else in
        `feasibility` (cost bands, purity ceiling, penalty drivers) is a soft
        signal Stage 11 weighs, not a reject condition here."""
        override = config_params.get("rejected_difficulty_classes")
        rejected_classes = set(override) if override is not None else REJECTED_DIFFICULTY_CLASSES

        difficulty_class = feasibility["difficulty_class"]
        properties = {
            "difficulty_class": "reject" if difficulty_class in rejected_classes else "pass",
        }
        overall = "reject" if "reject" in properties.values() else "pass"
        return {"properties": properties, "overall": overall}

    def models_used(self) -> list[ModelRef]:
        return [ModelRef(name="synthesis_feasibility_predictor", version="v1")]

    # -- Top-level assessment --

    def assess_synthesis_feasibility(
        self, candidate: Candidate, model: SynthesisFeasibilityEnsemble, config_params: dict
    ) -> dict:
        sequence = candidate.sequence
        modifications = candidate.predictions.get("modifications", {})

        penalties = self.identify_penalty_drivers(sequence)
        disulfide_pairs = self.count_disulfide_pairs(sequence, modifications)
        non_standard = self.identify_non_standard_building_blocks(modifications)

        crude_purity = self.compute_crude_purity(sequence, penalties)
        n_steps = self.compute_n_synthesis_steps(sequence, disulfide_pairs, modifications, non_standard)
        difficulty_class = self.compute_difficulty_class(penalties, disulfide_pairs, non_standard)
        cost_bands = self.compute_cost_bands(n_steps, difficulty_class, disulfide_pairs, non_standard)
        final_purity_ceiling = self.compute_final_purity_ceiling(
            crude_purity, penalties, disulfide_pairs, non_standard
        )

        ml_prior = self.compute_ml_feasibility_prior(sequence, model)

        return {
            "predicted_crude_purity": crude_purity,
            "n_synthesis_steps": n_steps,
            "difficulty_class": difficulty_class,
            "cost_band_100mg": cost_bands["per_100mg"],
            "cost_band_10g": cost_bands["per_10g"],
            "achievable_final_purity_ceiling": final_purity_ceiling,
            "penalty_drivers": penalties,
            "disulfide_pairs": disulfide_pairs,
            "non_standard_building_blocks": non_standard,
            "ml_feasibility_prior": ml_prior,
        }

    # -- Penalty identification --

    def identify_penalty_drivers(self, sequence: str) -> list[dict]:
        """Every difficulty-driving residue/motif, with position and a substitution suggestion."""
        drivers: list[dict] = []
        drivers.extend(self._find_arginine(sequence))
        drivers.extend(self._find_beta_branched_runs(sequence))
        drivers.extend(self._find_dg_ng_motifs(sequence))
        drivers.extend(self._find_oxidation_prone_residues(sequence))
        return sorted(drivers, key=lambda d: d["position"])

    def _find_arginine(self, sequence: str) -> list[dict]:
        return [
            {
                "position": i,
                "kind": "residue",
                "motif": "R",
                "yield_penalty": ARGININE_YIELD_PENALTY,
                "note": "Arginine: Pbf-protected, slow/sterically hindered coupling.",
                "suggestion": "If not functionally required at this position, consider Lys substitution.",
            }
            for i, aa in enumerate(sequence)
            if aa == "R"
        ]

    def _find_beta_branched_runs(self, sequence: str) -> list[dict]:
        """Runs of >=3 consecutive Val/Ile/Thr: backfolds mid-synthesis, stalling coupling."""
        drivers = []
        i = 0
        n = len(sequence)
        while i < n:
            if sequence[i] not in BETA_BRANCHED_RESIDUES:
                i += 1
                continue
            j = i
            while j < n and sequence[j] in BETA_BRANCHED_RESIDUES:
                j += 1
            run_length = j - i
            if run_length >= BETA_BRANCHED_RUN_MIN_LENGTH:
                drivers.append(
                    {
                        "position": i,
                        "kind": "motif",
                        "motif": sequence[i:j],
                        "yield_penalty": BETA_BRANCHED_RUN_YIELD_PENALTY * run_length,
                        "note": f"Run of {run_length} beta-branched/bulky residues (V/I/T) at position {i}: prone to chain backfolding.",
                        "suggestion": "Consider breaking up the run with a non-beta-branched substitution if function allows.",
                    }
                )
            i = j

        return drivers

    def _find_dg_ng_motifs(self, sequence: str) -> list[dict]:
        """Asp-Gly / Asn-Gly dipeptides: succinimide rearrangement to the wrong isomer."""
        drivers = []
        for i in range(len(sequence) - 1):
            motif = sequence[i : i + 2]
            if motif in DG_NG_MOTIFS:
                original = motif[0]
                substitute = "E" if original == "D" else "Q"
                drivers.append(
                    {
                        "position": i,
                        "kind": "motif",
                        "motif": motif,
                        "yield_penalty": DG_NG_MOTIF_YIELD_PENALTY,
                        "note": f"Position {i} {original}-Gly: risk of succinimide rearrangement to the wrong (iso-{original}) isomer.",
                        "suggestion": f"Consider {original}->{substitute} substitution at position {i} if function allows.",
                    }
                )
        return drivers

    def _find_oxidation_prone_residues(self, sequence: str) -> list[dict]:
        """Met/Trp: oxidize during the final cleavage step, not during coupling."""
        return [
            {
                "position": i,
                "kind": "residue",
                "motif": aa,
                "yield_penalty": OXIDATION_RESIDUE_YIELD_PENALTY,
                "note": f"{'Methionine' if aa == 'M' else 'Tryptophan'}: oxidation-prone during final cleavage.",
                "suggestion": "Use a scavenger-rich cleavage cocktail; consider substitution only if function allows.",
            }
            for i, aa in enumerate(sequence)
            if aa in OXIDATION_PRONE_RESIDUES
        ]

    # -- Disulfides and non-standard building blocks --

    def count_disulfide_pairs(self, sequence: str, modifications: dict) -> dict:
        """Cysteine-pair count and whether pairing requires directed folding (>=3 pairs)."""
        positions = [i for i, aa in enumerate(sequence) if aa == "C"]
        n_pairs = len(positions) // 2
        explicit_pairing = modifications.get("disulfide_pairing")

        return {
            "cysteine_positions": positions,
            "n_pairs": n_pairs,
            "unpaired_cysteine": len(positions) % 2 == 1,
            "requires_directed_folding": n_pairs >= DISULFIDE_PAIRS_REQUIRING_DIRECTED_FOLDING,
            "explicit_pairing": explicit_pairing,
        }

    def identify_non_standard_building_blocks(self, modifications: dict) -> list[dict]:
        """Non-standard building blocks present on this candidate, each with a cost multiplier and vendor-availability flag."""
        present = []
        for key, info in NON_STANDARD_BUILDING_BLOCKS.items():
            if modifications.get(key):
                present.append(
                    {
                        "kind": key,
                        "cost_multiplier": info["cost_multiplier"],
                        "routine_vendor_availability": info["routine_vendor_availability"],
                    }
                )
        return present

    # -- Crude purity --

    def compute_crude_purity(self, sequence: str, penalties: list[dict]) -> float:
        """Product of per-step yields across (length - 1) couplings, penalized per flagged position."""
        n_couplings = max(len(sequence) - 1, 0)
        if n_couplings == 0:
            return 1.0

        penalty_by_position: dict[int, float] = {}
        for driver in penalties:
            penalty_by_position[driver["position"]] = (
                penalty_by_position.get(driver["position"], 0.0) + driver["yield_penalty"]
            )

        purity = 1.0
        for position in range(n_couplings):
            step_yield = BASE_STEP_YIELD - penalty_by_position.get(position, 0.0)
            purity *= max(step_yield, 0.0)
        return purity

    # -- Synthesis step count --

    def compute_n_synthesis_steps(
        self, sequence: str, disulfide_pairs: dict, modifications: dict, non_standard: list[dict]
    ) -> int:
        """Backbone couplings + disulfide steps + cyclization + non-standard-block handling."""
        n_couplings = max(len(sequence) - 1, 0)
        n_disulfide_steps = disulfide_pairs["n_pairs"] * DISULFIDE_STEPS_PER_PAIR
        n_cyclization_steps = CYCLIZATION_STEPS if modifications.get("cyclic") else 0
        n_non_standard_steps = len(non_standard) * NON_STANDARD_RESIDUE_STEPS_EACH

        return n_couplings + n_disulfide_steps + n_cyclization_steps + n_non_standard_steps

    # -- Difficulty class --

    def compute_difficulty_class(
        self, penalties: list[dict], disulfide_pairs: dict, non_standard: list[dict]
    ) -> str:
        """low / medium / high, from an additive score over penalties, disulfides, and non-standard blocks."""
        score = len(penalties)
        if disulfide_pairs["requires_directed_folding"]:
            score += 3
        elif disulfide_pairs["n_pairs"] >= 2:
            score += 1
        score += sum(2 for block in non_standard if not block["routine_vendor_availability"])

        if score >= DIFFICULTY_HIGH_MIN_SCORE:
            return "high"
        if score >= DIFFICULTY_MEDIUM_MIN_SCORE:
            return "medium"
        return "low"

    # -- Cost bands --

    def compute_cost_bands(
        self, n_steps: int, difficulty_class: str, disulfide_pairs: dict, non_standard: list[dict]
    ) -> dict:
        """Cost bands for 100 mg (step/labor-dominated) and 10 g (purification/QC-overhead-dominated) scale."""
        cost_multiplier = 1.0
        for block in non_standard:
            cost_multiplier *= block["cost_multiplier"]

        small_scale_score = n_steps * cost_multiplier
        if difficulty_class == "high":
            small_scale_score *= 2.0
        elif difficulty_class == "medium":
            small_scale_score *= 1.4

        large_scale_score = n_steps * cost_multiplier
        if disulfide_pairs["requires_directed_folding"]:
            large_scale_score *= 2.5
        elif disulfide_pairs["n_pairs"] >= 2:
            large_scale_score *= 1.5
        large_scale_score *= 1.0 + 0.5 * len(non_standard)

        return {
            "per_100mg": self._score_to_cost_band(small_scale_score),
            "per_10g": self._score_to_cost_band(large_scale_score),
        }

    def _score_to_cost_band(self, score: float) -> str:
        if score < 30:
            return "low"
        if score < 60:
            return "medium"
        if score < 120:
            return "high"
        return "very_high"

    # -- Achievable final-purity ceiling --

    def compute_final_purity_ceiling(
        self, crude_purity: float, penalties: list[dict], disulfide_pairs: dict, non_standard: list[dict]
    ) -> float:
        """Crude purity times a purification-recovery penalty for co-eluting isomers/diastereomers."""
        recovery = FINAL_PURITY_BASE_RECOVERY

        extra_pairs = max(disulfide_pairs["n_pairs"] - 1, 0)
        recovery -= extra_pairs * FINAL_PURITY_DISULFIDE_RECOVERY_PENALTY

        n_dg_ng = sum(1 for driver in penalties if driver["motif"] in DG_NG_MOTIFS)
        recovery -= n_dg_ng * FINAL_PURITY_DG_NG_RECOVERY_PENALTY

        recovery -= len(non_standard) * FINAL_PURITY_NON_STANDARD_RECOVERY_PENALTY

        recovery = max(recovery, 0.0)
        return crude_purity * recovery

    # -- Supplementary ML prior --

    def compute_ml_feasibility_prior(self, sequence: str, model: SynthesisFeasibilityEnsemble) -> dict:
        """Synthesis-feasibility probability from synthesis_feasibility_predictor_v1; weak prior, never a gate."""
        score = model.predict_proba(sequence)
        return {
            "score": score,
            "status": "ok",
            # "caveat": "weak_generalization_not_a_gate"
        }
