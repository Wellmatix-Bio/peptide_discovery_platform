# Stage 7: Structure and Mechanism.
from __future__ import annotations
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import freesasa
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

from model_store.esmfold_v1 import ESMFoldPredictor  # noqa: E402
from model_store.pathway_mapping_predictor_v1 import (
    PathwayMappingPredictor,
)  # noqa: E402

# Default thresholds (screening table). Override per-run via
# StageConfig.params — every key here is read with config.params.get(key, default).
DEFAULT_THRESHOLDS = {
    # pLDDT is a 0-1 fraction for this checkpoint.
    "plddt_low_confidence_max": 0.5,
    # Pathway-label probability threshold for "engaged" in the mechanistic summary.
    "pathway_engagement_min_probability": 0.5,
    # Ramachandran-region backbone dihedral classification (alpha-helix / beta-sheet boxes).
    "helix_phi_min": -100.0,
    "helix_phi_max": -30.0,
    "helix_psi_min": -77.0,
    "helix_psi_max": -5.0,
    "sheet_phi_min": -180.0,
    "sheet_phi_max": -45.0,
    "sheet_psi_min": 90.0,
    "sheet_psi_max": 180.0,
}

# Pathway label -> target peptide function (CLAUDE.md's five); antimicrobial action has no pathway mapping here, it comes from Stage 6's AMP/MIC models.
PATHWAY_TO_FUNCTION = {
    "NF_KB": "immunomodulation",
    "CYTOKINE_MACROPHAGE": "immunomodulation",
    "TGFB_SMAD": "collagen synthesis",
    "VEGF_ANGIOGENESIS": "angiogenesis",
    "MAPK": "cell proliferation/migration",
    "ERK": "cell proliferation/migration",
    "PI3K_AKT_MTOR": "cell proliferation/migration",
    "WNT_BCATENIN": "cell proliferation/migration",
    "FGFR_JAK2_STAT3": "cell proliferation/migration",
}


def _get_esmfold_model() -> ESMFoldPredictor:
    global _esmfold_model
    if _esmfold_model is None:
        _esmfold_model = ESMFoldPredictor()
    return _esmfold_model


def _get_pathway_model() -> PathwayMappingPredictor:
    global _pathway_model
    if _pathway_model is None:
        _pathway_model = PathwayMappingPredictor()
    return _pathway_model


_esmfold_model: ESMFoldPredictor | None = None
_pathway_model: PathwayMappingPredictor | None = None


@dataclass(frozen=True)
class Stage7Models:
    """Bundle returned by build_models(): every model this stage depends on, lazy-loaded on first use of each."""

    esmfold: ESMFoldPredictor
    pathway: PathwayMappingPredictor


def build_models() -> Stage7Models:
    """Factory: import and initialize every Stage 7 model."""
    return Stage7Models(esmfold=_get_esmfold_model(), pathway=_get_pathway_model())


class Stage7(CandidateStage):
    name = "s07_structure_mechanism"
    produces = {"structure", "mechanism"}

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        """Predicts 3D structure (ESMFold) and pathway involvement, then combines them into a mechanistic evidence summary."""
        models = build_models()
        t = {
            **DEFAULT_THRESHOLDS,
            **{k: v for k, v in config.params.items() if k in DEFAULT_THRESHOLDS},
        }

        for candidate in tqdm(candidates, desc="Stage 7"):
            structure = self.predict_structure(candidate.sequence, models.esmfold, t)
            candidate.predictions["structure"] = structure

            pathway_involvement = self.predict_pathway_involvement(
                candidate.sequence, models.pathway
            )
            candidate.predictions["mechanism"] = (
                self.build_mechanistic_evidence_summary(
                    structure, pathway_involvement, t
                )
            )

        return candidates

    def models_used(self) -> list[ModelRef]:
        return [
            ModelRef(name="esmfold", version="v1"),
            ModelRef(name="pathway_mapping_predictor", version="v1"),
        ]

    def release_models(self) -> None:
        release_stage_models(globals(), stage_name=self.name)

    # -- Structure prediction --

    def predict_structure(
        self, sequence: str, model: ESMFoldPredictor, thresholds: dict
    ) -> dict:
        """ESMFold structure: coordinates, distance matrix, structural confidence, secondary-structure characteristics, exposure."""
        result = model.predict(sequence)
        # ca_coordinates = np.asarray(result["ca_coordinates"], dtype=np.float32)
        backbone_atoms = self._extract_backbone_atoms(result["pdb_str"])

        return {
            "pdb_str": result["pdb_str"],
            "ca_coordinates": result["ca_coordinates"],
            "ca_distance_matrix": result["ca_distance_matrix"],
            "confidence": self.compute_structural_confidence(
                result["per_residue_plddt"], thresholds
            ),
            "secondary_structure": self.compute_secondary_structure(
                backbone_atoms, thresholds
            ),
            "exposure_by_position": self.compute_exposure(
                result["pdb_str"], len(sequence)
            ),
        }

    def _extract_backbone_atoms(self, pdb_str: str) -> dict[int, dict[str, np.ndarray]]:
        """{residue_index: {"N": xyz, "CA": xyz, "C": xyz}}, parsed from the predicted PDB in residue order."""
        atoms: dict[int, dict[str, np.ndarray]] = {}
        residue_order: list[int] = []
        seen_residue_numbers: dict[int, int] = {}
        for line in pdb_str.splitlines():
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "CA", "C"):
                continue
            residue_number = int(line[22:26])
            if residue_number not in seen_residue_numbers:
                seen_residue_numbers[residue_number] = len(residue_order)
                residue_order.append(residue_number)
            index = seen_residue_numbers[residue_number]
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            atoms.setdefault(index, {})[atom_name] = xyz
        return atoms

    # -- Structural confidence --

    def compute_structural_confidence(
        self, per_residue_plddt: list[float], thresholds: dict
    ) -> dict:
        """Mean pLDDT plus the count/positions of low-confidence residues."""
        plddt = np.asarray(per_residue_plddt, dtype=np.float32)
        low_confidence_positions = [
            i for i, p in enumerate(plddt) if p < thresholds["plddt_low_confidence_max"]
        ]
        return {
            "mean_plddt": float(plddt.mean()) if len(plddt) else 0.0,
            "per_residue_plddt": per_residue_plddt,
            "low_confidence_positions": low_confidence_positions,
            "n_low_confidence": len(low_confidence_positions),
        }

    # -- Secondary structure (geometric, from ESMFold backbone dihedrals) --

    def compute_secondary_structure(
        self, backbone_atoms: dict[int, dict[str, np.ndarray]], thresholds: dict
    ) -> dict:
        """Per-residue H/E/C class from phi/psi backbone dihedrals, plus fold-composition fractions."""
        n_residues = len(backbone_atoms)
        classes = [
            self._classify_residue(backbone_atoms, i, n_residues, thresholds)
            for i in range(n_residues)
        ]
        ss_string = "".join(classes)

        counts = {c: ss_string.count(c) for c in "HEC"}
        fractions = {c: (counts[c] / n_residues if n_residues else 0.0) for c in "HEC"}
        dominant_class = max(fractions, key=fractions.get) if n_residues else "C"

        return {
            "predicted_ss": ss_string,
            "helix_fraction": fractions["H"],
            "sheet_fraction": fractions["E"],
            "coil_fraction": fractions["C"],
            "dominant_class": dominant_class,
        }

    def _classify_residue(
        self,
        backbone_atoms: dict[int, dict[str, np.ndarray]],
        i: int,
        n_residues: int,
        thresholds: dict,
    ) -> str:
        if i == 0 or i == n_residues - 1:
            return "C"  # phi needs residue i-1's C, psi needs residue i+1's N; chain termini have neither
        phi = self._dihedral(
            backbone_atoms[i - 1]["C"],
            backbone_atoms[i]["N"],
            backbone_atoms[i]["CA"],
            backbone_atoms[i]["C"],
        )
        psi = self._dihedral(
            backbone_atoms[i]["N"],
            backbone_atoms[i]["CA"],
            backbone_atoms[i]["C"],
            backbone_atoms[i + 1]["N"],
        )
        if (
            thresholds["helix_phi_min"] <= phi <= thresholds["helix_phi_max"]
            and thresholds["helix_psi_min"] <= psi <= thresholds["helix_psi_max"]
        ):
            return "H"
        if (
            thresholds["sheet_phi_min"] <= phi <= thresholds["sheet_phi_max"]
            and thresholds["sheet_psi_min"] <= psi <= thresholds["sheet_psi_max"]
        ):
            return "E"
        return "C"

    def _dihedral(
        self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
    ) -> float:
        """Dihedral angle (degrees) for four consecutive backbone atoms."""
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return float(np.degrees(np.arctan2(y, x)))

    # -- Exposure (geometric, from ESMFold structure via freesasa) --

    def compute_exposure(self, pdb_str: str, length: int) -> list[float]:
        """Per-residue relative solvent exposure in [0, 1], normalized SASA from freesasa on the predicted structure."""
        fd, path = tempfile.mkstemp(suffix=".pdb")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(pdb_str)
            structure = freesasa.Structure(path)
        finally:
            os.remove(path)

        result = freesasa.calc(structure)
        residue_areas = result.residueAreas().get("A", {})

        sasa_by_position = [
            residue_areas[str(i + 1)].total if str(i + 1) in residue_areas else 0.0
            for i in range(length)
        ]
        max_sasa = max(sasa_by_position) if any(sasa_by_position) else 1.0
        return [min(sasa / max_sasa, 1.0) for sasa in sasa_by_position]

    # -- Pathway involvement --

    def predict_pathway_involvement(
        self, sequence: str, model: PathwayMappingPredictor
    ) -> dict:
        """Per-pathway engagement probability from pathway_mapping_predictor_v1."""
        return model.predict(sequence)

    # -- Mechanistic evidence summary --

    def build_mechanistic_evidence_summary(
        self, structure: dict, pathway_involvement: dict, thresholds: dict
    ) -> dict:
        """Combines structure and pathway evidence into the Stage 7 MoA summary (CLAUDE.md's MoA-map output)."""
        engaged_pathways = [
            label
            for label, result in pathway_involvement.items()
            if result["probability"] >= thresholds["pathway_engagement_min_probability"]
        ]
        functions_supported = sorted(
            {
                PATHWAY_TO_FUNCTION[label]
                for label in engaged_pathways
                if label in PATHWAY_TO_FUNCTION
            }
        )

        return {
            "pathway_involvement": pathway_involvement,
            "engaged_pathways": engaged_pathways,
            "functions_supported": functions_supported,
            "structural_confidence": structure["confidence"]["mean_plddt"],
            "dominant_secondary_structure": structure["secondary_structure"][
                "dominant_class"
            ],
            "summary": self._render_summary(
                structure, engaged_pathways, functions_supported
            ),
        }

    def _render_summary(
        self,
        structure: dict,
        engaged_pathways: list[str],
        functions_supported: list[str],
    ) -> str:
        confidence = structure["confidence"]["mean_plddt"]
        ss_class = structure["secondary_structure"]["dominant_class"]
        pathway_text = (
            ", ".join(engaged_pathways) if engaged_pathways else "none above threshold"
        )
        function_text = (
            ", ".join(functions_supported) if functions_supported else "none"
        )
        return (
            f"Structure: mean pLDDT {confidence:.2f}, dominant fold {ss_class}. "
            f"Pathways engaged: {pathway_text}. Target functions supported: {function_text}."
        )
