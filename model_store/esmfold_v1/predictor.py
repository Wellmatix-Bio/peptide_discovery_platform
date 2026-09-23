"""Loader/inference wrapper for ESMFold single-sequence structure prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from common.model_sync import sync_model_weights, weights_dir_for

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)
WEIGHTS_DIR = MODEL_DIR / "weights"

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

MODEL_NAME = "facebook/esmfold_v1"
DEFAULT_CHUNK_SIZE = 64
CA_ATOM37_INDEX = 1  # atom37 layout: index 1 = CA atom


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


def _extract_ca_coordinates(pdb_str: str) -> np.ndarray:
    """(L, 3) CA coordinates in Angstroms, parsed from the predicted PDB in residue order."""
    coordinates = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(coordinates, dtype=np.float32)


def compute_ca_distance_matrix(ca_coordinates: np.ndarray) -> np.ndarray:
    """(L, L) pairwise CA-CA Euclidean distance matrix in Angstroms."""
    diff = ca_coordinates[:, None, :] - ca_coordinates[None, :, :]
    return np.sqrt((diff**2).sum(-1)).astype(np.float32)


class ESMFoldPredictor:
    """Lazy-loaded ESMFold (facebook/esmfold_v1) single-sequence structure predictor."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, fp16_esm: bool = True):
        self.chunk_size = chunk_size
        self.fp16_esm = fp16_esm
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        if not WEIGHTS_DIR.exists():
            raise FileNotFoundError(f"No local weights at {WEIGHTS_DIR}; expected a vendored snapshot of {MODEL_NAME}.")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))
        self.model = EsmForProteinFolding.from_pretrained(str(WEIGHTS_DIR))
        self.model.eval()

        if self.fp16_esm and torch.cuda.is_available():
            self.model.esm = self.model.esm.half()
        self.model.trunk.set_chunk_size(self.chunk_size)
        self.model.to(self.device)
        self.loaded = True

    def predict(self, sequence: str) -> dict:
        """Folds `sequence`; pLDDT is a 0-1 fraction here (this transformers version), not the usual 0-100 scale."""
        _validate_sequence(sequence)
        self._load()

        with torch.no_grad():
            output = self.model.infer([sequence])
        pdb_str = self.model.output_to_pdb(output)[0]

        per_residue_plddt = output["plddt"][0, : len(sequence), CA_ATOM37_INDEX].cpu().numpy()
        mean_plddt = float(per_residue_plddt.mean())

        ca_coordinates = _extract_ca_coordinates(pdb_str)
        ca_distance_matrix = compute_ca_distance_matrix(ca_coordinates)

        return {
            "sequence": sequence,
            "pdb_str": pdb_str,
            "mean_plddt": mean_plddt,
            "per_residue_plddt": per_residue_plddt.tolist(),
            "ca_coordinates": ca_coordinates.tolist(),
            "ca_distance_matrix": ca_distance_matrix.tolist(),
        }
