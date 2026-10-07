"""Loader/inference wrapper for the joint model scoring a candidate peptide's protease susceptibility; needs a Ca distance matrix"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel

from common.model_sync import sync_model_weights, weights_dir_for

from .architecture import JointModel

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)
ENZYMES_DIR = MODEL_DIR / "enzymes"
CHECKPOINT_PATH = MODEL_DIR / "joint_model_best.pt"

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
ESM_RAW_EMBED_DIM = 640

# Cleavage logit to probability per substrate residue; a P1 site at t means the bond C-terminal to t is cut.
CLEAVAGE_PROB_THRESHOLD = 0.5


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


def _validate_distance_matrix(distance_matrix: np.ndarray, sequence: str) -> None:
    n = len(sequence)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match "
            f"sequence length {n} -- expected ({n}, {n})."
        )


def compute_esm_embedding(sequence: str, tokenizer, esm_model, device: str) -> np.ndarray:
    """(L, ESM_RAW_EMBED_DIM) float32 array with BOS/EOS stripped via special_tokens_mask, matching source pipeline.py."""
    enc = tokenizer(sequence, return_tensors="pt", return_special_tokens_mask=True)
    special_mask = enc.pop("special_tokens_mask")[0].bool()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = esm_model(**enc)

    residue_emb = out.last_hidden_state[0][~special_mask]
    if residue_emb.shape[0] != len(sequence):
        raise ValueError(
            f"embedding length {residue_emb.shape[0]} != sequence length {len(sequence)}"
        )
    return residue_emb.float().cpu().numpy()


def _load_enzyme_catalog() -> list[dict]:
    with open(ENZYMES_DIR / "enzyme_catalog.csv", "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class CleavageSitePredictor:
    """Lazy-loaded JointModel scoring a peptide as substrate against 118 wound-relevant proteases per residue; needs a Ca distance matrix (see README.md)."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        self.esm_model = AutoModel.from_pretrained(ESM_MODEL_NAME).to(self.device).eval()
        self.pca_model = joblib.load(ENZYMES_DIR / "pca_model.joblib")

        self.enzyme_catalog = _load_enzyme_catalog()
        self.enzyme_by_accession = {row["enzyme_uniprot"]: row for row in self.enzyme_catalog}

        self.model = JointModel.load_pretrained(CHECKPOINT_PATH, device=self.device)
        self._enzyme_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.loaded = True

    def available_enzymes(self) -> list[dict]:
        """The bundled enzyme panel's catalog rows; lazy-loads only the catalog."""
        if not self.loaded:
            sync_model_weights(CODE_DIR)
            self.enzyme_catalog = _load_enzyme_catalog()
        return self.enzyme_catalog

    def _enzyme_tensors(self, accession: str) -> tuple[torch.Tensor, torch.Tensor]:
        """PCA-reduced embedding and distance matrix for a bundled enzyme, cached for the predictor's life."""
        if accession not in self.enzyme_by_accession:
            raise ValueError(
                f"Unknown enzyme accession {accession!r}. Call available_enzymes() "
                f"for the bundled panel."
            )
        if accession in self._enzyme_cache:
            return self._enzyme_cache[accession]

        embedding = np.load(ENZYMES_DIR / "embeddings" / f"{accession}.npy").astype(np.float32)
        distances = np.load(ENZYMES_DIR / "distance_matrices" / f"{accession}.npy").astype(np.float32)

        embedding_t = torch.from_numpy(embedding).unsqueeze(0).to(self.device)
        distances_t = torch.from_numpy(distances).unsqueeze(0).to(self.device)
        self._enzyme_cache[accession] = (embedding_t, distances_t)
        return embedding_t, distances_t

    def _substrate_tensors(
        self, sequence: str, distance_matrix: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_embedding = compute_esm_embedding(sequence, self.tokenizer, self.esm_model, self.device)
        reduced = self.pca_model.transform(raw_embedding).astype(np.float32)

        embedding_t = torch.from_numpy(reduced).unsqueeze(0).to(self.device)
        distances_t = torch.from_numpy(distance_matrix.astype(np.float32)).unsqueeze(0).to(self.device)
        return embedding_t, distances_t

    def predict_one(
        self, sequence: str, distance_matrix: np.ndarray, enzyme_accession: str
    ) -> dict:
        """Cleavage-site probability at every residue for one bundled enzyme; `distance_matrix` is the caller's (L, L) Ca matrix in Angstroms."""
        _validate_sequence(sequence)
        _validate_distance_matrix(distance_matrix, sequence)
        self._load()

        enzyme_embedding, enzyme_distances = self._enzyme_tensors(enzyme_accession)
        enzyme_mask = torch.ones(1, enzyme_embedding.shape[1], dtype=torch.bool, device=self.device)

        substrate_embedding, substrate_distances = self._substrate_tensors(sequence, distance_matrix)
        substrate_mask = torch.ones(1, len(sequence), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            out = self.model.forward_cleavage(
                enzyme_embedding, enzyme_distances, enzyme_mask,
                substrate_embedding, substrate_distances, substrate_mask,
            )
            probs = torch.sigmoid(out["cleavage_logits"])[0].cpu().numpy()

        positions = [i for i, p in enumerate(probs) if p >= CLEAVAGE_PROB_THRESHOLD]
        enzyme_row = self.enzyme_by_accession[enzyme_accession]
        return {
            "enzyme_accession": enzyme_accession,
            "enzyme_name": enzyme_row["protein_name"],
            "per_residue_probability": probs.tolist(),
            "cleavage_site_positions": positions,
            "max_probability": float(probs.max()) if len(probs) else 0.0,
            "n_predicted_sites": len(positions),
        }

    def predict(
        self, sequence: str, distance_matrix: np.ndarray, enzyme_accessions: list[str] | None = None
    ) -> dict:
        """Runs predict_one() against each enzyme (default: the full panel) and returns per-enzyme results plus an overall susceptibility flag."""
        self._load()
        accessions = enzyme_accessions or [row["enzyme_uniprot"] for row in self.enzyme_catalog]

        per_enzyme = {
            accession: self.predict_one(sequence, distance_matrix, accession)
            for accession in accessions
        }
        susceptible_to = [
            acc for acc, result in per_enzyme.items() if result["n_predicted_sites"] > 0
        ]
        return {
            "per_enzyme": per_enzyme,
            "susceptible_to": susceptible_to,
            "any_predicted_cleavage": len(susceptible_to) > 0,
        }
