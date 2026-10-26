"""Loader/inference wrapper for UniZyme (Transformer_DE): per-residue protease cleavage sites in a candidate peptide, from its sequence and the proteases of interest."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from common.model_sync import sync_model_weights, weights_dir_for

from .architecture import Transformer_DE

logger = logging.getLogger(__name__)

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)
ENZYMES_DIR = MODEL_DIR / "enzymes"
CHECKPOINT_PATH = MODEL_DIR / "UniZyme.pt"

# ESM2-35M is shared weights, synced like the 150M model: a virtual code dir names its weights folder.
_ESM2_CODE_DIR = CODE_DIR.parent / "esm2_t12_35M"
ESM_DIR = weights_dir_for(_ESM2_CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

MAX_LEN = 1500  # the model is trained on matrices zero-padded to this length
NUM_ATTENTION_HEADS = 2
D_MODEL = 480  # ESM2-35M embedding size
N_LAYERS = 1
K = 10
DEFAULT_BATCH_SIZE = 16

# Cleavage probability above which a residue counts as a predicted site; not calibrated for this model.
CLEAVAGE_PROB_THRESHOLD = 0.5


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )
    if len(sequence) > MAX_LEN:
        raise ValueError(
            f"sequence length {len(sequence)} exceeds the model's maximum of {MAX_LEN}"
        )


def _validate_distance_matrix(distance_matrix: np.ndarray, sequence: str) -> None:
    n = len(sequence)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match "
            f"sequence length {n} -- expected ({n}, {n})."
        )


def _load_enzyme_catalog() -> list[dict]:
    with open(
        ENZYMES_DIR / "enzyme_catalog.csv", "r", encoding="utf-8", newline=""
    ) as f:
        return list(csv.DictReader(f))


def _inverse_distance(distance_matrix: np.ndarray) -> torch.Tensor:
    """The model's distance input: 1 / (Ca-Ca distance in Angstroms + 1)."""
    distances = np.nan_to_num(np.asarray(distance_matrix, dtype=np.float32), nan=0.0)
    return torch.from_numpy(1.0 / (distances + 1.0))


def _pad_square(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0]
    return F.pad(matrix, (0, MAX_LEN - n, 0, MAX_LEN - n))


def _mask(length: int, device) -> torch.Tensor:
    mask = torch.zeros(1, NUM_ATTENTION_HEADS, MAX_LEN, MAX_LEN)
    mask[..., :length, :length] = 1.0
    return mask.to(device)


class CleavageSitePredictor:
    """Lazy-loaded UniZyme scoring a peptide as substrate against the bundled protease panel (incl. DPP4, NEP, PPCE) per residue."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False
        self._enzyme_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
        ] = {}
        self._substrate_cache: dict[str, torch.Tensor] = {}
        self._structure_cache: dict[str, np.ndarray] = {}
        self._encoded_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._folder = None

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        sync_model_weights(_ESM2_CODE_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(ESM_DIR, local_files_only=True)
        self.esm_model = (
            AutoModel.from_pretrained(ESM_DIR, local_files_only=True)
            .to(self.device)
            .eval()
        )

        self.enzyme_catalog = _load_enzyme_catalog()
        self.enzyme_by_accession = {
            row["enzyme_uniprot"]: row for row in self.enzyme_catalog
        }

        model = Transformer_DE(
            d_model=D_MODEL, n_heads=NUM_ATTENTION_HEADS, n_layers=N_LAYERS, K=K
        )
        state = torch.load(CHECKPOINT_PATH, map_location=self.device)
        # The checkpoint was saved from nn.DataParallel, so strip the "module." prefix.
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "cleavage_site_predictor_v2.checkpoint_mismatch",
                extra={
                    "missing": result.missing_keys,
                    "unexpected": result.unexpected_keys,
                },
            )
        self.model = model.to(self.device).eval()
        self.loaded = True

    def available_enzymes(self) -> list[dict]:
        """The bundled enzyme panel's catalog rows (accession, name, MEROPS id, length, sequence); lazy-loads only the catalog."""
        if not self.loaded:
            sync_model_weights(CODE_DIR)
            self.enzyme_catalog = _load_enzyme_catalog()
        return self.enzyme_catalog

    def _esm_features(self, sequence: str) -> torch.Tensor:
        """(L, 480) final-layer ESM2-35M residue embeddings with BOS/EOS stripped, cached per sequence."""
        cached = self._substrate_cache.get(sequence)
        if cached is None:
            enc = self.tokenizer(
                sequence, return_tensors="pt", return_special_tokens_mask=True
            )
            special = enc.pop("special_tokens_mask")[0].bool()
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                cached = (
                    self.esm_model(**enc)
                    .last_hidden_state[0][~special.to(self.device)]
                    .float()
                )
            if len(self._substrate_cache) >= 64:
                self._substrate_cache.pop(next(iter(self._substrate_cache)))
            self._substrate_cache[sequence] = cached
        return cached

    def _fold(self, sequence: str) -> np.ndarray:
        """(L, L) Ca-Ca distance matrix in Angstroms from an ESMFold structure of the peptide, cached per sequence."""
        cached = self._structure_cache.get(sequence)
        if cached is None:
            if self._folder is None:
                from model_store.esmfold_v1 import ESMFoldPredictor

                self._folder = ESMFoldPredictor()
            cached = np.asarray(
                self._folder.predict(sequence)["ca_distance_matrix"], dtype=np.float32
            )
            if len(self._structure_cache) >= 64:
                self._structure_cache.pop(next(iter(self._structure_cache)))
            self._structure_cache[sequence] = cached
        return cached

    def _enzyme_tensors(
        self, accession: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """ESM features, inverse-distance matrix and frustration matrix for a bundled enzyme, unpadded, cached."""
        if accession not in self.enzyme_by_accession:
            raise ValueError(
                f"Unknown enzyme accession {accession!r}. Call available_enzymes() for the bundled panel."
            )
        if accession not in self._enzyme_cache:
            row = self.enzyme_by_accession[accession]
            features = self._esm_features(row["enzyme_sequence"])
            coords = np.load(ENZYMES_DIR / "structures" / f"{accession}.npy").astype(
                np.float32
            )
            distances = _inverse_distance(
                np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
            )
            energy = torch.from_numpy(
                np.load(ENZYMES_DIR / "energy_matrices" / f"{accession}.npy").astype(
                    np.float32
                )
            )
            self._enzyme_cache[accession] = (
                features,
                distances.to(self.device),
                energy.to(self.device),
                len(row["enzyme_sequence"]),
            )
        return self._enzyme_cache[accession]

    def _encoded_enzyme(
        self, accession: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder output, active-site logits and mask for a bundled enzyme; the substrate never changes them, so cached."""
        if accession not in self._encoded_cache:
            features, distances, energy, length = self._enzyme_tensors(accession)
            mask = _mask(length, self.device)
            with torch.no_grad():
                encoded, site_logits = self.model.encode_enzyme(
                    F.pad(features, (0, 0, 0, MAX_LEN - length)).unsqueeze(0),
                    _pad_square(distances).unsqueeze(0),
                    _pad_square(energy).unsqueeze(0),
                    mask,
                )
            self._encoded_cache[accession] = (encoded, site_logits, mask)
        return self._encoded_cache[accession]

    def predict_batch(
        self,
        sequences: list[str],
        enzyme_accession: str,
        distance_matrices: list[np.ndarray | None] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[dict]:
        """Cleavage-site probabilities for many peptides against one bundled enzyme, `batch_size` peptides per forward pass.

        Peptides are folded with ESMFold unless `distance_matrices` ((L, L) Angstroms each, None to fold) is given.
        """
        if not sequences:
            return []
        for sequence in sequences:
            _validate_sequence(sequence)
        matrices = (
            list(distance_matrices)
            if distance_matrices is not None
            else [None] * len(sequences)
        )
        if len(matrices) != len(sequences):
            raise ValueError("distance_matrices must have one entry per sequence.")
        matrices = [
            self._fold(seq) if m is None else m for seq, m in zip(sequences, matrices)
        ]
        for seq, m in zip(sequences, matrices):
            _validate_distance_matrix(m, seq)
        self._load()

        encoded, site_logits, mask = self._encoded_enzyme(enzyme_accession)
        name = self.enzyme_by_accession[enzyme_accession]["protein_name"]
        results = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            features = torch.stack(
                [
                    F.pad(self._esm_features(s), (0, 0, 0, MAX_LEN - len(s)))
                    for s in chunk
                ]
            )
            distances = torch.stack(
                [
                    _pad_square(_inverse_distance(m).to(self.device))
                    for m in matrices[start : start + batch_size]
                ]
            ).unsqueeze(
                1
            )  # (batch, 1, L, L): the head axis keeps batch from broadcasting against heads
            masks = torch.cat([_mask(len(s), self.device) for s in chunk])
            with torch.no_grad():
                cleavage = self.model.cleave_substrates(
                    features, distances, masks, encoded, site_logits, mask
                )
            for seq, row in zip(chunk, cleavage.squeeze(-1).cpu().numpy()):
                probs = row[: len(seq)]
                positions = [
                    i for i, p in enumerate(probs) if p >= CLEAVAGE_PROB_THRESHOLD
                ]
                results.append(
                    {
                        "enzyme_accession": enzyme_accession,
                        "enzyme_name": name,
                        "per_residue_probability": [float(p) for p in probs],
                        "cleavage_site_positions": positions,
                        "max_probability": float(probs.max()) if len(probs) else 0.0,
                        "n_predicted_sites": len(positions),
                    }
                )
        return results

    def predict_one(
        self,
        sequence: str,
        enzyme_accession: str,
        distance_matrix: np.ndarray | None = None,
    ) -> dict:
        """Cleavage-site probability at every residue for one bundled enzyme.

        Only the sequence is needed: the peptide is folded with ESMFold for its Ca distance matrix. Pass
        `distance_matrix` ((L, L) Angstroms) to reuse a structure the caller already has.
        """
        return self.predict_batch([sequence], enzyme_accession, [distance_matrix])[0]

    def predict(
        self,
        sequence: str,
        enzyme_accessions: list[str],
        distance_matrix: np.ndarray | None = None,
    ) -> dict:
        """Runs predict_one() against each listed enzyme and returns per-enzyme results plus an overall susceptibility flag.

        The panel has thousands of enzymes, so the proteases of interest must be named; the peptide is folded once and reused.
        """
        if not enzyme_accessions:
            raise ValueError(
                "enzyme_accessions must name at least one enzyme; see available_enzymes() for the panel."
            )
        _validate_sequence(sequence)
        if distance_matrix is None:
            distance_matrix = self._fold(sequence)
        self._load()
        accessions = list(enzyme_accessions)

        per_enzyme = {
            accession: self.predict_one(sequence, accession, distance_matrix)
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
