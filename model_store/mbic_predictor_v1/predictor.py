"""Loader/inference wrapper for the MBIC (biofilm inhibition) SVR model.

Predicts pMBIC = 6 - log10(activity_uM), a pIC50-style log-scale potency
value against biofilm, NOT a raw concentration and NOT a [0, 1] probability.
To recover the concentration: activity_uM = 10 ** (6 - pMBIC). Higher pMBIC
means a lower (more potent) MBIC concentration. See README.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import joblib
import torch
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
from transformers import AutoTokenizer, EsmModel

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

PHYSCHEM_COLS = [
    "charge", "charge_density", "gravy", "hydrophobic_moment", "amphipathicity",
    "helical_propensity", "aliphatic_index", "aromaticity", "boman_index",
    "instability_index", "isoelectric_point", "hydrophobic_ratio", "length",
    "d_residue_fraction",
]


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence.upper()):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence.upper()) - AMINO_ACID_SET)}"
        )


def esm_embedding(sequence: str, tokenizer, esm_model, device: str) -> np.ndarray:
    return esm_embedding_batch([sequence], tokenizer, esm_model, device)[0]


def esm_embedding_batch(sequences: list[str], tokenizer, esm_model, device: str) -> np.ndarray:
    seqs = [s.strip().upper() for s in sequences]
    with torch.no_grad():
        enc = tokenizer(seqs, return_tensors="pt", padding=True, add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = esm_model(**enc).last_hidden_state
        # Exclude padding and the BOS/EOS special tokens from the mean pool.
        # EOS position varies per row (each sequence has its own length), so
        # it must be zeroed per-row, not with a single batch-wide index.
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0
        lengths = enc["attention_mask"].sum(dim=1)
        mask[torch.arange(mask.size(0), device=device), lengths - 1] = 0
        mask = mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.float().cpu().numpy()


def esm_embedding_cached(sequence: str, feature_extractor: FeatureExtractor) -> np.ndarray:
    return esm_embedding_cached_batch([sequence], feature_extractor)[0]


def esm_embedding_cached_batch(
    sequences: list[str], feature_extractor: FeatureExtractor
) -> np.ndarray:
    """Mean-pool over cached per-token hidden states, excluding the BOS/EOS
    special tokens (positions 0 and -1 of the cached, already-trimmed
    per-sequence tensor) and any padding (already stripped by the cache)."""
    seqs = [s.strip().upper() for s in sequences]
    embeddings = feature_extractor.get_esm2_embedding_batch(seqs)
    pooled = np.stack([e.hidden_states[1:-1].mean(axis=0) for e in embeddings])
    return pooled.astype(np.float32)


def physchem_descriptors(sequence: str, ph: float = 7.4, amide: bool = False, window: int = 18, angle: int = 100) -> dict:
    return physchem_descriptors_batch([sequence], ph=ph, amide=amide, window=window, angle=angle)[0]


def physchem_descriptors_batch(sequences: list[str], ph: float = 7.4, amide: bool = False, window: int = 18, angle: int = 100) -> list[dict]:
    raws = [s.strip() for s in sequences]
    seqs = [r.upper() for r in raws]
    d_fractions = [
        (sum(1 for c in raw if c.islower()) / len(raw)) if len(raw) else np.nan
        for raw in raws
    ]

    def _global(method, **kw):
        g = GlobalDescriptor(seqs)
        getattr(g, method)(**kw)
        return g.descriptor.ravel()

    def _scaled(scale, method, **kw):
        p = PeptideDescriptor(seqs, scale)
        getattr(p, method)(**kw)
        return p.descriptor.ravel()

    # Eisenberg hydrophobic moment <uH>, max over an alpha-helical (100deg) window.
    moment = _scaled("eisenberg", "calculate_moment", window=window, angle=angle, modality="max")

    # Amphipathicity = <uH> normalized by mean |H|; distinguishes a genuinely
    # faced helix from a peptide that is merely uniformly hydrophobic.
    p_abs = PeptideDescriptor(seqs, "eisenberg")
    p_abs.calculate_global(modality="mean")
    mean_h = np.abs(p_abs.descriptor.ravel())
    amphipathicity = np.where(mean_h > 1e-9, moment / np.where(mean_h > 1e-9, mean_h, 1.0), np.nan)

    charge = _global("calculate_charge", ph=ph, amide=amide)
    charge_density = _global("charge_density", ph=ph, amide=amide)
    gravy = _scaled("gravy", "calculate_global", modality="mean")
    helical_propensity = _scaled("levitt_alpha", "calculate_global", modality="mean")
    aliphatic_index = _global("aliphatic_index")
    aromaticity = _global("aromaticity")
    boman_index = _global("boman_index")
    instability_index = _global("instability_index")
    isoelectric_point = _global("isoelectric_point")
    hydrophobic_ratio = _global("hydrophobic_ratio")
    length = _global("length")

    return [
        {
            "charge": float(charge[i]),
            "charge_density": float(charge_density[i]),
            "gravy": float(gravy[i]),
            "hydrophobic_moment": float(moment[i]),
            "amphipathicity": float(amphipathicity[i]),
            "helical_propensity": float(helical_propensity[i]),
            "aliphatic_index": float(aliphatic_index[i]),
            "aromaticity": float(aromaticity[i]),
            "boman_index": float(boman_index[i]),
            "instability_index": float(instability_index[i]),
            "isoelectric_point": float(isoelectric_point[i]),
            "hydrophobic_ratio": float(hydrophobic_ratio[i]),
            "length": float(length[i]),
            "d_residue_fraction": d_fractions[i],
        }
        for i in range(len(sequences))
    ]


class MBICPredictor:
    """Lazy-loaded SVR over ESM2 embeddings + modlAMP physchem descriptors +
    one-hot species. Returns pMBIC (log-scale potency), not a probability or
    raw concentration — see README.md for the inverse transform."""

    def __init__(self, model_dir: Path = MODEL_DIR, use_feature_cache: bool = False):
        self.model_dir = Path(model_dir)
        self.use_feature_cache = use_feature_cache
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        bundle = joblib.load(self.model_dir / "svr.joblib")
        self.model = bundle["model"]
        prep = bundle["preprocessing"]
        self.species_ohe = prep["species_ohe"]
        self.physchem_scaler = prep["physchem_scaler"]
        self.esm2_scaler = prep["esm2_scaler"]
        self.esm2_pca = prep["esm2_pca"]
        self.physchem_cols = prep["physchem_cols"]
        self.known_species = set(self.species_ohe.categories_[0])
        if not self.use_feature_cache:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
            self.esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(self.device).eval()
        self.loaded = True

    def predict_pmbic(
        self, sequence: str, species: str, feature_extractor: FeatureExtractor | None = None
    ) -> float:
        """Predict pMBIC = 6 - log10(activity_uM) for `sequence` against `species`.

        `species` should match a training-set organism name (see README.md /
        model_card.json for the vocabulary, e.g. "Pseudomonas aeruginosa").
        An unrecognized species is NOT an error: the underlying OneHotEncoder
        was fit with handle_unknown="ignore", so it silently becomes an
        all-zero species vector (equivalent to "no species signal") rather
        than raising or matching any specific organism. Treat predictions for
        out-of-vocabulary species with extra caution.
        """
        _validate_sequence(sequence)
        self._load()

        physchem = physchem_descriptors(sequence)
        physchem_row = pd.DataFrame([physchem])[self.physchem_cols].to_numpy(dtype=np.float32)
        physchem_scaled = self.physchem_scaler.transform(physchem_row)

        esm2_raw = (
            esm_embedding_cached(sequence, feature_extractor)
            if self.use_feature_cache
            else esm_embedding(sequence, self.tokenizer, self.esm_model, self.device)
        ).reshape(1, -1)
        esm2_scaled = self.esm2_scaler.transform(esm2_raw)
        esm2_pcs = self.esm2_pca.transform(esm2_scaled)

        species_oh = self.species_ohe.transform(pd.DataFrame([{"target_species": species}]))

        X = np.concatenate([species_oh, physchem_scaled, esm2_pcs], axis=1)
        return float(self.model.predict(X)[0])

    def predict_pmbic_batch(
        self,
        sequences: list[str],
        species: str,
        feature_extractor: FeatureExtractor | None = None,
    ) -> list[float]:
        """Batched predict_pmbic: one ESM2 forward pass for the whole batch,
        against one fixed `species` applied to every sequence (matches how
        Stage 6 scores per organism across all candidates)."""
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        if not sequences:
            return []
        self._load()

        physchem_rows = physchem_descriptors_batch(sequences)
        physchem_matrix = pd.DataFrame(physchem_rows)[self.physchem_cols].to_numpy(dtype=np.float32)
        physchem_scaled = self.physchem_scaler.transform(physchem_matrix)

        esm2_raw = (
            esm_embedding_cached_batch(sequences, feature_extractor)
            if self.use_feature_cache
            else esm_embedding_batch(sequences, self.tokenizer, self.esm_model, self.device)
        )
        esm2_scaled = self.esm2_scaler.transform(esm2_raw)
        esm2_pcs = self.esm2_pca.transform(esm2_scaled)

        species_oh = self.species_ohe.transform(
            pd.DataFrame([{"target_species": species}] * len(sequences))
        )

        X = np.concatenate([species_oh, physchem_scaled, esm2_pcs], axis=1)
        return [float(v) for v in self.model.predict(X)]
