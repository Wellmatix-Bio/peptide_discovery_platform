"""Loader/inference wrapper for the receptor-peptide binding-affinity baseline (frozen ESM2-150M pooled vectors -> concat MLP -> pKd)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

RECEPTORS = ("GLP1R", "GCGR", "GIPR")
SPLITS = ("cluster", "random")
DEFAULT_SPLIT = "cluster"

EMBED_DIM = 640
HIDDEN = 256

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN = 1024  # the notebook truncates longer inputs; peptides here are far shorter


class ConcatMLPRegressor(nn.Module):
    """concat(pooled_target, pooled_binder) -> Linear -> ReLU -> Dropout -> Linear(1) = pKd; layer names match the saved state dicts."""

    def __init__(self, dim_a: int, dim_b: int, hidden: int = HIDDEN, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_a + dim_b, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, b], dim=-1)).squeeze(-1)


def _validate_sequence(sequence: str, what: str) -> None:
    if not sequence:
        raise ValueError(f"{what} must be non-empty.")
    bad = sorted(set(sequence) - AMINO_ACID_SET)
    if bad:
        raise ValueError(
            f"{what} contains residues outside the 20 uppercase standard amino acids: {bad}"
        )
    if len(sequence) > MAX_LEN - 2:
        raise ValueError(f"{what} is longer than the {MAX_LEN - 2}-residue encoder limit.")


def pooled_vectors(
    sequences: list[str], feature_extractor: FeatureExtractor
) -> torch.Tensor:
    """Mean over every attended token including BOS/EOS, this model's trained convention."""
    embeddings = feature_extractor.get_esm2_embedding_batch(sequences)
    return torch.from_numpy(
        np.stack([e.hidden_states.mean(axis=0) for e in embeddings]).astype(np.float32)
    )


class BindingAffinityPredictor:
    """Lazy-loaded frozen-encoder baseline returning predicted pKd = 9 - log10(Kd [nM]) (higher = tighter); see README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR, split: str = DEFAULT_SPLIT):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
        self.model_dir = Path(model_dir)
        self.split = split
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        with open(self.model_dir / "receptors.json", "r", encoding="utf-8") as f:
            self.receptors = {k: v["sequence"] for k, v in json.load(f).items()}
        checkpoint = torch.load(
            self.model_dir / f"baseline_split_{self.split}.pt", map_location="cpu"
        )
        self.model = ConcatMLPRegressor(EMBED_DIM, EMBED_DIM)
        self.model.load_state_dict(checkpoint["state"])
        self.model.eval()
        self.loaded = True

    @torch.no_grad()
    def predict_pkd_pairs(
        self,
        targets: list[str],
        binders: list[str],
        feature_extractor: FeatureExtractor,
    ) -> list[float]:
        """pKd for each (target, binder) pair; both sides are encoded with the same frozen ESM2 and pooled."""
        if len(targets) != len(binders):
            raise ValueError(f"got {len(targets)} targets for {len(binders)} binders.")
        if not binders:
            return []
        for i, (t, b) in enumerate(zip(targets, binders)):
            try:
                _validate_sequence(t, "target")
                _validate_sequence(b, "binder")
            except ValueError as exc:
                raise ValueError(f"pair at index {i} invalid: {exc}") from exc
        self._load()
        unique = list(dict.fromkeys(list(targets) + list(binders)))
        vectors = dict(zip(unique, pooled_vectors(unique, feature_extractor)))
        a = torch.stack([vectors[t] for t in targets])
        b = torch.stack([vectors[s] for s in binders])
        return [float(v) for v in self.model(a, b).numpy()]

    def predict_pkd_batch(
        self,
        peptides: list[str],
        feature_extractor: FeatureExtractor,
        receptor: str,
    ) -> list[float]:
        """pKd of each peptide against one of the human receptors GLP1R, GCGR, GIPR."""
        if receptor not in RECEPTORS:
            raise ValueError(f"receptor must be one of {RECEPTORS}, got {receptor!r}.")
        self._load()
        target = self.receptors[receptor]
        return self.predict_pkd_pairs([target] * len(peptides), peptides, feature_extractor)

    def predict_pkd_all_batch(
        self, peptides: list[str], feature_extractor: FeatureExtractor
    ) -> dict[str, list[float]]:
        return {
            receptor: self.predict_pkd_batch(peptides, feature_extractor, receptor)
            for receptor in RECEPTORS
        }

    def predict_pkd(
        self, peptide: str, feature_extractor: FeatureExtractor, receptor: str
    ) -> float:
        return self.predict_pkd_batch([peptide], feature_extractor, receptor)[0]
