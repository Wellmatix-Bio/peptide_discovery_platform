"""Loader/inference wrapper for the half-life XGBoost model (5 cluster-fold regressors on ESM2-150M + physchem + context)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

EMBED_DIM = 640

N_FOLDS = 5
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH, MAX_LENGTH = 5, 50

# Category levels in the order pandas assigned them at training time (sorted); XGBoost splits on the integer codes.
SPECIES_LEVELS = (
    "cat", "chicken", "cow", "dog", "human", "insect", "monkey", "mouse",
    "pig", "rabbit", "rat", "sheep", "unknown_or_other",
)  # fmt: skip
MATRIX_LEVELS = (
    "buffer_or_enzyme", "cell_based", "other_unspecified", "plasma", "serum", "whole_blood",
)  # fmt: skip
VIVO_VITRO_LEVELS = ("ex_vivo", "in_vitro", "in_vivo", "unknown")

MOD_FLAG_COLS = (
    "is_nterm_modified",
    "is_cterm_modified",
    "is_pegylated",
    "is_d_amino_acid",
    "is_cyclic",
)
EMBED_COLS = [f"esm_{i}" for i in range(EMBED_DIM)]
DESC_COLS = [
    "desc_length",
    "desc_net_charge",
    "desc_pI",
    "desc_hydrophobicity",
    "desc_hydrophobic_moment",
    "desc_aromatic_fraction",
    "desc_cationic_fraction",
]
ENV_COLS = ("species", "matrix", "vivo_vitro")


@dataclass(frozen=True)
class HalfLifeContext:
    """Measurement context and modification flags the model conditions on; defaults describe an unmodified peptide in human plasma in vitro."""

    species: str = "human"
    matrix: str = "plasma"
    vivo_vitro: str = "in_vitro"
    is_nterm_modified: bool = False
    is_cterm_modified: bool = False
    is_pegylated: bool = False
    is_d_amino_acid: bool = False
    is_cyclic: bool = False

    def validate(self) -> None:
        for field_name, levels in (
            ("species", SPECIES_LEVELS),
            ("matrix", MATRIX_LEVELS),
            ("vivo_vitro", VIVO_VITRO_LEVELS),
        ):
            if getattr(self, field_name) not in levels:
                raise ValueError(
                    f"{field_name}={getattr(self, field_name)!r} not seen in training; expected one of {levels}."
                )


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    bad = sorted(set(sequence) - STANDARD_AMINO_ACIDS)
    if bad:
        raise ValueError(
            f"sequence contains residues outside the 20 uppercase standard amino acids: {bad}"
        )
    if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
        raise ValueError(
            f"sequence length {len(sequence)} outside the trained range {MIN_LENGTH}-{MAX_LENGTH}."
        )


class HalfLifePredictor:
    """Lazy-loaded mean of five cluster-fold XGBoost regressors returning log10(half-life in seconds); see README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.fold_models = [
            joblib.load(self.model_dir / f"fold_{i}.joblib") for i in range(N_FOLDS)
        ]
        self.loaded = True

    def build_features(
        self,
        sequences: list[str],
        feature_extractor: FeatureExtractor,
        contexts: Sequence[HalfLifeContext],
    ) -> pd.DataFrame:
        """Feature frame in the exact column order and dtypes the fold models were trained on."""
        self._load()
        # Mean over every attended token including BOS/EOS, this model's trained convention.
        embeddings = feature_extractor.get_esm2_embedding_batch(sequences)
        embed = pd.DataFrame(
            np.stack([e.hidden_states.mean(axis=0) for e in embeddings]),
            columns=EMBED_COLS,
        )

        modlamp = feature_extractor.get_modlamp_descriptors_batch(sequences)
        desc = pd.DataFrame(
            {
                "desc_length": [d["modlamp_Length"] for d in modlamp],
                "desc_net_charge": [d["modlamp_Charge"] for d in modlamp],
                "desc_pI": [d["modlamp_pI"] for d in modlamp],
                "desc_hydrophobicity": [d["modlamp_HydrophRatio"] for d in modlamp],
                "desc_hydrophobic_moment": [
                    d["modlamp_eisenberg_moment"] for d in modlamp
                ],
                "desc_aromatic_fraction": [d["modlamp_Aromaticity"] for d in modlamp],
                "desc_cationic_fraction": [
                    sum(c in "KRH" for c in s) / len(s) for s in sequences
                ],
            }
        )[DESC_COLS]

        flags = pd.DataFrame(
            {c: [int(getattr(ctx, c)) for ctx in contexts] for c in MOD_FLAG_COLS}
        )
        frame = pd.concat([embed, desc, flags], axis=1)
        for col, levels in zip(
            ENV_COLS, (SPECIES_LEVELS, MATRIX_LEVELS, VIVO_VITRO_LEVELS)
        ):
            frame[col] = pd.Categorical(
                [getattr(ctx, col) for ctx in contexts], categories=list(levels)
            )
        return frame

    def predict_log10_half_life_batch(
        self,
        sequences: list[str],
        feature_extractor: FeatureExtractor,
        context: HalfLifeContext | Sequence[HalfLifeContext] | None = None,
    ) -> list[float]:
        """log10(half-life in seconds), one value per sequence; `context` is shared by the batch or given per sequence."""
        if not sequences:
            return []
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        if context is None:
            context = HalfLifeContext()
        contexts = (
            [context] * len(sequences)
            if isinstance(context, HalfLifeContext)
            else list(context)
        )
        if len(contexts) != len(sequences):
            raise ValueError(
                f"got {len(contexts)} contexts for {len(sequences)} sequences."
            )
        for ctx in contexts:
            ctx.validate()

        X = self.build_features(sequences, feature_extractor, contexts)
        predictions = np.mean([m.predict(X) for m in self.fold_models], axis=0)
        return [float(v) for v in predictions]

    def predict_half_life_seconds_batch(
        self,
        sequences: list[str],
        feature_extractor: FeatureExtractor,
        context: HalfLifeContext | Sequence[HalfLifeContext] | None = None,
    ) -> list[float]:
        return [
            float(10**v)
            for v in self.predict_log10_half_life_batch(
                sequences, feature_extractor, context
            )
        ]

    def predict_log10_half_life(
        self,
        sequence: str,
        feature_extractor: FeatureExtractor,
        context: HalfLifeContext | None = None,
    ) -> float:
        return self.predict_log10_half_life_batch(
            [sequence], feature_extractor, context
        )[0]
