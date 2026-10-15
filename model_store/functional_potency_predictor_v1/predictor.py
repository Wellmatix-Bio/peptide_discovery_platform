"""Loader/inference wrapper for the functional-agonism potency ensemble (Ridge + RF + XGBoost + MLP on ESM2-150M embeddings)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from common.model_sync import sync_model_weights, weights_dir_for
from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

RECEPTORS = ("GCGR", "GLP1R", "GIPR")

# The four members whose unweighted mean was evaluated in the training notebook (its label omits RF; the code includes it).
ALL_MEMBERS = ("ridge", "rf", "xgb", "mlp")
DEFAULT_MEMBERS = ALL_MEMBERS

# Tokeniser and canonical mapping from the training notebook; the models saw plain_sequence(seq).
TOKEN_RE = re.compile(r"\[[^\]]*\]|[A-Za-z]")
LIPID_RE = re.compile(r"([A-Z])(\(.*\))")
PLAIN_MAP = {"Aib": "A", "Sar": "G", "Nle": "L", "dNle": "L"}

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def tokenize(seq: str) -> list[str]:
    seq = seq.strip()
    tokens = TOKEN_RE.findall(seq)
    if "".join(tokens) != seq:
        raise ValueError(f"Cannot tokenise sequence {seq!r}")
    return tokens


def plain_sequence(seq: str) -> str:
    """Canonical one-letter version of a sequence ('X' for anything unrecognised)."""
    out = []
    for t in tokenize(seq):
        inner = t[1:-1] if t.startswith("[") else t
        lipid = LIPID_RE.fullmatch(inner)
        out.append(
            lipid.group(1)
            if lipid
            else PLAIN_MAP.get(inner, inner.upper() if len(inner) == 1 else "X")
        )
    return "".join(out)


def _canonical(sequence: str) -> str:
    plain = plain_sequence(sequence)
    unknown = sorted(set(plain) - STANDARD_AMINO_ACIDS)
    if not plain or unknown:
        raise ValueError(
            f"sequence {sequence!r} has residues the model cannot embed: {unknown or 'empty'}"
        )
    return plain


def esm_mean_embeddings(
    plain_sequences: list[str], feature_extractor: FeatureExtractor
) -> np.ndarray:
    """Mean over residue positions of the last hidden layer with BOS/EOS excluded: this model's trained convention."""
    embeddings = feature_extractor.get_esm2_embedding_batch(plain_sequences)
    return np.stack([e.hidden_states[1:-1].mean(axis=0) for e in embeddings])


class FunctionalPotencyPredictor:
    """Lazy-loaded per-receptor ensemble returning predicted pEC50 (higher = more potent); the mean of `members`, see README.md."""

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        members: Sequence[str] = DEFAULT_MEMBERS,
    ):
        unknown = [m for m in members if m not in ALL_MEMBERS]
        if unknown or not members:
            raise ValueError(f"members must be a non-empty subset of {ALL_MEMBERS}, got {tuple(members)}.")
        self.model_dir = Path(model_dir)
        self.members = tuple(members)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.models = {
            (member, receptor): joblib.load(self.model_dir / member / f"{receptor}.joblib")
            for member in self.members
            for receptor in RECEPTORS
        }
        with open(self.model_dir / self.members[0] / "GLP1R.json", "r", encoding="utf-8") as f:
            self.embedding_dim = int(json.load(f)["embedding_dim"])
        self.loaded = True

    def predict_pec50_all_batch(
        self, sequences: list[str], feature_extractor: FeatureExtractor
    ) -> dict[str, list[float]]:
        """Ensemble pEC50 for every receptor from one shared ESM2 pass."""
        if not sequences:
            return {receptor: [] for receptor in RECEPTORS}
        plain = []
        for i, sequence in enumerate(sequences):
            try:
                plain.append(_canonical(sequence))
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        self._load()
        X = esm_mean_embeddings(plain, feature_extractor)
        if X.shape[1] != self.embedding_dim:
            raise ValueError(
                f"ESM2 embedding dim {X.shape[1]} != trained dim {self.embedding_dim}."
            )
        return {
            receptor: [
                float(v)
                for v in np.mean(
                    [self.models[(m, receptor)].predict(X) for m in self.members], axis=0
                )
            ]
            for receptor in RECEPTORS
        }

    def predict_pec50_batch(
        self,
        sequences: list[str],
        feature_extractor: FeatureExtractor,
        receptor: str,
    ) -> list[float]:
        if receptor not in RECEPTORS:
            raise ValueError(f"receptor must be one of {RECEPTORS}, got {receptor!r}.")
        return self.predict_pec50_all_batch(sequences, feature_extractor)[receptor]

    def predict_pec50(
        self, sequence: str, feature_extractor: FeatureExtractor, receptor: str
    ) -> float:
        return self.predict_pec50_batch([sequence], feature_extractor, receptor)[0]
