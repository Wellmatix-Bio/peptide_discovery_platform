"""Loader/inference wrapper for the functional-agonism potency ensemble (Ridge + RF + XGBoost + MLP on ESM2-150M embeddings)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import joblib
import numpy as np

from common.model_sync import sync_model_weights, weights_dir_for

# TYPE-ONLY, and it has to be: importing it for real runs pipeline/__init__.py, which eagerly
# imports every stage, one of which (s06's metabolic stage) imports this package back. Importing
# model_store.functional_potency_predictor_v1 BEFORE pipeline therefore failed outright with a
# partially-initialized-module ImportError. `from __future__ import annotations` above makes
# every annotation a string, so nothing below needs the real class at run time.
if TYPE_CHECKING:
    from pipeline.feature_extractor import FeatureExtractor

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

RECEPTORS = ("GCGR", "GLP1R", "GIPR")

#: Receptors this ensemble predicts but CANNOT RANK, from the leave-one-cluster-out metrics in
#: model_card.json. GIPR scores R2 -0.108 and Pearson 0.094 on n=55 -- a negative R2 means the
#: model does worse than always predicting the training mean, so its ordering of two candidates
#: carries no information. Trained on 55 labels against 271 for GLP1R.
#:
#: The prediction is still returned: suppressing it would silently change the response shape and
#: an unreliable number a caller knows about is more useful than a missing one they must guess
#: at. Everything downstream must say so -- Stage 6 records this list on each candidate and the
#: API returns it per candidate, the same way a screen that did not run reports `not_screened`
#: rather than passing quietly.
#:
#: This is the single definition. pipeline/s04_generation/routeA_metabolic.py imports it to keep
#: GIPR out of the GA's default fitness, which was the first and for a while the only place that
#: acted on it.
UNRELIABLE_POTENCY_RECEPTORS = ("GIPR",)

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
