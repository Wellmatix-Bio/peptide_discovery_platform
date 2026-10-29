"""Loader/inference wrapper for the antioxidant-peptide CNN + BiLSTM + Transformer average-logit ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common.model_sync import sync_model_weights, weights_dir_for

from .architecture import CNN, BiLSTM, TransformerNet

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Padded length = longest peptide in the training sheet.
MAX_LEN = 68
MIN_LENGTH = 1
EPS = 1e-7

BASE_MODELS = ("cnn", "bilstm", "transformer")


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    bad = sorted(set(sequence) - set(AMINO_ACIDS))
    if bad:
        raise ValueError(
            f"sequence contains residues outside the 20 uppercase standard amino acids: {bad}"
        )
    if not MIN_LENGTH <= len(sequence) <= MAX_LEN:
        raise ValueError(
            f"sequence length {len(sequence)} outside the trained range {MIN_LENGTH}-{MAX_LEN}."
        )


def one_hot(sequences: list[str], max_len: int = MAX_LEN) -> torch.Tensor:
    """(N, max_len, 20) float32, all-zero rows are padding."""
    X = np.zeros((len(sequences), max_len, len(AMINO_ACIDS)), dtype=np.float32)
    for i, s in enumerate(sequences):
        for j, aa in enumerate(s[:max_len]):
            X[i, j, AA_TO_IDX[aa]] = 1.0
    return torch.from_numpy(X)


class AntioxidantPredictor:
    """Lazy-loaded average-logit ensemble returning an antioxidant probability, see README.md."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        builders = {
            "cnn": lambda: CNN(len(AMINO_ACIDS), MAX_LEN),
            "bilstm": lambda: BiLSTM(len(AMINO_ACIDS)),
            "transformer": lambda: TransformerNet(len(AMINO_ACIDS), MAX_LEN),
        }
        self.models = {}
        for name in BASE_MODELS:
            model = builders[name]()
            state = torch.load(
                self.model_dir / f"{name}.pt", map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)
            self.models[name] = model.to(self.device).eval()
        self.loaded = True

    def predict_proba_batch(self, sequences: list[str], batch_size: int = 256) -> list[float]:
        """Sigmoid of the mean base-model logit, one value per sequence."""
        if not sequences:
            return []
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        self._load()
        X = one_hot(sequences)
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = X[start : start + batch_size].to(self.device)
                logits = []
                for model in self.models.values():
                    # Round-trip through clipped probabilities to match how the ensemble was scored.
                    p = torch.sigmoid(model(xb)).clamp(EPS, 1 - EPS)
                    logits.append(torch.log(p / (1 - p)))
                z = torch.stack(logits).mean(0)
                out.extend(float(v) for v in torch.sigmoid(z).cpu())
        return out

    def predict_proba(self, sequence: str) -> float:
        return self.predict_proba_batch([sequence])[0]
