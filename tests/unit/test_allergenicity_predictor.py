# Unit tests for allergenicity_predictor_v1: the real fold weights with a fake embedder, so no ESM is loaded.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from model_store.allergenicity_predictor_v1 import AllergenicityPredictor
from model_store.allergenicity_predictor_v1.predictor import MAX_LENGTH, MODEL_DIR, TRAINED_MIN_LENGTH

pytestmark = pytest.mark.skipif(not (MODEL_DIR / "ensemble.json").exists(), reason="allergenicity weights not present locally")


class FakeExtractor:
    """Deterministic per-sequence embeddings: BOS + residues + EOS rows of 640 numbers."""

    def get_esm2_embedding_batch(self, sequences):
        out = []
        for s in sequences:
            rng = np.random.default_rng(sum(map(ord, s)))
            out.append(SimpleNamespace(hidden_states=rng.normal(size=(len(s) + 2, 640)).astype("float32")))
        return out


@pytest.fixture(scope="module")
def predictor() -> AllergenicityPredictor:
    return AllergenicityPredictor()


SEQS = ["GHKASSGHKA", "KTTKS", "ACDEFGHIKLMNPQRSTVWY" * 2]


def test_probability_is_the_sigmoid_of_the_score(predictor) -> None:
    for r in predictor.predict_batch(SEQS, FakeExtractor()):
        assert 0.0 <= r["p_allergen"] <= 1.0
        assert r["p_allergen"] == pytest.approx(1 / (1 + np.exp(-r["score"])))


def test_batch_and_single_predictions_agree(predictor) -> None:
    fe = FakeExtractor()
    batch = predictor.predict_proba_batch(SEQS, fe)
    assert [predictor.predict_proba(s, fe) for s in SEQS] == pytest.approx(batch, abs=1e-5)


def test_empty_batch_returns_empty(predictor) -> None:
    assert predictor.predict_batch([], FakeExtractor()) == []


@pytest.mark.parametrize("sequence", ["", "ACDEX", "acdef", "AC", "A" * (MAX_LENGTH + 1)])
def test_invalid_sequences_are_rejected(predictor, sequence: str) -> None:
    with pytest.raises(ValueError, match="index 0"):
        predictor.predict_batch([sequence], FakeExtractor())


def test_trained_length_range_is_enforced() -> None:
    assert (TRAINED_MIN_LENGTH, MAX_LENGTH) == (3, 49)
