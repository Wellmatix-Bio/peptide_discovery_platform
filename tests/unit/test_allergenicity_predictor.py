# Unit tests for allergenicity_predictor_v1: the real SVM/LDA/MLP weights with a fake embedder, so no ESM is loaded.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from model_store.allergenicity_predictor_v1 import AllergenicityPredictor
from model_store.allergenicity_predictor_v1.predictor import MAX_LENGTH, MEMBERS, TRAINED_MIN_LENGTH


class FakeExtractor:
    """Deterministic per-sequence embeddings: BOS + residues + EOS rows of 640 numbers."""

    def get_esm2_embedding_batch(self, sequences):
        out = []
        for s in sequences:
            rng = np.random.default_rng(abs(hash(s)) % (2**32))
            out.append(SimpleNamespace(hidden_states=rng.normal(size=(len(s) + 2, 640)).astype("float32")))
        return out


@pytest.fixture(scope="module")
def predictor() -> AllergenicityPredictor:
    return AllergenicityPredictor()


SEQS = ["ACDEFGHIKLMNPQRSTVWY" * 3, "GHKASGHKASGHKASGHKAS" * 3, "MKKLLPTAAAGLLLLAAQPAMA" * 4]


def test_members_are_probabilities_and_the_ensemble_is_their_mean(predictor) -> None:
    results = predictor.predict_batch(SEQS, FakeExtractor())
    assert len(results) == len(SEQS)
    for r in results:
        assert set(r["members"]) == set(MEMBERS)
        assert all(0.0 <= p <= 1.0 for p in r["members"].values())
        assert r["p_allergen"] == pytest.approx(np.mean(list(r["members"].values())))


def test_batch_and_single_predictions_agree(predictor) -> None:
    fe = FakeExtractor()
    batch = predictor.predict_proba_batch(SEQS, fe)
    assert [predictor.predict_proba(s, fe) for s in SEQS] == pytest.approx(batch)


def test_empty_batch_returns_empty(predictor) -> None:
    assert predictor.predict_batch([], FakeExtractor()) == []


@pytest.mark.parametrize("sequence", ["", "ACDEX", "acdef", "A" * (MAX_LENGTH + 1)])
def test_invalid_sequences_are_rejected(predictor, sequence: str) -> None:
    with pytest.raises(ValueError, match="index 0"):
        predictor.predict_batch([sequence], FakeExtractor())


def test_trained_length_range_is_exposed_but_not_enforced(predictor) -> None:
    assert TRAINED_MIN_LENGTH == 50
    assert len(predictor.predict_proba_batch(["ACDEF"], FakeExtractor())) == 1
