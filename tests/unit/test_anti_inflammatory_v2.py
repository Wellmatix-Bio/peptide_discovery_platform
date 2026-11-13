# anti_inflammatory_predictor_v2: input handling, features, windows and architecture shapes; the weights test is skipped without them.
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from model_store.anti_inflammatory_predictor_v2 import AntiInflammatoryPredictor
from model_store.anti_inflammatory_predictor_v2 import predictor as p

WEIGHTS = p.MODEL_DIR / "weights.pt"


def test_handcrafted_features_have_626_finite_values():
    for seq in ("GIGKFLHSAKKF", "A", "KRKRKRKR", "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"):
        x = p.handcrafted_features(seq)
        assert x.shape == (626,) and np.isfinite(x).all()


def test_windows_are_terminal_then_cleavage_and_fit_the_slots():
    seq = "AKRAAKKAAARRAAAAGGGGKR"
    wins = p.get_windows(seq)
    assert wins[:2] == [seq[:10], seq[:20]] and wins[2:4] == [seq[-10:], seq[-20:]]
    assert len(wins) <= p.N_SLOTS and all(len(w) <= p.WIN_LEN for w in wins)
    assert len(p.get_windows("AAAAAA")) == 4  # no K/R: no cleavage windows


def test_invalid_input_is_rejected_and_range_is_flagged_not_rejected():
    predictor = AntiInflammatoryPredictor()
    for bad in ("", "AXA", "aaaa"):
        with pytest.raises(ValueError, match="index 0 invalid"):
            predictor.predict_proba_batch([bad], None)
    assert p.in_trained_length_range("A" * p.MIN_LENGTH) and not p.in_trained_length_range("A" * (p.MAX_LENGTH + 1))


def test_network_shapes_and_parameters_are_deterministic_in_eval():
    torch.manual_seed(0)
    net = p.FusionNet(d_hand=80).eval()
    B, L = 3, 12
    args = (torch.randn(B, 640), torch.randn(B, L, 640), torch.ones(B, L, dtype=torch.bool),
            torch.randn(B, p.N_SLOTS, p.WIN_LEN, 640), torch.ones(B, p.N_SLOTS, p.WIN_LEN, dtype=torch.bool),
            torch.ones(B, p.N_SLOTS, dtype=torch.bool), torch.randn(B, 80))
    with torch.no_grad():
        first, second = net(*args), net(*args)
    assert first.shape == (B,) and torch.equal(first, second)


class FakeExtractor:
    """Stands in for the shared extractor: BOS + residues + EOS rows of seeded noise per sequence."""

    def get_esm2_embedding_batch(self, sequences):
        out = []
        for s in sequences:
            rng = np.random.default_rng(sum(map(ord, s)))
            out.append(type("E", (), {"hidden_states": rng.standard_normal((len(s) + 2, 640)).astype(np.float32)})())
        return out


@pytest.mark.skipif(not WEIGHTS.exists(), reason="weights not present")
def test_loaded_model_scores_in_unit_interval_deterministically_and_flags_length():
    predictor = AntiInflammatoryPredictor(device="cpu")
    seqs = ["GIGKFLHSAKKF", "HAEGTFTSDVSSYLEGQAAK", "KRAAKKGG" * 5]
    a = predictor.predict_proba_batch(seqs[:2], FakeExtractor())
    b = predictor.predict_proba_batch(seqs[:2], FakeExtractor())
    assert a == b and all(0.0 <= v <= 1.0 for v in a)
    with pytest.warns(UserWarning, match="outside the trained length range"):
        predictor.predict_proba_batch(seqs, FakeExtractor())
    flagged = predictor.predict_batch(seqs, FakeExtractor())
    assert [r["in_length_range"] for r in flagged] == [True, True, False]
    # a sequence scores the same alone and inside a batch
    alone = predictor.predict_proba(seqs[1], FakeExtractor())
    assert alone == pytest.approx(a[1], abs=1e-5)
