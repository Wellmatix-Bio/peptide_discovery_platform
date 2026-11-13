# common.cytokine_ensemble and the six cytokine-induction packages: averaging, labels, applicability, and card/code agreement. Needs no real weights.
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common.cytokine_ensemble import CytokineEnsemble, ESM_DIM, _mlp_net

PACKAGES = {
    "ifn_gamma_induction_predictor_v1": ("IfnGammaInductionPredictor", ("MLP_lenbin",)),
    "tnf_alpha_induction_predictor_v1": ("TnfAlphaInductionPredictor", ("MLP_posw", "SVM_RBF_smote", "XGBoost_posw")),
    "il4_induction_predictor_v1": ("Il4InductionPredictor", ("XGBoost_posw",)),
    "il6_induction_predictor_v1": ("Il6InductionPredictor", ("XGBoost_lenbin", "MLP_adasyn", "SVM_RBF_posw")),
    "il10_induction_predictor_v1": ("Il10InductionPredictor", ("MLP_posw",)),
    "il13_induction_predictor_v1": ("Il13InductionPredictor", ("MLP_posw", "XGBoost_lenbin")),
}
STORE = Path(__file__).resolve().parents[2] / "model_store"


class FakeExtractor:
    """Every residue's hidden state is the sequence's first-letter code, so the pooled embedding is predictable."""

    def get_esm2_embedding_batch(self, sequences):
        out = []
        for s in sequences:
            row = np.full(ESM_DIM, float(ord(s[0]) - 64), dtype=np.float32)
            out.append(SimpleNamespace(hidden_states=np.vstack([np.zeros(ESM_DIM), np.tile(row, (len(s), 1)), np.zeros(ESM_DIM)])))
        return out


def logistic_member(coef_sign: float) -> dict:
    X = np.vstack([np.full((20, ESM_DIM), 1.0), np.full((20, ESM_DIM), 20.0)])
    y = np.array([0] * 20 + [1] * 20)
    scaler = StandardScaler().fit(X)
    est = LogisticRegression().fit(scaler.transform(X), y if coef_sign > 0 else 1 - y)
    return {"kind": "LR-L2", "scaler": scaler, "est": est, "state": None, "dim": ESM_DIM}


def mlp_member() -> dict:
    torch.manual_seed(0)
    X = np.random.RandomState(0).normal(size=(30, ESM_DIM))
    net = _mlp_net(ESM_DIM)
    return {"kind": "MLP", "scaler": StandardScaler().fit(X), "est": None, "state": net.state_dict(), "dim": ESM_DIM}


@pytest.fixture
def ensemble(tmp_path):
    joblib.dump(logistic_member(+1), tmp_path / "up.joblib")
    joblib.dump(logistic_member(-1), tmp_path / "down.joblib")
    joblib.dump(mlp_member(), tmp_path / "net.joblib")
    (tmp_path / "info.json").write_text(json.dumps({"training": {"length_min": 8, "length_max": 20}}))

    class Fake(CytokineEnsemble):
        MEMBERS = ("up", "down", "net")

    fake = Fake(tmp_path, tmp_path)
    return fake


def test_probability_is_the_mean_of_the_members(ensemble):
    result = ensemble.predict_batch(["AAAAAAAAAA", "TTTTTTTTTT"], FakeExtractor())
    for row in result:
        assert row["probability"] == pytest.approx(np.mean(list(row["members"].values())))
        assert set(row["members"]) == {"up", "down", "net"}


def test_label_uses_the_half_threshold(ensemble):
    result = ensemble.predict_batch(["AAAAAAAAAA", "TTTTTTTTTT"], FakeExtractor())
    assert all(r["label"] == int(r["probability"] >= 0.5) for r in result)
    assert result[0]["members"]["up"] < 0.5 < result[1]["members"]["up"]


def test_length_outside_the_trained_range_is_flagged_not_refused(ensemble):
    result = ensemble.predict_batch(["AAAAAA", "A" * 10, "A" * 40], FakeExtractor())
    assert [r["in_length_range"] for r in result] == [False, True, False]


def test_predict_proba_batch_matches_predict_batch(ensemble):
    seqs = ["AAAAAAAAAA", "TTTTTTTTTT"]
    assert ensemble.predict_proba_batch(seqs, FakeExtractor()) == [r["probability"] for r in ensemble.predict_batch(seqs, FakeExtractor())]


@pytest.mark.parametrize("bad", ["", "ACDEFX", "acdef"])
def test_invalid_sequences_are_rejected_with_their_index(ensemble, bad):
    with pytest.raises(ValueError, match="index 1"):
        ensemble.predict_proba_batch(["AAAAAAAAAA", bad], FakeExtractor())


def test_empty_batch_returns_empty_without_loading(ensemble):
    assert ensemble.predict_batch([], FakeExtractor()) == []
    assert not ensemble.loaded


@pytest.mark.parametrize("package", PACKAGES)
def test_package_members_and_card_name_match_class(package):
    class_name, members = PACKAGES[package]
    module = importlib.import_module(f"model_store.{package}")
    assert getattr(module, class_name).MEMBERS == members
    card = json.loads((STORE / package / "model_card.json").read_text(encoding="utf-8"))
    assert card["model_name"] == package
    assert card["version"] == "v1"


def test_predict_batch_attaches_an_applicability_assessment(ensemble):
    """The fixture's info.json has a length range but no per-length counts: in range is still not 'in domain'."""
    rows = ensemble.predict_batch(["A" * 5, "A" * 10], FakeExtractor())
    assert [r["applicability"]["status"] for r in rows] == ["out_of_domain", "limited_evidence"]
    assert rows[1]["applicability"]["not_checked"] == ["sequence_similarity", "embedding_distribution"]
