# Unit tests for skin_sensitization_predictor_v1: the real weights and RDKit featurization, no embedder needed.

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from model_store.skin_sensitization_predictor_v1 import DECISION_THRESHOLD, SkinSensitizationPredictor
from model_store.skin_sensitization_predictor_v1.features import FEATURES, feature_vector, peptide_smiles
from model_store.skin_sensitization_predictor_v1.predictor import MEMBERS


@pytest.fixture(scope="module")
def predictor() -> SkinSensitizationPredictor:
    return SkinSensitizationPredictor()


def test_threshold_is_fixed() -> None:
    assert DECISION_THRESHOLD == 0.55
    assert SkinSensitizationPredictor.threshold == 0.55


def test_ensemble_is_the_mean_of_four_probabilities(predictor) -> None:
    results = predictor.predict_batch(["KTTKS", "GHK", "GKTTKS"])
    for r in results:
        assert set(r["members"]) == set(MEMBERS)
        assert all(0.0 <= p <= 1.0 for p in r["members"].values())
        assert r["p_sensitizer"] == pytest.approx(np.mean(list(r["members"].values())))


def test_batch_and_single_agree(predictor) -> None:
    seqs = ["KTTKS", "GHK"]
    assert [predictor.predict_proba(s) for s in seqs] == pytest.approx(predictor.predict_proba_batch(seqs))


def test_empty_batch(predictor) -> None:
    assert predictor.predict_batch([]) == []


@pytest.mark.parametrize("sequence", ["", "ACDEX", "acdef"])
def test_invalid_sequences_are_rejected(predictor, sequence) -> None:
    with pytest.raises(ValueError, match="index 0"):
        predictor.predict_batch([sequence])


def test_modifications_must_match_sequences(predictor) -> None:
    with pytest.raises(ValueError):
        predictor.predict_batch(["KTTKS"], [None, None])


def test_smiles_for_modifications() -> None:
    assert peptide_smiles("GK") == "NCC(=O)N[C@@H](CCCCN)C(=O)O"
    assert peptide_smiles("GK", {"n_acetyl": True, "c_amide": True}).startswith("CC(=O)N")
    assert peptide_smiles("GK", {"c_amide": True}).endswith("C(=O)N")
    assert peptide_smiles("GK", {"lipid_tail": {"position": "n_terminus"}}).startswith("CCCCCCCCCCCCCCCC(=O)N")
    on_lys = peptide_smiles("GK", {"lipid_tail": {"position": 1}})
    assert on_lys.startswith("NCC(=O)") and "NC(=O)CCCCCCCCCCCCCCC" in on_lys


def test_d_amino_acid_flips_stereo() -> None:
    assert peptide_smiles("AA", {"d_amino_acid": [0]}) != peptide_smiles("AA")


def test_feature_vector_has_the_trained_columns() -> None:
    f = feature_vector(peptide_smiles("KTTKS"))
    assert len(f) == len(FEATURES) == 14
    assert f[FEATURES.index("heavy_atoms")] > 0 and f[FEATURES.index("electrophilic_alerts")] == 0


def test_lipid_tail_changes_the_prediction(predictor) -> None:
    plain, lipid = predictor.predict_proba_batch(
        ["GKTTKS", "GKTTKS"], [None, {"lipid_tail": {"position": "n_terminus"}}]
    )
    assert plain != lipid
