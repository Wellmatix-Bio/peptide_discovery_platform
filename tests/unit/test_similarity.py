# Sequence similarity to a model's reference set, and how it changes the applicability assessment.
import gzip
import json

import numpy as np
import pytest

from common.applicability import ReferenceProfile, Status, with_similarity
from common.similarity import REFERENCE_FILE, SimilarityIndex, best_identities

REFERENCE = ["ALPMHIRKLW", "GGGGGSSSSS", "KKKRRRWWWFFF", "ACDEFGHIKL"]


def test_a_peptide_in_the_reference_set_is_identical_to_itself():
    assert best_identities(["ALPMHIRKLW"], REFERENCE)[0] == 1.0


def test_a_peptide_inside_a_longer_reference_peptide_is_fully_matched():
    assert best_identities(["PMHIRK"], REFERENCE)[0] == 1.0


def test_an_unrelated_peptide_scores_lower_than_a_near_copy():
    near = best_identities(["ALPMHIRKLV"], REFERENCE)[0]
    far = best_identities(["DEDEDEQNQN"], REFERENCE)[0]
    assert near >= 0.9 and far < 0.5


def test_excluded_reference_peptides_are_not_matched():
    exclude = [np.array([True, False, False, False])]
    assert best_identities(["ALPMHIRKLW"], REFERENCE, exclude=exclude)[0] < 1.0


def write_weights(tmp_path, similar_at=0.7, distant_below=0.4):
    with gzip.open(tmp_path / REFERENCE_FILE, "wt") as f:
        f.write("\n".join(REFERENCE))
    return {"similarity": {"similar_at": similar_at, "distant_below": distant_below}}


def test_the_index_names_a_band_for_each_peptide(tmp_path):
    index = SimilarityIndex.load(tmp_path, write_weights(tmp_path))
    rows = index.assess(["ALPMHIRKLW", "DEDEDEQNQN"])
    assert [r["verdict"] for r in rows] == ["similar", "distant"]
    assert rows[0]["best_identity"] == 1.0 and rows[0]["similar_at"] == 0.7


def test_weights_without_a_profile_or_reference_file_have_no_index(tmp_path):
    assert SimilarityIndex.load(tmp_path, {}) is None
    assert SimilarityIndex.load(tmp_path, {"similarity": {"similar_at": 0.7, "distant_below": 0.4}}) is None  # file missing


PROFILE = ReferenceProfile(8, 20, {n: (20, 20) for n in range(8, 21)})


def sim(verdict, identity):
    return {"best_identity": identity, "verdict": verdict, "similar_at": 0.67, "distant_below": 0.4}


def test_a_distant_peptide_is_downgraded_to_limited_evidence_with_the_reason():
    base = PROFILE.assess(12)
    assert base["status"] == Status.IN_DOMAIN
    result = with_similarity(base, sim("distant", 0.31))
    assert result["status"] == Status.LIMITED_EVIDENCE
    assert result["checks"]["sequence_similarity"] == "fail"
    assert "31%" in result["reasons"][-1] and "distant" in result["reasons"][-1]
    assert result["not_checked"] == ["embedding_distribution"]
    assert result["similarity"]["verdict"] == "distant"


@pytest.mark.parametrize("verdict,identity,check", [("similar", 0.9, "pass"), ("moderate", 0.5, "weak")])
def test_similar_and_moderate_peptides_keep_their_status(verdict, identity, check):
    result = with_similarity(PROFILE.assess(12), sim(verdict, identity))
    assert result["status"] == Status.IN_DOMAIN and result["checks"]["sequence_similarity"] == check


def test_similarity_never_raises_a_status():
    out = with_similarity(PROFILE.assess(30), sim("similar", 1.0))  # length out of range
    assert out["status"] == Status.OUT_OF_DOMAIN


def test_the_original_assessment_is_not_mutated():
    base = PROFILE.assess(12)
    with_similarity(base, sim("distant", 0.2))
    assert "similarity" not in base and base["not_checked"] == ["sequence_similarity", "embedding_distribution"]


def test_shipped_il10_weights_describe_a_calibrated_profile():
    from pathlib import Path

    from common.model_sync import weights_dir_for

    info_path = weights_dir_for(Path(__file__).resolve().parents[2] / "model_store" / "il10_induction_predictor_v1") / "info.json"
    if not (info_path.is_file() and (info_path.parent / REFERENCE_FILE).is_file()):
        pytest.skip("weights not present")
    info = json.loads(info_path.read_text())
    block = info["similarity"]
    assert 0 < block["distant_below"] < block["similar_at"] < 1 and block["n_reference"] == info["training"]["n"]
    index = SimilarityIndex.load(info_path.parent, info)
    assert index.assess([index.reference[0]])[0]["verdict"] == "similar"
