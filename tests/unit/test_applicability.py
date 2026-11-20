# common.applicability: length range, per-length reference support, and the shipped cytokine profiles. Needs no ESM2.
import json
from pathlib import Path

import pytest

from common.applicability import MIN_EXAMPLES_PER_CLASS, NOT_CHECKED, ReferenceProfile, Status, domain_profile

STORE = Path(__file__).resolve().parents[2] / "model_store"


def profile(counts, low=8, high=20, minimum=10):
    info = {"training": {"length_min": low, "length_max": high}, "domain": {"min_examples_per_class": minimum, "length_counts": counts}}
    return ReferenceProfile.from_info(info)


COUNTS = {"8": {"pos": 2, "neg": 40}, "9": {"pos": 30, "neg": 30}, "10": {"pos": 10, "neg": 10}, "11": {"pos": 0, "neg": 0}, "12": {"pos": 50, "neg": 0}}


@pytest.mark.parametrize("length", [7, 21, 0])
def test_a_length_outside_the_trained_range_is_out_of_domain(length):
    result = profile(COUNTS).assess(length)
    assert result["status"] == Status.OUT_OF_DOMAIN and result["checks"]["length_range"] == "fail"


def test_a_well_supported_length_is_in_domain():
    result = profile(COUNTS).assess(9)
    assert result["status"] == Status.IN_DOMAIN
    assert result["reference"] == {"positives": 30, "negatives": 30, "min_per_class": 10}


def test_the_minimum_is_inclusive():
    assert profile(COUNTS).assess(10)["status"] == Status.IN_DOMAIN


@pytest.mark.parametrize("length", [8, 12])
def test_too_few_examples_of_either_class_is_limited_evidence(length):
    result = profile(COUNTS).assess(length)
    assert result["status"] == Status.LIMITED_EVIDENCE and result["checks"]["reference_support"] == "fail"
    assert "at least 10 of each" in result["reasons"][0]


def test_a_gap_inside_the_range_with_no_training_peptides_is_out_of_domain():
    result = profile(COUNTS).assess(11)
    assert result["status"] == Status.OUT_OF_DOMAIN and "no peptide of length 11" in result["reasons"][0]


def test_a_length_missing_from_the_counts_is_a_gap_too():
    assert profile(COUNTS).assess(15)["status"] == Status.OUT_OF_DOMAIN


def test_a_model_without_counts_is_never_called_in_domain():
    info = {"training": {"length_min": 8, "length_max": 20}}
    result = ReferenceProfile.from_info(info).assess(12)
    assert result["status"] == Status.LIMITED_EVIDENCE and result["checks"]["reference_support"] == "unknown"
    assert result["reference"] is None


def test_the_assessment_says_which_checks_it_did_not_run():
    assert profile(COUNTS).assess(9)["not_checked"] == list(NOT_CHECKED) == ["sequence_similarity", "embedding_distribution"]


def test_the_minimum_travels_with_the_profile():
    assert profile(COUNTS, minimum=40).assess(9)["status"] == Status.LIMITED_EVIDENCE


def test_domain_profile_counts_positives_and_negatives_per_length():
    block = domain_profile([8, 8, 8, 9], [1, 0, 0, 1])
    assert block["length_counts"] == {"8": {"pos": 1, "neg": 2}, "9": {"pos": 1, "neg": 0}}
    assert block["min_examples_per_class"] == MIN_EXAMPLES_PER_CLASS


# --- the shipped cytokine profiles (weights are gitignored, so these skip on a checkout without them) ---------------------------

CYTOKINES = ["tnf_alpha", "il6", "il10", "il4", "il13", "ifn_gamma"]


def shipped(cytokine):
    path = STORE / "model_weights" / f"{cytokine}_induction_predictor_v1" / "info.json"
    if not path.is_file():
        pytest.skip("model weights are not present")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("cytokine", CYTOKINES)
def test_every_shipped_profile_accounts_for_every_training_peptide(cytokine):
    info = shipped(cytokine)
    counts = info["domain"]["length_counts"]
    assert sum(c["pos"] + c["neg"] for c in counts.values()) == info["training"]["n"]
    assert sum(c["pos"] for c in counts.values()) / info["training"]["n"] == pytest.approx(info["training"]["positive_rate"], abs=1e-4)
    assert min(map(int, counts)) == info["training"]["length_min"] and max(map(int, counts)) == info["training"]["length_max"]


def test_alpmhir_is_judged_per_model_not_once():
    """The seven-residue peptide from the studio brief: below IL-6's and IFN-gamma's range, thinly covered by IL-4 and TNF-alpha."""
    status = {c: ReferenceProfile.from_info(shipped(c)).assess(7)["status"] for c in CYTOKINES}
    assert status["il6"] == status["ifn_gamma"] == status["il10"] == status["il13"] == Status.OUT_OF_DOMAIN
    assert status["il4"] == status["tnf_alpha"] == Status.LIMITED_EVIDENCE
    assert len(set(status.values())) > 1


def test_il6_has_no_negative_training_peptides_at_nine_or_ten_residues():
    result = ReferenceProfile.from_info(shipped("il6")).assess(9)
    assert result["status"] == Status.LIMITED_EVIDENCE and result["reference"]["negatives"] == 0


def test_status_ranges_group_contiguous_lengths_by_status():
    from common.applicability import ReferenceProfile

    counts = {n: (20, 20) for n in (8, 9, 10, 12)}
    counts[11] = (3, 40)
    profile = ReferenceProfile(8, 12, counts)
    assert profile.status_ranges(6, 14) == {"in_domain": [[8, 10], [12, 12]], "limited_evidence": [[11, 11]], "out_of_domain": [[6, 7], [13, 14]]}
