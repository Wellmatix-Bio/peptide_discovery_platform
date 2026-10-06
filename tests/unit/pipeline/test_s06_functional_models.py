"""Stage 6's threshold filter, with the emphasis on predictions that were NOT produced.

Stage 6 is the stage that decides whether a candidate meets the functions the brief asked for. It
gets the honest-reporting rule RIGHT, and these tests exist to keep it that way: a prediction that
is missing, non-finite, out of its model's applicability domain, or carries a non-ok status is
recorded as `passed: None` and listed in `missing_required_predictions` -- never as a quiet pass
and never as a failure.

That distinction matters because stages 5 and 8 got it wrong in five places, each time by reading
an absent value as a pass. Stage 6's own design note explains the opposite trade it makes: a
missing prediction does not REJECT either, because one out-of-vocabulary pathogen like MRSA would
otherwise reject every candidate regardless of its real properties.

The consequence of that trade is pinned below, deliberately: a candidate with no usable evidence
at all survives stage 6, carrying `status: insufficient_evidence`. It is a defensible choice and
it is not an obvious one, so it is tested rather than left to be rediscovered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.s06_functional_models.stage import (  # noqa: E402
    STAGE6_THRESHOLDS,
    check_pathogen_vocabulary,
    filter_predictions,
    merge_stage6_thresholds,
)

#: Comfortably above every "min_" threshold, so only the field a test changes decides the outcome.
GOOD = {
    "amp_probability": 0.99,
    "proliferation_migration": {"migration": 0.99, "status": "ok"},
    "angiogenic_activity": {"angiogenic": 0.99, "status": "ok"},
    "anti_inflammatory_probability": 0.99,
}


#: MIC is reported as log10(uM) and compared after 10**value, against max_mic per pathogen.
#: 0.0 is 1 uM, far under the 16.0 default.
MIC_OK = {"mic": {"log_mic_um": {"Pseudomonas_aeruginosa": 0.0}, "status": "ok"}}


def run(predictions=None, functions=("anti_inflammatory",), pathogens=(), thresholds=None):
    return filter_predictions(
        {**GOOD, **MIC_OK, **(predictions or {})},
        list(functions),
        list(pathogens),
        thresholds or STAGE6_THRESHOLDS,
    )


def test_a_candidate_meeting_every_threshold_passes_and_claims_nothing_missing():
    got = run()
    assert got["status"] == "passed"
    assert got["passed"] is True
    assert got["missing_required_predictions"] == []
    assert got["failed_requirements"] == []


def test_a_value_below_its_threshold_fails_and_names_the_reason():
    got = run({"amp_probability": 0.10}, functions=("antimicrobial",),
              pathogens=("Pseudomonas_aeruginosa",))
    assert got["status"] == "rejected"
    assert got["passed"] is False
    assert "low_amp_probability" in got["failed_requirements"]


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), float("inf")],
    ids=["none", "nan", "inf"],
)
def test_an_unusable_value_is_missing_rather_than_passed_or_failed(value):
    """The rule stages 5 and 8 broke five times. None, NaN and infinity are all 'no answer'."""
    got = run({"anti_inflammatory_probability": value})
    key = "anti_inflammatory_probability"
    assert key in got["missing_required_predictions"]
    assert got["evaluated_predictions"][key]["passed"] is None
    assert got["failed_requirements"] == []
    assert got["status"] == "insufficient_evidence"


def test_a_prediction_outside_its_applicability_domain_is_not_usable():
    """A model that answered, but about a peptide it cannot speak for, has not answered."""
    got = run(
        {"angiogenic_activity": {"angiogenic": 0.99, "applicability_domain": False}},
        functions=("angiogenesis",),
    )
    assert "angiogenic_activity" in got["missing_required_predictions"]
    assert got["evaluated_predictions"]["angiogenic_activity"]["passed"] is None


def test_a_non_ok_status_is_not_usable_even_with_a_number_beside_it():
    got = run(
        {"angiogenic_activity": {"angiogenic": 0.99, "status": "skipped_too_short"}},
        functions=("angiogenesis",),
    )
    assert "angiogenic_activity" in got["missing_required_predictions"]


def test_a_missing_threshold_makes_the_check_unusable_rather_than_automatic():
    """With no threshold there is nothing to compare against, so there is no verdict to report."""
    got = run(thresholds={**STAGE6_THRESHOLDS, "min_anti_inflammatory_probability": None})
    key = "anti_inflammatory_probability"
    assert key in got["missing_required_predictions"]
    assert got["evaluated_predictions"][key]["passed"] is None


@pytest.mark.parametrize("bad", [-1.0, float("nan")], ids=["negative", "nan"])
def test_a_nonsensical_threshold_is_an_error_not_a_silent_skip(bad):
    with pytest.raises(ValueError, match="Invalid Stage 6 threshold"):
        run(thresholds={**STAGE6_THRESHOLDS, "min_anti_inflammatory_probability": bad})


def test_only_the_requested_functions_are_checked():
    """A low angiogenesis score must not reject a brief that never asked for angiogenesis."""
    got = run({"angiogenic_activity": {"angiogenic": 0.0, "status": "ok"}})
    assert got["status"] == "passed"
    assert "angiogenic_activity" not in got["evaluated_predictions"]


def test_a_candidate_with_no_usable_evidence_at_all_still_survives():
    """PINNED BECAUSE IT IS SURPRISING, not because it is obviously right.

    `passed` is derived from failures alone, so a candidate whose every required prediction is
    missing has nothing to fail and survives the stage. Stage 6's design note explains the trade:
    rejecting on missing evidence would mean one out-of-vocabulary pathogen rejects everything.
    The verdict still says `insufficient_evidence`, so a reader is not told the candidate passed
    its checks -- only that it was not rejected. If that trade is ever revisited, this test is
    where the decision is recorded.
    """
    got = run({"anti_inflammatory_probability": None})
    assert got["status"] == "insufficient_evidence"
    assert got["passed"] is True, "s06 filters on failures, not on evidence"
    assert got["missing_required_predictions"] == ["anti_inflammatory_probability"]


def test_a_pathogen_no_model_supports_is_named_up_front():
    unsupported = check_pathogen_vocabulary(["Staphylococcus_aureus", "Candida_albicans"])
    assert any("Candida_albicans" in values for values in unsupported.values())


def test_overriding_one_pathogens_threshold_keeps_the_others():
    """max_mic is a per-pathogen dict. Replacing it wholesale would silently drop every threshold
    the override did not mention -- which is a way to stop screening without noticing."""
    merged = merge_stage6_thresholds({"max_mic": {"MRSA": 4.0}})
    assert merged["max_mic"]["MRSA"] == 4.0
    assert merged["max_mic"]["Pseudomonas_aeruginosa"] == STAGE6_THRESHOLDS["max_mic"][
        "Pseudomonas_aeruginosa"
    ]


def test_a_flat_threshold_override_still_replaces_cleanly():
    merged = merge_stage6_thresholds({"min_amp_probability": 0.5})
    assert merged["min_amp_probability"] == 0.5
    assert merged["min_angiogenic_activity"] == STAGE6_THRESHOLDS["min_angiogenic_activity"]


def test_asking_for_antimicrobial_without_naming_a_pathogen_is_never_satisfiable():
    """Found while writing these tests, and surprising enough to pin.

    The MIC check is per-pathogen. A brief that asks for `antimicrobial` but lists no pathogens
    has nothing to check it against, so stage 6 records `mic:required_pathogens` as missing and
    the verdict is `insufficient_evidence` no matter how good the candidate is. The API's own
    schema does not require a pathogen alongside the antimicrobial function, so this is reachable
    from a valid request.
    """
    got = filter_predictions(
        {**GOOD, **MIC_OK}, ["antimicrobial"], [], STAGE6_THRESHOLDS
    )
    assert "mic:required_pathogens" in got["missing_required_predictions"]
    assert got["status"] == "insufficient_evidence"


def test_naming_a_pathogen_makes_the_antimicrobial_check_real():
    got = filter_predictions(
        {**GOOD, **MIC_OK}, ["antimicrobial"], ["Pseudomonas_aeruginosa"], STAGE6_THRESHOLDS
    )
    assert got["status"] == "passed"
    assert got["evaluated_predictions"]["mic:Pseudomonas_aeruginosa"]["passed"] is True


def test_a_mic_above_its_threshold_rejects():
    """log_mic_um 2.0 is 100 uM, over the 16.0 default for this pathogen."""
    got = filter_predictions(
        {**GOOD, "mic": {"log_mic_um": {"Pseudomonas_aeruginosa": 2.0}, "status": "ok"}},
        ["antimicrobial"],
        ["Pseudomonas_aeruginosa"],
        STAGE6_THRESHOLDS,
    )
    assert got["status"] == "rejected"
    assert "high_mic:Pseudomonas_aeruginosa" in got["failed_requirements"]


def test_a_pathogen_the_model_did_not_score_is_missing_not_passed():
    """The MRSA gap in mic_predictor_v1: a requested pathogen with no prediction must not be
    treated as having met its threshold."""
    got = filter_predictions(
        {**GOOD, **MIC_OK}, ["antimicrobial"], ["MRSA"], STAGE6_THRESHOLDS
    )
    assert "mic:MRSA" in got["missing_required_predictions"]
    assert got["evaluated_predictions"]["mic:MRSA"]["passed"] is None
    assert got["failed_requirements"] == []


def test_each_prediction_is_read_from_the_field_its_model_actually_writes():
    """A guard against the mistake these tests made first time round.

    The value for `angiogenic_activity` lives under `angiogenic`, not `score`. Reading the wrong
    key yields None, which stage 6 correctly reports as missing -- so a fixture with the wrong
    field name produces a test that passes while exercising nothing. It took a deliberate
    sabotage of the applicability-domain check, which the test did NOT catch, to notice.
    """
    usable = run(
        {"angiogenic_activity": {"angiogenic": 0.99, "status": "ok"}},
        functions=("angiogenesis",),
    )
    assert usable["evaluated_predictions"]["angiogenic_activity"]["value"] == 0.99
    assert usable["evaluated_predictions"]["angiogenic_activity"]["passed"] is True

    wrong_field = run(
        {"angiogenic_activity": {"score": 0.99, "status": "ok"}},
        functions=("angiogenesis",),
    )
    assert wrong_field["evaluated_predictions"]["angiogenic_activity"]["value"] is None
