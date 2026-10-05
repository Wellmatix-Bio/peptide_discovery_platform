# Unit tests for s07_structure_mechanism.
from __future__ import annotations

import pytest

from pipeline.s07_structure_mechanism.stage import DEFAULT_THRESHOLDS, Stage7

STRUCTURE = {"confidence": {"mean_plddt": 0.8}, "secondary_structure": {"dominant_class": "helix"}}


def summarize(probabilities, **overrides):
    involvement = {label: {"probability": p} for label, p in probabilities.items()}
    return Stage7().build_mechanistic_evidence_summary(
        STRUCTURE, involvement, {**DEFAULT_THRESHOLDS, **overrides}
    )


def test_pathways_split_into_activated_and_inhibited():
    summary = summarize({"NF_KB": 0.1, "TGFB_SMAD": 0.9, "MAPK": 0.7})
    assert summary["activated_pathways"] == ["TGFB_SMAD", "MAPK"]
    assert summary["inhibited_pathways"] == ["NF_KB"]
    assert "engaged_pathways" not in summary


def test_the_cutoff_opens_an_uncertain_band():
    summary = summarize({"NF_KB": 0.4, "MAPK": 0.6, "ERK": 0.9, "WNT_BCATENIN": 0.1},
                        pathway_engagement_min_probability=0.75)
    assert summary["activated_pathways"] == ["ERK"]
    assert summary["inhibited_pathways"] == ["WNT_BCATENIN"]


def test_a_pathway_is_never_both_activated_and_inhibited():
    summary = summarize({"NF_KB": 0.5})
    assert summary["activated_pathways"] == ["NF_KB"]
    assert summary["inhibited_pathways"] == []


def test_functions_come_from_activated_pathways_only():
    summary = summarize({"TGFB_SMAD": 0.05, "VEGF_ANGIOGENESIS": 0.95})
    assert summary["functions_supported"] == ["angiogenesis"]


def test_summary_text_names_both_directions():
    text = summarize({"NF_KB": 0.1, "TGFB_SMAD": 0.9})["summary"]
    assert "activated: TGFB_SMAD" in text and "inhibited: NF_KB" in text


# ----------------------------------------------------------------------
# Pathway direction reporting.
#
# The v2 predictor's probability is P(ACTIVATOR), not P(involved): its training set dropped the
# label-0 rows, so the model cannot say "this peptide does not touch this pathway"
# (model_store/pathway_mapping_predictor_v2/README.md). Every label gets a direction, and
# confidence is the only honest gate on it.
#
# A label between the cutoffs used to fall out of both lists and vanish from the output, which a
# reader takes as "not relevant" when it means "the model could not call it".
# ----------------------------------------------------------------------

STRUCTURE = {
    "confidence": {"mean_plddt": 80.0},
    "secondary_structure": {"dominant_class": "H"},
}


def _summary(probabilities: dict, cutoff: float) -> dict:
    stage = Stage7.__new__(Stage7)
    return stage.build_mechanistic_evidence_summary(
        STRUCTURE,
        {label: {"probability": p} for label, p in probabilities.items()},
        {"pathway_engagement_min_probability": cutoff},
    )


def test_every_pathway_is_accounted_for_in_exactly_one_list():
    """Nothing may silently disappear: each label is activated, inhibited or undetermined."""
    probabilities = {"A": 0.99, "B": 0.72, "C": 0.51, "D": 0.50, "E": 0.28, "F": 0.01}
    got = _summary(probabilities, 0.7)
    buckets = (
        got["activated_pathways"] + got["inhibited_pathways"] + got["undetermined_pathways"]
    )
    assert sorted(buckets) == sorted(probabilities)
    assert len(buckets) == len(set(buckets)), "a pathway appears in more than one list"


def test_a_pathway_the_model_could_not_call_is_named():
    got = _summary({"A": 0.99, "C": 0.51}, 0.7)
    assert got["undetermined_pathways"] == ["C"]
    assert "C" not in got["activated_pathways"]
    assert "C" not in got["inhibited_pathways"]


def test_the_summary_text_states_what_could_not_be_called():
    got = _summary({"A": 0.99, "C": 0.51}, 0.7)
    assert "could not call" in got["summary"]
    assert "C" in got["summary"].split("could not call")[1]


def test_an_undetermined_pathway_does_not_support_a_function():
    """functions_supported is derived from activated_pathways only. A label the model could not
    call must not count as evidence for a desired function."""
    got = _summary({"A": 0.51}, 0.7)
    assert got["functions_supported"] == []


def test_confident_calls_are_unaffected():
    got = _summary({"A": 0.99, "F": 0.01}, 0.7)
    assert got["activated_pathways"] == ["A"]
    assert got["inhibited_pathways"] == ["F"]
    assert got["undetermined_pathways"] == []


def test_at_the_default_cutoff_nothing_is_undetermined():
    """Documents a real consequence rather than asserting it is desirable: with the shipped
    default of 0.5 the two bands meet, so EVERY pathway is called with a direction and a 0.501
    coin-flip is reported as definitely activating. See docs/BASELINE.md."""
    got = _summary({"A": 0.51, "B": 0.50, "C": 0.49}, 0.5)
    assert got["undetermined_pathways"] == []
    assert len(got["activated_pathways"]) + len(got["inhibited_pathways"]) == 3
