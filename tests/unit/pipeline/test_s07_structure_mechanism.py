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
