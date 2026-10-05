"""Stage 5's verdict rollup, with the emphasis on screens that DID NOT RUN.

The house rule is that a check which did not run is never reported as a pass (CONTRIBUTING.md).
Stage 5 broke it three ways at once, and all three are regression-tested here:

1. `secondary_structure` read `predictions[...]["flag"]` unconditionally. When `s4pred` is absent
   the screen returns `{"available": False, "reason": ...}` and carries no "flag" key, so this
   raised **KeyError('flag')** -- not a wrong verdict but a crash, on every candidate, in the
   default configuration, since `s4pred` does not resolve from a fresh clone.
2. `solubility` with a null score read as a pass.
3. `aggregation_tendency` with a null score read as a pass.

`tests/test_licensing.py` already asserted this rule for stage 8 -- but by matching SOURCE TEXT,
which is why it said nothing about any of the above. These tests call the code.

The stage is built with `__new__` to skip `__init__`, which loads models; the verdict rollup is
pure and needs none of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.s05_physchem_screening.stage import Stage5  # noqa: E402

#: A candidate on which every threshold passes, so anything that is not a pass below is caused by
#: the one field the test changes.
CLEAN = {
    "molecular_weight": 1600.0,
    "length": 16,
    "net_charge": 2,
    "isoelectric_point": 11.0,  # far from pH 7.4; a pI near it flags by design
    "hydrophobic_fraction": 0.3,
    "hydrophobic_moment": 0.2,
    "amphipathicity": 0.2,
    "solubility": {"score": 0.9, "status": "ok"},
    "aggregation_tendency": {"score": 0.1, "status": "ok"},
    "secondary_structure_consistency": {"available": True, "flag": False},
    "disulfide_complexity": {"category": "low"},
    "instability_index": 10.0,
    "oxidation_risk": {"risk_category": "low"},
    "deamidation_risk": {"risk_category": "low"},
}

#: What stage 5 returns when s4pred is not installed -- the real shape, taken from
#: compute_secondary_structure_consistency's early return, not invented here.
S4PRED_ABSENT = {"available": False, "reason": "s4pred is not installed, so the screen did not run"}


def verdict(**overrides):
    return Stage5.__new__(Stage5).compute_screening_verdict("GLFDIVKKVVGALGSL", {**CLEAN, **overrides}, {}, 50)


def test_a_fully_scored_candidate_passes_and_claims_nothing_unscreened():
    got = verdict()
    assert got["overall"] == "pass"
    assert "not_screened" not in got


def test_s4pred_absent_does_not_raise():
    """The reported bug: KeyError('flag'). A missing optional dependency must not take the stage
    down -- that is the whole point of making the import optional."""
    verdict(secondary_structure_consistency=S4PRED_ABSENT)


def test_s4pred_absent_is_recorded_as_not_screened_never_as_a_pass():
    got = verdict(secondary_structure_consistency=S4PRED_ABSENT)
    assert got["properties"]["secondary_structure"] == "not_screened"


def test_s4pred_absent_makes_the_overall_verdict_a_flag():
    """Not a pass, because nothing was measured. Not a reject either, because there is no finding
    to reject the candidate on."""
    assert verdict(secondary_structure_consistency=S4PRED_ABSENT)["overall"] == "flag"


def test_the_verdict_names_which_checks_did_not_run():
    got = verdict(secondary_structure_consistency=S4PRED_ABSENT)
    assert got["not_screened"] == ["secondary_structure"]
    assert "not the same as passed" in got["not_screened_note"]


@pytest.mark.parametrize("flag,expected", [(False, "pass"), (True, "flag")])
def test_s4pred_present_still_decides_as_before(flag: bool, expected: str):
    """The fix must not change the verdict when the screen DID run."""
    got = verdict(secondary_structure_consistency={"available": True, "flag": flag})
    assert got["properties"]["secondary_structure"] == expected


@pytest.mark.parametrize("field", ["solubility", "aggregation_tendency"])
def test_a_null_score_is_not_screened_rather_than_a_pass(field: str):
    """Both of these read `score is not None and <threshold>` with `else "pass"`, so a model that
    could not run on this peptide cleared it."""
    got = verdict(**{field: {"score": None, "status": "unavailable"}})
    assert got["properties"][field] == "not_screened"
    assert got["overall"] == "flag"
    assert field in got["not_screened"]


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("solubility", 0.0, "flag"),
        ("aggregation_tendency", 1.0, "flag"),
    ],
)
def test_a_real_score_past_its_threshold_still_flags(field: str, value: float, expected: str):
    """The null-handling must not have swallowed the thresholds it guards."""
    assert verdict(**{field: {"score": value}})["properties"][field] == expected


def test_several_unscreened_checks_are_all_listed():
    got = verdict(
        secondary_structure_consistency=S4PRED_ABSENT,
        solubility={"score": None},
        aggregation_tendency={"score": None},
    )
    assert got["not_screened"] == ["aggregation_tendency", "secondary_structure", "solubility"]
    assert got["overall"] == "flag"


def test_a_reject_still_outranks_an_unscreened_check():
    """Ordering matters: an unrun screen must not downgrade a genuine reject to a flag."""
    got = verdict(secondary_structure_consistency=S4PRED_ABSENT, molecular_weight=1e9)
    assert got["overall"] == "reject"
