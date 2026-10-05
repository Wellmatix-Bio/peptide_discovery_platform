"""Stage 8's safety verdict, for screens that DID NOT RUN.

`aggregation_tendency` was fixed here earlier -- a null score had read `else "pass"`, recording a
screen that never ran as one the candidate had passed. **The same mistake was left in place two
lines below, twice**, in `solubility` and `cleavage_stability`, both written as
`score is not None and <threshold>` with `else "pass"`.

`cleavage_stability` is the worse of the two: its threshold is a hard REJECT, so a candidate that
could not be screened was cleared on a check nothing performed.

The stage is built with `__new__` to skip `__init__`, which loads models; the verdict rollup is
pure and needs none of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.s08_safety_developability.stage import Stage8  # noqa: E402

#: A candidate every stage-8 threshold clears, so a non-pass below comes from the field under test.
CLEAN = {
    "hemolysis": {"phc50": 0.1},
    "cytotoxicity": {"score": 0.1},
    "aggregation_tendency": {"score": 0.1},
    "solubility": {"score": 0.9},
    "cleavage_stability": {"score": 0.9},
}

#: Every field whose score may legitimately be None: the model could not run on this peptide, or
#: the optional dependency behind it is absent.
NULLABLE = ["aggregation_tendency", "solubility", "cleavage_stability"]


def verdict(**overrides):
    return Stage8.__new__(Stage8).compute_safety_verdict({**CLEAN, **overrides}, {})


def test_a_fully_scored_candidate_passes_and_claims_nothing_unscreened():
    got = verdict()
    assert got["overall"] == "pass"
    assert "not_screened" not in got


@pytest.mark.parametrize("field", NULLABLE)
def test_a_null_score_is_not_screened_rather_than_a_pass(field: str):
    got = verdict(**{field: {"score": None}})
    assert got["properties"][field] == "not_screened"
    assert got["overall"] == "flag"
    assert got["not_screened"] == [field]
    assert "not the same as passed" in got["not_screened_note"]


def test_an_unrun_reject_check_does_not_clear_the_candidate():
    """cleavage_stability's threshold is a hard reject. Unrun must not read as cleared."""
    assert verdict(cleavage_stability={"score": None})["properties"]["cleavage_stability"] != "pass"


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("cleavage_stability", 0.0, "reject"),
        ("aggregation_tendency", 1.0, "reject"),
        ("solubility", 0.0, "flag"),
    ],
)
def test_a_real_score_past_its_threshold_still_decides(field: str, value: float, expected: str):
    """The null-handling must not have swallowed the thresholds it guards."""
    assert verdict(**{field: {"score": value}})["properties"][field] == expected


def test_a_reject_still_outranks_an_unscreened_check():
    got = verdict(cleavage_stability={"score": None}, cytotoxicity={"score": 1.0})
    assert got["overall"] == "reject"


def test_every_unscreened_check_is_listed():
    got = verdict(**{f: {"score": None} for f in NULLABLE})
    assert got["not_screened"] == sorted(NULLABLE)
    assert got["overall"] == "flag"
