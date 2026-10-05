"""The history recorder's shape matchers, checked against CAPTURED API responses.

Why this file exists, and not more tests against the fake upstream in test_proxy_peptide.py:

That fake returned `ranked_candidates` as a LIST. The real API returns it as an integer count and
puts the list in `candidates`. is_job_results() required a list, so it matched no real response --
results reads were never recorded, no envelope was ever stored, and every history row kept its
submission summary forever. The whole accounts suite passed throughout, because the fake encoded
the same misunderstanding as the code it was meant to check.

A fake can agree with a bug. A captured response cannot. These read the files in
web/src/test/fixtures/, which were taken from a running deployment
(see that directory's README.md), so the only way to make them pass is to match what the API
actually sends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from accounts.history import _results_summary, _status_summary, is_created_job, is_job_results, is_job_status

FIXTURES = Path(__file__).resolve().parents[3] / "web/src/test/fixtures"
RESULTS = sorted(FIXTURES.glob("results-*.json"))
STATUSES = sorted(FIXTURES.glob("status-*.json"))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_captured_fixtures_are_present():
    """Guards against this file passing vacuously if the fixtures move or are deleted."""
    assert len(RESULTS) >= 2, f"expected captured results fixtures in {FIXTURES}"
    assert len(STATUSES) >= 2, f"expected captured status fixtures in {FIXTURES}"


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_a_real_results_response_is_recognised(path: Path):
    body = load(path)
    assert is_job_results(body), (
        "is_job_results() does not match a response captured from the running API; results reads"
        " would not be recorded at all"
    )
    assert not is_job_status(body), "a results response must not also match the status envelope"
    assert not is_created_job(body), "a results response must not also match the create envelope"


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.name)
def test_a_real_results_response_summarises_without_raising(path: Path):
    """The counts are the API's own fields, reported as given rather than recounted from the list.

    A results response for a run still in progress carries NULL counts -- results-2046288866976989184
    is a real one, captured from a run whose results.json was overwritten mid-flight. The summary
    must say it does not know, not invent a number and not print the word "None".
    """
    body = load(path)
    summary = _results_summary(body)
    assert "final candidate(s)" in summary
    for field in ("n_final", "ranked_candidates"):
        value = body[field]
        if value is None:
            assert "None" not in summary, f"a null {field} leaked Python's None into the summary"
        else:
            assert str(value) in summary, f"the API's own {field} is not reported"


def test_a_null_count_is_shown_as_unknown_rather_than_guessed():
    """Belt and braces for the above, independent of which fixtures happen to exist."""
    summary = _results_summary({"run_id": "r", "candidates": [], "n_final": None,
                                "ranked_candidates": None})
    assert "?" in summary
    assert "None" not in summary and "0" not in summary


@pytest.mark.parametrize("path", STATUSES, ids=lambda p: p.name)
def test_a_real_status_response_is_recognised(path: Path):
    body = load(path)
    assert is_job_status(body)
    assert not is_job_results(body), "a status response must not also match the results envelope"
    summary = _status_summary(body)
    assert body["vertex_state"] in summary


def test_the_counts_are_integers_not_lists():
    """The misunderstanding this whole file exists to prevent, stated as an assertion."""
    body = load(RESULTS[0])
    assert isinstance(body["ranked_candidates"], int), "ranked_candidates is a COUNT"
    assert isinstance(body["candidates"], list), "candidates is the LIST"
