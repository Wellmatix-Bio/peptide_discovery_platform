"""run_state() in Python must agree with runState() in TypeScript, case for case.

There are two implementations of one rule, in two languages, because the API reconciles the
worker's self-report with the live Vertex state and the web app has to do the same for a response
from an older API that does not carry the field. Two copies drift. So rather than writing a
second table here and trusting a comment to keep them aligned, this READS THE TYPESCRIPT TABLE
and runs the Python function against it.

If web/src/test/runstate.test.ts gains a case, this picks it up. If its format changes enough
that the cases cannot be found, the guard below fails loudly instead of passing on an empty list
-- a parity test that silently checks nothing is worse than no parity test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.api_e2e.api import run_state

TS_CASES = Path(__file__).resolve().parents[2] / "web/src/test/runstate.test.ts"

#: ["running", "JOB_STATE_SUCCEEDED", "abandoned"], and the null-status variant.
CASE = re.compile(
    r"""\[\s*(?:"(?P<status>[a-z]+)"|(?P<null>null))\s*,\s*"(?P<vertex>JOB_STATE_[A-Z]+)"\s*,"""
    r"""\s*"(?P<expected>[a-z]+)"\s*\]""",
)

#: The count at the time of writing. A LOWER BOUND, not an equality: adding a case to the
#: TypeScript table should not fail this file, but losing the ability to parse it must.
MINIMUM_CASES = 14


def ts_cases() -> list[tuple[str | None, str, str]]:
    found = [
        (match.group("status"), match.group("vertex"), match.group("expected"))
        for match in CASE.finditer(TS_CASES.read_text(encoding="utf-8"))
    ]
    assert len(found) >= MINIMUM_CASES, (
        f"parsed only {len(found)} cases from {TS_CASES.name}; the table's format probably"
        " changed, and this test would otherwise pass without checking anything"
    )
    return found


@pytest.mark.parametrize(("status", "vertex_state", "expected"), ts_cases())
def test_python_agrees_with_the_typescript_table(
    status: str | None, vertex_state: str, expected: str
) -> None:
    assert run_state(status, vertex_state) == expected


def test_a_worker_that_reported_running_and_then_died_is_not_still_running() -> None:
    """The case the whole reconciliation exists for, pinned on its own.

    An earlier version required status to be None or "pending" before calling a terminal job
    abandoned, so a worker killed mid-run stayed "running" forever and the UI showed a finished
    job as still working.
    """
    assert run_state("running", "JOB_STATE_SUCCEEDED") == "abandoned"
    assert run_state("running", "JOB_STATE_FAILED") == "failed"


def test_no_vertex_state_at_all_is_submitted_not_succeeded() -> None:
    """The function's first line used to be `run_state = "succeeded"`, before the branch that
    always overwrote it. Nothing reached it, but a reader had to prove that; a job we know
    nothing about is submitted."""
    assert run_state(None, None) == "submitted"
    assert run_state(None, "JOB_STATE_UNSPECIFIED") == "submitted"
