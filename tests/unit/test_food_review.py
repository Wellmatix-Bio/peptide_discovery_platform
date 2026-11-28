"""The approval gate's bookkeeping: the latest decision per peptide, the history, and where the gate stands."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from food.shared.dossier import GATE_RULE, ReviewRequest, gate_summary, latest, new_record, parse_log


def record(peptide, decision, reason="", reviewer="Hazara"):
    return new_record(peptide, "AAAA", ReviewRequest(decision=decision, reason=reason, reviewer=reviewer))


def test_the_latest_line_for_a_peptide_is_its_current_decision_and_the_log_round_trips():
    records = [record("P1", "hold"), record("P2", "approve", "ok"), record("P1", "reject", "no")]
    text = "".join(r.model_dump_json() + "\n" for r in records)
    back = parse_log(text)
    assert back == records and {k: v["decision"] for k, v in latest(back).items()} == {"P1": "reject", "P2": "approve"}
    assert parse_log("") == []


def test_the_gate_status_follows_how_many_shortlisted_peptides_have_a_decision():
    shortlisted = ["P1", "P2", "P3"]
    assert gate_summary([], {})["status"] == "NO_SHORTLIST"
    assert gate_summary(shortlisted, {})["status"] == "AWAITING_REVIEW"
    partial = gate_summary(shortlisted, latest([record("P1", "approve")]))
    assert partial["status"] == "IN_REVIEW" and partial["approved"] == ["P1"] and partial["pending"] == ["P2", "P3"] and partial["rule"] == GATE_RULE
    done = gate_summary(shortlisted, latest([record("P1", "approve"), record("P2", "hold"), record("P3", "reject"), record("P9", "approve")]))  # P9 is not shortlisted
    assert done["status"] == "REVIEW_COMPLETE" and (done["approved"], done["on_hold"], done["rejected"], done["pending"]) == (["P1"], ["P2"], ["P3"], [])


def test_the_reviewer_is_required_and_the_reason_is_trimmed():
    with pytest.raises(ValidationError):
        ReviewRequest(decision="approve", reviewer="")
    with pytest.raises(ValidationError):
        ReviewRequest(decision="maybe", reviewer="H")
    r = record("P1", "approve", "  fine  ", " Hazara ")
    assert (r.reason, r.reviewer) == ("fine", "Hazara") and r.decided_at.endswith("+00:00")
