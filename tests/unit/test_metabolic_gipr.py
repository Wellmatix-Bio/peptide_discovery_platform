# GIPR is returned but flagged: the potency model cannot rank it, and every layer must say so.

from __future__ import annotations

from backend.api_e2e.api import _metabolic_candidate_response
from model_store.functional_potency_predictor_v1 import UNRELIABLE_POTENCY_RECEPTORS
from pipeline.s04_generation.routeA_metabolic import DEFAULT_TARGET_RECEPTORS
from pipeline.s06_functional_models.metabolic_stage import (
    RECEPTORS,
    UNRELIABLE_POTENCY_NOTE,
)


def test_gipr_is_the_unreliable_one() -> None:
    """Pins the fact the rest of this file is about. If the model is retrained and GIPR becomes
    rankable, this fails and every assertion below should be revisited rather than patched."""
    assert UNRELIABLE_POTENCY_RECEPTORS == ("GIPR",)


def test_route_a_defaults_exclude_exactly_the_unreliable_receptors() -> None:
    """routeA_metabolic writes its default out by hand, because importing the predictor there
    would pull torch in at module load. This is what stops the two drifting apart -- and they
    had: the constant that used to record the exclusion beside it was read by nothing."""
    assert set(DEFAULT_TARGET_RECEPTORS) == set(RECEPTORS) - set(
        UNRELIABLE_POTENCY_RECEPTORS
    )


def test_the_flag_reaches_the_api_response() -> None:
    candidate = {
        "id": "c1",
        "sequence": "HAEGTFTSD",
        "predictions": {
            "pec50": {"GIPR": 8.1, "GLP1R": 9.0},
            "ec50_nm": {"GIPR": 7.9, "GLP1R": 1.0},
            "potency_unreliable_receptors": ["GIPR"],
            "potency_unreliable_note": UNRELIABLE_POTENCY_NOTE,
        },
    }
    response = _metabolic_candidate_response(candidate)
    assert response.potency_unreliable_receptors == ["GIPR"]
    assert response.potency_unreliable_note == UNRELIABLE_POTENCY_NOTE
    # Flagged, NOT withheld: the caller still gets the number, which is the whole point.
    assert response.pec50["GIPR"] == 8.1
    assert response.ec50_nm["GIPR"] == 7.9


def test_a_run_that_did_not_target_gipr_carries_no_warning() -> None:
    """An empty list, not a note about a receptor the run never asked for."""
    response = _metabolic_candidate_response(
        {"id": "c2", "sequence": "HAEGTFTSD", "predictions": {"pec50": {"GLP1R": 9.0}}}
    )
    assert response.potency_unreliable_receptors == []
    assert response.potency_unreliable_note is None


def test_the_note_says_the_number_still_counts() -> None:
    """The flag must not read as 'this was excluded'. Stage 6 still gates on these receptors and
    Stage 11 still scores them, so the note has to be explicit about that or it misleads."""
    assert "still count" in UNRELIABLE_POTENCY_NOTE
    assert "unscreened" in UNRELIABLE_POTENCY_NOTE


def test_the_api_copies_of_the_flag_match_their_owning_definitions():
    """api.py keeps its own copy of the receptor list and the note.

    It has to: importing either original pulls torch into a process that only submits jobs and
    reads JSON. So the copies are held here instead, where the heavy imports are already paid
    for. A receptor added or removed upstream fails in this test rather than drifting silently
    into a response that flags the wrong thing.
    """
    from backend.api_e2e import api
    from model_store.functional_potency_predictor_v1 import UNRELIABLE_POTENCY_RECEPTORS
    from pipeline.s06_functional_models.metabolic_stage import UNRELIABLE_POTENCY_NOTE

    assert tuple(api.UNRELIABLE_POTENCY_RECEPTORS) == tuple(UNRELIABLE_POTENCY_RECEPTORS)
    assert api.UNRELIABLE_POTENCY_NOTE == UNRELIABLE_POTENCY_NOTE
