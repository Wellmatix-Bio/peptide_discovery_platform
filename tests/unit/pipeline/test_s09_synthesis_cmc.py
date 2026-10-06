"""Stage 9's synthesis gate and the rule-based scores behind it.

Stage 9 is the last stage that can remove a candidate, and it removes on ONE thing: the difficulty
class. Everything else it produces -- cost bands, purity ceiling, the ML feasibility prior -- is a
soft signal stage 11 weighs, and the tests below pin that separation, because a cost band quietly
becoming a reject condition would discard candidates on an estimate nobody agreed to gate on.

A note on the bug class that cost stages 5 and 8 five fixes: it does not arise here.
`compute_difficulty_class` is rule-based arithmetic over penalty drivers, disulfide pairs and
non-standard building blocks, so it always returns one of low/medium/high and never None. The ML
prior is the only model in the stage and is explicitly never a gate. That is worth knowing rather
than assuming, so `test_the_gate_never_sees_a_null_difficulty_class` checks it directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.s09_synthesis_cmc.stage import (  # noqa: E402
    DIFFICULTY_HIGH_MIN_SCORE,
    DIFFICULTY_MEDIUM_MIN_SCORE,
    REJECTED_DIFFICULTY_CLASSES,
    Stage9,
)


def stage() -> Stage9:
    """__init__ loads models; the scoring below is pure and needs none of them."""
    return Stage9.__new__(Stage9)


# -- The gate ----------------------------------------------------------------


@pytest.mark.parametrize("difficulty", ["low", "medium"])
def test_a_synthesisable_candidate_passes(difficulty: str):
    got = stage().compute_synthesis_verdict({"difficulty_class": difficulty}, {})
    assert got["overall"] == "pass"
    assert got["properties"]["difficulty_class"] == "pass"


def test_a_high_difficulty_candidate_is_rejected():
    got = stage().compute_synthesis_verdict({"difficulty_class": "high"}, {})
    assert got["overall"] == "reject"
    assert got["properties"]["difficulty_class"] == "reject"


def test_the_rejected_classes_are_configurable():
    got = stage().compute_synthesis_verdict(
        {"difficulty_class": "medium"}, {"rejected_difficulty_classes": ["medium", "high"]}
    )
    assert got["overall"] == "reject"


def test_an_empty_override_rejects_nothing_rather_than_falling_back_to_the_default():
    """`[]` is a choice, not an absence. Treating it as 'use the default' would silently reject
    high-difficulty candidates for a run that explicitly asked for none to be rejected."""
    got = stage().compute_synthesis_verdict(
        {"difficulty_class": "high"}, {"rejected_difficulty_classes": []}
    )
    assert got["overall"] == "pass"


def test_no_override_uses_the_shipped_default():
    assert REJECTED_DIFFICULTY_CLASSES == {"high"}
    got = stage().compute_synthesis_verdict({"difficulty_class": "high"}, {})
    assert got["overall"] == "reject"


def test_only_difficulty_class_can_reject():
    """Cost bands and the purity ceiling are stage 11's business. If one of them ever starts
    gating here, candidates begin disappearing on an estimate nobody agreed to gate on."""
    got = stage().compute_synthesis_verdict(
        {
            "difficulty_class": "low",
            "cost_band_100mg": "very_high",
            "cost_band_10g": "very_high",
            "achievable_final_purity_ceiling": 0.0,
            "predicted_crude_purity": 0.0,
            "ml_feasibility_prior": {"score": 0.0, "status": "ok"},
        },
        {},
    )
    assert got["overall"] == "pass"
    assert list(got["properties"]) == ["difficulty_class"]


# -- The difficulty class itself ---------------------------------------------


def test_a_clean_peptide_is_low_difficulty():
    assert stage().compute_difficulty_class([], {"n_pairs": 0, "requires_directed_folding": False}, []) == "low"


def test_difficulty_rises_with_penalty_drivers():
    penalties = [{"position": i} for i in range(DIFFICULTY_HIGH_MIN_SCORE)]
    pairs = {"n_pairs": 0, "requires_directed_folding": False}
    assert stage().compute_difficulty_class(penalties, pairs, []) == "high"
    fewer = [{"position": i} for i in range(DIFFICULTY_MEDIUM_MIN_SCORE)]
    assert stage().compute_difficulty_class(fewer, pairs, []) == "medium"


def test_directed_folding_dominates_the_score():
    """Directed folding adds 3 against a 'high' threshold of 5, so it is most of the way there on
    its own -- it is the expensive part of making one of these. Two penalty drivers finish it."""
    folding = {"n_pairs": 3, "requires_directed_folding": True}
    assert stage().compute_difficulty_class([], folding, []) == "medium"
    two = [{"position": 0}, {"position": 1}]
    assert stage().compute_difficulty_class(two, folding, []) == "high"


def test_a_building_block_no_vendor_stocks_raises_difficulty():
    """Each one adds 2. Two reach 'medium', three reach 'high'."""
    pairs = {"n_pairs": 0, "requires_directed_folding": False}
    unavailable = {"routine_vendor_availability": False}
    assert stage().compute_difficulty_class([], pairs, [unavailable] * 2) == "medium"
    assert stage().compute_difficulty_class([], pairs, [unavailable] * 3) == "high"


def test_a_routinely_available_building_block_does_not():
    blocks = [{"routine_vendor_availability": True}] * 3
    pairs = {"n_pairs": 0, "requires_directed_folding": False}
    assert stage().compute_difficulty_class([], pairs, blocks) == "low"


def test_the_gate_never_sees_a_null_difficulty_class():
    """Checked rather than assumed, because reading a null as a pass is the mistake that needed
    five fixes across stages 5 and 8. compute_difficulty_class is pure arithmetic over its three
    inputs and has no branch that can return None."""
    pairs = {"n_pairs": 0, "requires_directed_folding": False}
    for penalties in ([], [{"position": 0}], [{"position": i} for i in range(9)]):
        for blocks in ([], [{"routine_vendor_availability": False}]):
            assert stage().compute_difficulty_class(penalties, pairs, blocks) in {
                "low",
                "medium",
                "high",
            }


# -- Supporting calculations -------------------------------------------------


def test_disulfide_pairs_are_counted_from_cysteines():
    got = stage().count_disulfide_pairs("CACAC", {})
    assert got["cysteine_positions"] == [0, 2, 4]
    assert got["n_pairs"] == 1, "three cysteines make one pair and leave one unpaired"


def test_three_pairs_require_directed_folding():
    assert stage().count_disulfide_pairs("CCCCCC", {})["requires_directed_folding"] is True


def test_crude_purity_falls_as_the_peptide_lengthens():
    short = stage().compute_crude_purity("AAAA", [])
    longer = stage().compute_crude_purity("AAAAAAAAAAAAAAAAAAAA", [])
    assert 0.0 < longer < short <= 1.0


def test_a_single_residue_needs_no_couplings():
    assert stage().compute_crude_purity("A", []) == 1.0


def test_penalty_drivers_are_reported_in_sequence_order():
    """They are shown to a chemist as positions to substitute; out of order is a usability bug."""
    drivers = stage().identify_penalty_drivers("RRIIVVMMCCWWNGDG")
    assert drivers, "this sequence is built from penalised motifs and should flag several"
    assert [d["position"] for d in drivers] == sorted(d["position"] for d in drivers)


def test_a_benign_sequence_flags_nothing():
    assert stage().identify_penalty_drivers("AAAAKKAAAA") == []
