"""Cleavage stability in Stage 8, wound-healing (on) and metabolic (off, TODO(cleavage)).

Uses a stub in place of CleavageSitePredictor, so no model weights are needed.
Run: pytest test.py   (or: python test.py)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pipeline.s08_safety_developability.stage import (  # noqa: E402
    CLEAVAGE_DECAY_RATE,
    CLEAVAGE_STABILITY_REJECT_MIN,
    WOUND_PROTEASE_ACCESSIONS,
    Stage8,
)

MMP9 = "P41245"  # in WOUND_PROTEASE_ACCESSIONS
NOT_WOUND = "P27487"  # DPP-4: metabolic panel, not on the wound allowlist
SEQUENCE = "GPLGIAGQ"


class StubCleavageModel:
    """Stands in for CleavageSitePredictor: fixed per-enzyme site positions and probabilities."""

    def __init__(self, panel: list[str], sites: dict[str, dict[int, float]] | None = None):
        self.panel = panel
        self.sites = sites or {}
        self.calls: list[str] = []

    def available_enzymes(self) -> list[dict]:
        return [{"enzyme_uniprot": acc} for acc in self.panel]

    def predict_one(self, sequence, distance_matrix, enzyme_accession):
        self.calls.append(enzyme_accession)
        assert distance_matrix.shape == (len(sequence), len(sequence))
        probs = [0.0] * len(sequence)
        for pos, p in self.sites.get(enzyme_accession, {}).items():
            probs[pos] = p
        return {
            "enzyme_accession": enzyme_accession,
            "enzyme_name": f"stub-{enzyme_accession}",
            "per_residue_probability": probs,
            "cleavage_site_positions": [i for i, p in enumerate(probs) if p >= 0.5],
            "max_probability": max(probs),
            "n_predicted_sites": sum(p >= 0.5 for p in probs),
        }


def candidate(**predictions):
    return SimpleNamespace(sequence=SEQUENCE, predictions=predictions)


def stability(model, cand=None, params=None):
    return Stage8.__new__(Stage8).compute_cleavage_stability(cand or candidate(), model, params or {})


# --- Wound-healing: compute_cleavage_stability -------------------------------------------------

def test_wound_allowlist_contains_the_test_enzyme():
    assert MMP9 in WOUND_PROTEASE_ACCESSIONS
    assert NOT_WOUND not in WOUND_PROTEASE_ACCESSIONS


def test_no_predicted_sites_scores_one():
    got = stability(StubCleavageModel([MMP9]))
    assert got["status"] == "ok"
    assert got["score"] == 1.0
    assert got["n_sites"] == 0 and got["sites"] == []


def test_score_decays_with_probability_weighted_sites():
    model = StubCleavageModel([MMP9], {MMP9: {2: 0.9, 5: 0.6}})
    got = stability(model)
    assert got["n_sites"] == 2
    assert got["score"] == pytest.approx(math.exp(-CLEAVAGE_DECAY_RATE * (0.9 + 0.6)))


def test_sites_below_the_probability_threshold_are_ignored():
    got = stability(StubCleavageModel([MMP9], {MMP9: {3: 0.49}}))
    assert got["score"] == 1.0


def test_exposure_weights_each_site():
    cand = candidate(structure={"exposure_by_position": [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]})
    got = stability(StubCleavageModel([MMP9], {MMP9: {2: 0.8}}), cand)
    assert got["sites"][0]["exposure"] == 0.5
    assert got["sites"][0]["weighted_severity"] == pytest.approx(0.4)
    assert got["score"] == pytest.approx(math.exp(-CLEAVAGE_DECAY_RATE * 0.4))


def test_buried_site_does_not_count():
    cand = candidate(structure={"exposure_by_position": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
    assert stability(StubCleavageModel([MMP9], {MMP9: {2: 0.99}}), cand)["score"] == 1.0


def test_exposure_length_mismatch_raises():
    cand = candidate(structure={"exposure_by_position": [1.0, 1.0]})
    with pytest.raises(ValueError):
        stability(StubCleavageModel([MMP9]), cand)


def test_stage7_distance_matrix_is_used_when_present():
    matrix = np.full((len(SEQUENCE), len(SEQUENCE)), 7.0)
    stability(StubCleavageModel([MMP9]), candidate(structure={"ca_distance_matrix": matrix.tolist()}))


def test_decay_rate_is_configurable():
    model = StubCleavageModel([MMP9], {MMP9: {2: 1.0}})
    assert stability(model, params={"cleavage_decay_rate": 2.0})["score"] == pytest.approx(math.exp(-2.0))


def test_only_allowlisted_enzymes_are_screened():
    model = StubCleavageModel([MMP9, NOT_WOUND], {NOT_WOUND: {2: 1.0}})
    got = stability(model)
    assert model.calls == [MMP9]
    assert got["n_wound_enzymes_screened"] == 1
    assert got["score"] == 1.0


def test_allowlist_override():
    model = StubCleavageModel([MMP9, NOT_WOUND], {NOT_WOUND: {2: 1.0}})
    got = stability(model, params={"wound_protease_accessions": [NOT_WOUND]})
    assert model.calls == [NOT_WOUND]
    assert got["score"] == pytest.approx(math.exp(-CLEAVAGE_DECAY_RATE))


def test_no_allowlisted_enzyme_in_panel_is_null_not_one():
    got = stability(StubCleavageModel([NOT_WOUND]))
    assert got["score"] is None
    assert got["status"] == "no_wound_relevant_enzymes_in_panel"


# --- Wound-healing verdict gate -------------------------------------------------------------------

CLEAN = {
    "hemolysis": {"phc50": 0.1},
    "cytotoxicity": {"score": 0.1},
    "aggregation_tendency": {"score": 0.1},
    "solubility": {"score": 0.9},
    "cleavage_stability": {"score": 0.9},
}


def wound_verdict(score, params=None):
    preds = {**CLEAN, "cleavage_stability": {"score": score}}
    return Stage8.__new__(Stage8).compute_safety_verdict(preds, params or {})


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.9, "pass"),
        (CLEAVAGE_STABILITY_REJECT_MIN, "pass"),
        (CLEAVAGE_STABILITY_REJECT_MIN - 0.01, "reject"),
        (None, "not_screened"),
    ],
)
def test_wound_verdict_gates_on_cleavage_stability(score, expected):
    assert wound_verdict(score)["properties"]["cleavage_stability"] == expected


def test_wound_reject_threshold_is_configurable():
    assert wound_verdict(0.5, {"cleavage_stability_reject_min": 0.6})["properties"]["cleavage_stability"] == "reject"


# --- Metabolic: cleavage is switched off ------------------------------------------------------------

def test_metabolic_verdict_ignores_cleavage_stability():
    pytest.importorskip("tqdm")
    from pipeline.s08_safety_developability.metabolic_stage import Stage8Metabolic

    preds = {
        **CLEAN,
        "hemolysis": {"p_hemolytic": 0.0},
        "plasma_half_life": {"minutes": 10_000.0},
        "cleavage_stability": {"score": 0.0},  # would reject if it were a gate
    }
    got = Stage8Metabolic.__new__(Stage8Metabolic).compute_safety_verdict(preds, {})
    assert "cleavage_stability" not in got["properties"]
    assert "cleavage_stability" not in got.get("not_screened", [])
    assert got["overall"] != "reject"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
