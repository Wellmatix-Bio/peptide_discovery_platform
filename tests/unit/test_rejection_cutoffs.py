# The cutoff text the results API attaches to each rejection reason.

from __future__ import annotations

import json

from backend.api_e2e import api


def _cutoffs(monkeypatch, pipeline: str, reasons: list[str], stages: dict) -> dict[str, str]:
    monkeypatch.setattr(api.storage, "read_text", lambda path: json.dumps({"stages": stages}))
    return api._rejection_cutoffs("run", pipeline, reasons)


def test_metabolic_defaults_and_overrides(monkeypatch) -> None:
    stages = {"s06_metabolic_functional_screening": {"params": {"ec50_max_nm": {"GLP1R": 250}}}}
    out = _cutoffs(monkeypatch, "metabolic", ["hemolysis", "plasma_half_life", "high_ec50:GLP1R", "cytotoxicity"], stages)
    assert out["hemolysis"] == "P(hemolytic) above 0.5"
    assert out["plasma_half_life"] == "below 60 min"
    assert out["high_ec50:GLP1R"] == "above 250 nM"
    assert out["cytotoxicity"] == "P(cytotoxic) above 0.5"


def test_wound_healing_hemolysis_is_a_phc50_cutoff(monkeypatch) -> None:
    assert _cutoffs(monkeypatch, "wound_healing", ["hemolysis"], {})["hemolysis"] == "pHC50 above 4"


def test_cosmetic_cutoffs(monkeypatch) -> None:
    out = _cutoffs(monkeypatch, "cosmetic", ["allergenicity", "half_life", "irritation_high"], {})
    assert out["allergenicity"] == "P(allergen) at or above 0.5"
    assert out["half_life"].startswith("log10 half-life below 3.55")
    assert "90th percentile" in out["irritation_high"]


def test_unknown_reason_has_no_cutoff_and_unreadable_config_uses_defaults(monkeypatch) -> None:
    monkeypatch.setattr(api.storage, "read_text", lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))
    out = api._rejection_cutoffs("run", "metabolic", ["something_new", "hemolysis"])
    assert "something_new" not in out
    assert out["hemolysis"] == "P(hemolytic) above 0.5"
