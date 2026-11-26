"""The model registry of a run: a gs:// model store holds weights, so the cards come from the model_store folder that ships with the code."""
from __future__ import annotations

import logging

import runner


def services(tmp_path, model_store):
    return runner.build_run_services("r1", str(tmp_path), model_store, feature_extractor=object())


def test_a_gcs_model_store_gets_the_cards_that_ship_with_the_code_and_no_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="common.model_registry"):
        ctx = services(tmp_path, "gs://bucket/model_weights")
    assert len(ctx.models) > 10 and ctx.models.get("cytotoxicity_predictor", "v1") is not None  # the cards of the code folder, not an empty registry
    assert not [r for r in caplog.records if "not readable" in r.getMessage()]


def test_a_local_model_store_is_read_as_it_is(tmp_path):
    card = tmp_path / "demo_predictor_v1"
    card.mkdir()
    (card / "model_card.json").write_text('{"name": "demo_predictor", "version": "v1"}', encoding="utf-8")
    ctx = services(tmp_path, str(tmp_path))
    assert ctx.models.get("demo_predictor", "v1") is not None and len(ctx.models) == 1
