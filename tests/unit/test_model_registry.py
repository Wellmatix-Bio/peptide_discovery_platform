import json
import logging

from common.model_registry import ModelRef, ModelRegistry


def _card(root, dirname, card="auto"):
    d = root / dirname
    d.mkdir()
    if card == "auto":
        card = {"model_name": dirname}
    if isinstance(card, (dict, list)):
        card = json.dumps(card)
    if card is not None:
        (d / "model_card.json").write_text(card, encoding="utf-8")


def test_missing_store_is_empty(tmp_path):
    reg = ModelRegistry(tmp_path / "nope")
    assert len(reg) == 0 and reg.list() == [] and reg.get("x") is None
    assert reg.store_path == tmp_path / "nope"


def test_loads_cards_and_lookups(tmp_path):
    _card(tmp_path, "amp_classifier_v1")
    _card(tmp_path, "hemolysis_predictor_v1")
    _card(tmp_path, "hemolysis_predictor_v2")
    reg = ModelRegistry(tmp_path)
    assert reg.get("amp_classifier", "v1") == ModelRef("amp_classifier", "v1")
    assert reg.get("hemolysis_predictor", "v1") is not None
    assert reg.get("hemolysis_predictor").version == "v2"
    assert [r.version for r in reg.versions("hemolysis_predictor")] == ["v1", "v2"]
    assert reg.get("hemolysis_predictor", "v9") is None
    assert ModelRef("amp_classifier", "v1") in reg
    assert len(reg.list()) == 3


def test_name_falls_back_to_dir_when_card_lacks_it(tmp_path):
    _card(tmp_path, "mic_predictor_v1", card={"task": "regression"})
    assert ModelRegistry(tmp_path).get("mic_predictor", "v1") is not None


def test_checksum_from_card(tmp_path):
    _card(tmp_path, "a_v1", card={"model_name": "a_v1", "sha256": "abc"})
    assert ModelRegistry(tmp_path).get("a", "v1").checksum == "abc"


def test_bad_cards_skipped_with_warning(tmp_path, caplog):
    _card(tmp_path, "good_v1")
    _card(tmp_path, "broken_v1", card="{not json")
    _card(tmp_path, "listy_v1", card=[1, 2])
    _card(tmp_path, "noversion", card={"model_name": "noversion"})
    _card(tmp_path, "nocard_v1", card=None)
    with caplog.at_level(logging.WARNING):
        reg = ModelRegistry(tmp_path)
    assert [r.name for r in reg.list()] == ["good"]
    assert len(caplog.records) == 3


def test_store_path_is_a_file(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    assert len(ModelRegistry(f)) == 0
