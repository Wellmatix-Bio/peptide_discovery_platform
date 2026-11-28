"""The digestion endpoints: the real engine over the API, equal to what stage s03 of the pipeline starts from."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api_e2e import api
from food.b_digestion.engine import run as engine_run
from food.b_digestion.rules import BUILTIN_ENZYMES
from food.b_digestion.settings import DEFAULT_SETTINGS
from food.b_digestion.validation import RULE_VALIDATION
from food.a_proteins.library import library

L = library()
CLIENT = TestClient(api.app)
SETTINGS, RUN = "/api/v1/food/digestion/settings", "/api/v1/food/digestion/run"
CHOSEN = [p for p in L.proteins if p.food_source == "Whey"][:3]
IDS = [p.substrate_id for p in CHOSEN]


def digest(**body):
    return CLIENT.post(RUN, json=body)


def test_the_registry_lists_every_rule_with_its_pattern_its_ph_and_what_the_benchmark_says():
    body = CLIENT.get(SETTINGS).json()
    names = {e["name"] for e in body["enzymes"]}
    assert names == set(BUILTIN_ENZYMES) and {p["id"] for p in body["settings"]["pathways"]} == {p.id for p in DEFAULT_SETTINGS.pathways}
    by = {e["name"]: e for e in body["enzymes"]}
    assert by["trypsin"]["validation"].startswith("validated") and by["chymotrypsin_high"]["validation"].startswith("not validated")
    assert by["trypsin"]["pattern"] == BUILTIN_ENZYMES["trypsin"].pattern and by["pepsin_ph1.3"]["ph_range"] == [1.0, 2.0]
    assert body["limits"]["max_proteins"] == 50 and any("hypothesis" in s for s in body["limitations"])


def test_the_validation_text_matches_the_benchmark_report_it_cites():
    report = json.loads((Path(__file__).resolve().parents[2] / "docs" / "validation" / "food" / "protease_validation.json").read_text())

    def balanced(entry):
        return round(entry["balanced_accuracy"], 3), round(entry["balanced_accuracy_ci95"][0], 3)

    expected = {
        "trypsin": balanced(report["merops"]["trypsin 1 | trypsin | strict"]),
        "chymotrypsin_low": balanced(report["merops"]["chymotrypsin A | chymotrypsin_low | strict"]),
        "chymotrypsin_high": balanced(report["merops"]["chymotrypsin A | chymotrypsin_high | strict"]),
        "pepsin_ph1.3": balanced(report["yeast_pepsin"]["pepsin A yeast primary | pepsin_ph1.3"]),
        "pepsin_ph2.0": balanced(report["yeast_pepsin"]["pepsin A yeast primary | pepsin_ph2.0"]),
    }
    for rule, (accuracy, lower) in expected.items():
        text = RULE_VALIDATION[rule]
        assert f"balanced accuracy {accuracy:.3f}" in text and f"lower bound {lower:.3f}" in text, (rule, text, accuracy, lower)
    assert set(RULE_VALIDATION) == set(BUILTIN_ENZYMES)


def test_library_proteins_digest_to_exactly_what_the_pipeline_engine_gives():
    body = digest(library_proteins=IDS, limit=2000).json()
    origins, peptides, _ = engine_run(L.substrates(L.resolve(IDS)), DEFAULT_SETTINGS)
    assert body["evidence_class"] == "THEORETICAL" and body["n_proteins"] == 3 and body["n_peptides"] == len(peptides) and body["n_origins"] == len(origins)
    got = {p["peptide_id"]: p for p in body["peptides"]}
    assert set(got) == {p["peptide_id"] for p in peptides[:2000]}
    sample = peptides[0]
    assert (got[sample["peptide_id"]]["mono_mass"], got[sample["peptide_id"]]["avg_mass"]) == (sample["mono_mass"], sample["avg_mass"])


def test_every_origin_sequence_sits_at_its_stated_position_in_its_parent():
    body = digest(library_proteins=IDS, limit=2000).json()
    seq = {p.substrate_id: p.sequence for p in CHOSEN}
    precursor = {s.substrate_id: s.precursor_start for s in L.substrates(CHOSEN)}
    checked = 0
    for p in body["peptides"]:
        for o in p["origins"]:
            assert seq[o["substrate_id"]][o["start_in_product"] - 1:o["end_in_product"]] == p["sequence"]
            assert o["start_in_precursor"] == precursor[o["substrate_id"]] + o["start_in_product"] - 1 and o["end_in_precursor"] == precursor[o["substrate_id"]] + o["end_in_product"] - 1
            checked += 1
    assert checked > 500


def test_a_pathway_subset_uses_only_its_enzymes_and_its_cleavage_map_is_the_rules_sites():
    body = digest(library_proteins=IDS[:1], pathways=["trypsin_only"], limit=2000).json()
    assert {o["pathway"] for p in body["peptides"] for o in p["origins"]} == {"trypsin_only"} and {tuple(o["enzymes"]) for p in body["peptides"] for o in p["origins"]} == {("trypsin",)}
    (cmap,) = body["cleavage_map"]
    assert cmap["pathway"] == "trypsin_only" and cmap["length"] == len(CHOSEN[0].sequence)
    assert [s["p1_position"] for s in cmap["sites"]] == [i + 1 for i in BUILTIN_ENZYMES["trypsin"].sites(CHOSEN[0].sequence)]
    assert all(s["cleaved_by"] == ["intestinal:trypsin"] for s in cmap["sites"]) and body["rule_validation"] == {"trypsin": RULE_VALIDATION["trypsin"]}


def test_fewer_missed_cleavages_give_fewer_fragments_and_the_settings_used_are_returned():
    full = digest(library_proteins=IDS[:1]).json()
    none = digest(library_proteins=IDS[:1], max_missed_cleavages=0).json()
    assert none["n_peptides"] < full["n_peptides"] and none["settings"]["max_missed_cleavages"] == 0 and full["settings"]["max_missed_cleavages"] == 2
    assert all(o["missed_cleavages_final"] == 0 for p in digest(library_proteins=IDS[:1], max_missed_cleavages=0, limit=2000).json()["peptides"] for o in p["origins"] if o["pathway"] == "trypsin_only")


def test_pages_do_not_overlap_and_truncation_is_stated():
    first = digest(library_proteins=IDS, limit=100).json()
    second = digest(library_proteins=IDS, limit=100, offset=100).json()
    assert first["truncated"] is True and len(first["peptides"]) == 100 and not {p["peptide_id"] for p in first["peptides"]} & {p["peptide_id"] for p in second["peptides"]}
    assert digest(library_proteins=IDS[:1], limit=2000).json()["truncated"] is False


def test_an_uploaded_fasta_digests_the_same_way_and_a_bad_record_is_skipped():
    body = digest(fasta=f">MY1\n{CHOSEN[0].sequence}\n>BAD\nAXB\n", pathways=["pepsin_only"], limit=2000).json()
    assert body["n_proteins"] == 1 and {p["parents"][0] for p in body["peptides"]} == {f"MY1:1-{len(CHOSEN[0].sequence)}"}


@pytest.mark.parametrize("body", [
    {},  # no input
    {"fasta": ">a\nAAAAAA\n", "library_proteins": ["P02754:17-178"]},  # both
    {"fasta": ">a\nAXB\n"},  # nothing usable
    {"fasta": "AAAA"},  # not FASTA
    {"library_proteins": ["NOT:1-9"]},
    {"library_proteins": ["P02754:17-178"], "pathways": ["no_such_pathway"]},
    {"library_proteins": ["P02754:17-178"], "proteases": ["no_such_protease"]},
    {"library_proteins": ["P02754:17-178"], "proteases": []},
    {"library_proteins": ["P02754:17-178"], "proteases": ["trypsin"], "pathways": ["trypsin_only"]},  # one or the other
    {"library_proteins": ["P02754:17-178"], "min_length": 30, "max_length": 10},
    {"library_proteins": ["P02754:17-178"], "limit": 5000},
    {"library_proteins": ["P02754:17-178"], "max_missed_cleavages": 9},
])
def test_a_request_that_cannot_run_is_a_422(body):
    assert CLIENT.post(RUN, json=body).status_code == 422


def test_more_than_the_limit_of_proteins_is_refused():
    response = digest(library_proteins=[p.substrate_id for p in L.proteins[:51]])
    assert response.status_code == 422 and "At most 50 proteins" in response.text


def test_a_rule_is_never_taken_from_the_request():
    assert CLIENT.post(RUN, json={"library_proteins": IDS[:1], "custom_enzymes": {"x": {"pattern": "(a+)+$"}}}).status_code == 422  # unknown field: no user regex reaches the server


def test_a_peptide_id_is_pep_and_twelve_hex_characters():
    from food.b_digestion.engine import peptide_id

    assert re.match(r"PEP-[0-9A-F]{12}$", peptide_id("ACDE"))


def test_the_settings_name_the_proteases_a_run_can_be_limited_to_with_their_single_enzyme_pathways():
    body = CLIENT.get(SETTINGS).json()
    assert body["proteases"] == {"pepsin": "pepsin_only", "trypsin": "trypsin_only", "chymotrypsin": "chymotrypsin_only"}
    assert set(body["proteases"].values()) <= {p["id"] for p in body["settings"]["pathways"]}


@pytest.mark.parametrize("chosen, pathways", [
    (["trypsin"], ["trypsin_only"]),
    (["chymotrypsin", "trypsin"], ["trypsin_only", "chymotrypsin_only", "trypsin_chymotrypsin"]),  # together = the default pathway of the same stages
    (["pepsin", "trypsin", "chymotrypsin"], ["pepsin_only", "trypsin_only", "chymotrypsin_only", "gastric_to_intestinal"]),
    (["pepsin", "trypsin"], ["pepsin_only", "trypsin_only", "pepsin_trypsin"]),  # no default pathway of these stages: a new id
])
def test_chosen_proteases_run_each_alone_and_together_in_digestion_order(chosen, pathways):
    body = digest(library_proteins=IDS[:1], proteases=chosen, limit=2000).json()
    assert [p["id"] for p in body["settings"]["pathways"]] == pathways
    together = body["settings"]["pathways"][-1]
    if len(chosen) > 1:
        assert [s["name"] for s in together["stages"]] == sorted({s["name"] for s in together["stages"]}, key=["gastric", "intestinal"].index)
        assert {e for s in together["stages"] for e in s["enzymes"]} == {e for pid in pathways[:-1] for p in body["settings"]["pathways"] if p["id"] == pid for s in p["stages"] for e in s["enzymes"]}
    assert {pw for pep in body["peptides"] for pw in pep["pathways"]} <= set(pathways)
    assert [m["pathway"] for m in body["cleavage_map"]] == pathways


def test_protease_pathways_are_built_from_the_default_stage_definitions_with_no_new_rule_or_ph():
    from food.b_digestion.settings import PROTEASE_PATHWAYS, protease_pathways

    defaults = {st for p in DEFAULT_SETTINGS.pathways for st in p.stages}
    built = protease_pathways(DEFAULT_SETTINGS, list(PROTEASE_PATHWAYS))
    assert all(st in defaults for p in built for st in p.stages)
    with pytest.raises(ValueError):
        protease_pathways(DEFAULT_SETTINGS, ["papain"])
