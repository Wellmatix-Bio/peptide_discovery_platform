"""The food protein library: its data is intact, discovery finds the right proteins, and an upload is matched without ever being rejected."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import pytest

from food.a_proteins.library import (
    ACCESSION_ONLY_MESSAGE,
    DATA,
    NOT_IN_LIBRARY_MESSAGE,
    STATUS_ACCESSION_ONLY,
    STATUS_EXACT,
    STATUS_NOT_IN_LIBRARY,
    FoodLibrary,
    UnknownLibraryProteins,
    library,
)
from food.a_proteins.fasta import Substrate

PHASE1_FASTA = Path(__file__).resolve().parents[1] / "fixtures" / "food" / "phase1" / "digestion_substrates.fasta"
L = library()


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def upload(sid: str, seq: str) -> Substrate:
    acc = sid.split(":")[0]
    return Substrate(substrate_id=sid, accession=acc, precursor_start=1, precursor_end=len(seq), sequence=seq)


def test_the_data_is_the_phase_1_library_and_matches_its_manifest():
    assert md5(DATA / "digestion_substrates.fasta") == L.manifest["files_md5"]["digestion_substrates.fasta"] == md5(PHASE1_FASTA)  # the set the parity tests use
    assert md5(DATA / "source_records.csv") == L.manifest["files_md5"]["source_records.csv"]
    assert (len(L.proteins), len({p.accession for p in L.proteins}), len(L.records)) == (524, 504, 504) == (L.manifest["n_substrates"], L.manifest["n_accessions"], L.manifest["n_source_records"])
    assert L.manifest["uniprot_release"] == ["2026_04"] and L.manifest["retrieved_at"] == ["2026-11-13"]


def test_every_protein_has_a_record_a_food_source_and_a_flag():
    assert all(p.food_source and p.allergen_source is not None and p.uniprot_release == "2026_04" and p.protein_family for p in L.proteins)
    assert {p.food_source for p in L.proteins} == {"Whey", "Casein", "Soy", "Pea", "Wheat"}


def test_the_summary_counts_proteins_by_source_family_and_flag():
    s = L.summary()
    assert s["total"] == 524 and s["by_food_source"] == {"Wheat": 314, "Whey": 70, "Soy": 57, "Pea": 60, "Casein": 23}
    assert sum(f["count"] for f in s["families"]) == 524 and sum(s["allergen_source_flags"].values()) == 524
    assert s["manifest"]["uniprot_release"] == ["2026_04"]


def test_search_filters_by_source_family_and_text_and_pages():
    total, page = L.search(food_source="casein", limit=5)
    assert total == 23 and len(page) == 5 and {p.food_source for p in page} == {"Casein"}
    family = page[0].protein_family
    assert all(p.protein_family == family for p in L.search(family=family, limit=500)[1])
    one = page[0]
    assert [p.substrate_id for p in L.search(q=one.accession)[1]] == [p.substrate_id for p in L.proteins if p.accession == one.accession]
    assert L.search(q="no such protein xyz")[0] == 0
    first, second = L.search(limit=3)[1], L.search(limit=3, offset=3)[1]
    assert not {p.substrate_id for p in first} & {p.substrate_id for p in second} and L.search(limit=3, offset=520)[1] == L.proteins[520:523]


def test_the_listing_never_carries_the_sequence():
    assert "sequence" not in L.search(limit=1)[1][0].info() and "header" not in L.search(limit=1)[1][0].info()


def test_resolve_keeps_the_order_drops_repeats_and_names_what_is_missing():
    a, b = L.proteins[3].substrate_id, L.proteins[1].substrate_id
    assert [p.substrate_id for p in L.resolve([a, b, a])] == [a, b]
    with pytest.raises(UnknownLibraryProteins) as caught:
        L.resolve([a, "NOPE:1-9", "ALSO:1-9"])
    assert caught.value.missing == ["NOPE:1-9", "ALSO:1-9"]


def test_the_fasta_and_records_of_chosen_proteins_round_trip():
    chosen = L.resolve([L.proteins[0].substrate_id, L.proteins[10].substrate_id])
    from food.a_proteins.fasta import read_fasta

    path = Path(__file__).parent / "_tmp_library.fasta"
    path.write_text(L.fasta_text(chosen), encoding="utf-8")
    try:
        subs, issues = read_fasta(path)
    finally:
        path.unlink()
    assert issues == [] and [s.sequence for s in subs] == [p.sequence for p in chosen] and subs[0].food_source == chosen[0].food_source  # the FS= tag survives
    rows = list(csv.DictReader(io.StringIO(L.records_csv(chosen))))
    assert [r["accession"] for r in rows] == [p.accession for p in chosen] and rows[0]["library_substrate_id"] == chosen[0].substrate_id and rows[0]["uniprot_release"] == "2026_04"


def test_an_upload_with_a_library_sequence_matches_whatever_id_it_has():
    p = L.proteins[5]
    (m,) = L.match_upload([upload("MY-WHEY-1", p.sequence)])
    assert (m.status, m.library_substrate_id, m.protein_family, m.allergen_source) == (STATUS_EXACT, p.substrate_id, p.protein_family, p.allergen_source)


def test_an_upload_that_is_not_in_the_library_is_reported_not_rejected():
    (m,) = L.match_upload([upload("NEW1", "MKVLAAGIVGLLLAQPSAFAEQ" * 3)])
    assert m.status == STATUS_NOT_IN_LIBRARY and m.message == NOT_IN_LIBRARY_MESSAGE and m.library_substrate_id is None and m.allergen_source is None


def test_the_same_accession_with_another_sequence_gets_no_library_data():
    p = L.proteins[7]
    changed = p.sequence[:-1] + ("A" if p.sequence[-1] != "A" else "G")
    (m,) = L.match_upload([upload(p.substrate_id, changed)])
    assert m.status == STATUS_ACCESSION_ONLY and m.message == ACCESSION_ONLY_MESSAGE and m.protein_family is None


def test_the_records_of_matched_uploads_are_keyed_by_the_uploads_accession():
    p = L.proteins[5]
    uploads = [upload("MY-WHEY-1", p.sequence), upload("NEW1", "MKVLAAGIVGLLLAQPSAFAEQ" * 3)]
    matches = L.match_upload(uploads)
    rows = list(csv.DictReader(io.StringIO(L.matched_records_csv(uploads, matches))))
    assert [r["accession"] for r in rows] == ["MY-WHEY-1"] and rows[0]["library_substrate_id"] == p.substrate_id and rows[0]["protein_family"] == p.protein_family
    assert L.matched_records_csv(uploads[1:], matches[1:]) is None  # nothing matched: no records


def test_a_library_built_from_other_data_loads_the_same_way(tmp_path):
    for name in ("digestion_substrates.fasta", "source_records.csv", "MANIFEST.json"):
        (tmp_path / name).write_bytes((DATA / name).read_bytes())
    assert len(FoodLibrary(tmp_path).proteins) == 524


def test_every_library_protein_has_both_ends_from_the_fasta_header():
    from food.a_proteins.fasta import TERMINUS_MEANING
    assert all(p.n_terminus in TERMINUS_MEANING and p.c_terminus in TERMINUS_MEANING for p in L.proteins)
    p = L.by_id["A0A023GQG9:22-749"]
    assert (p.n_terminus, p.c_terminus) == ("SIGNAL_CLEAVED", "SEQ_END") and p.info()["n_terminus"] == "SIGNAL_CLEAVED"


# --- where the library files come from: FOOD_LIBRARY_DIR --------------------------------------------------------

def test_with_no_setting_the_bundled_copy_is_used(monkeypatch):
    from food.a_proteins import library as fl

    monkeypatch.delenv("FOOD_LIBRARY_DIR", raising=False)
    assert fl.data_dir() == fl.BUNDLED_DATA


def test_a_local_folder_is_used_as_it_is_under_dev_mode(monkeypatch, tmp_path):
    import shutil

    from food.a_proteins import library as fl

    for name in fl.LIBRARY_FILES:
        shutil.copy(fl.BUNDLED_DATA / name, tmp_path / name)
    monkeypatch.setenv("FOOD_LIBRARY_DIR", str(tmp_path))
    assert fl.data_dir() == tmp_path and len(fl.FoodLibrary().proteins) == len(L.proteins)


def test_a_gcs_folder_is_downloaded_once_and_loads_the_same_library(monkeypatch):
    from food.a_proteins import library as fl

    reads = []

    def fake_read_text(path, **kwargs):
        reads.append(str(path))
        return (fl.BUNDLED_DATA / str(path).rsplit("/", 1)[1]).read_text(encoding="utf-8")

    monkeypatch.setattr(fl.storage, "read_text", fake_read_text)
    monkeypatch.setattr(fl, "_downloaded", {})
    monkeypatch.setenv("FOOD_LIBRARY_DIR", "gs://bucket/food_library")
    first = fl.data_dir()
    assert fl.data_dir() == first and sorted(reads) == sorted(f"gs://bucket/food_library/{n}" for n in fl.LIBRARY_FILES)  # three files, read once
    assert [p.substrate_id for p in fl.FoodLibrary().proteins] == [p.substrate_id for p in L.proteins]


def test_a_missing_file_in_the_gcs_folder_names_the_file_and_where_it_was_looked_for(monkeypatch):
    from food.a_proteins import library as fl

    def fake_read_text(path, **kwargs):
        if str(path).endswith("source_records.csv"):
            raise FileNotFoundError("no such object")
        return "x"

    monkeypatch.setattr(fl.storage, "read_text", fake_read_text)
    monkeypatch.setattr(fl, "_downloaded", {})
    monkeypatch.setenv("FOOD_LIBRARY_DIR", "gs://bucket/empty")
    with pytest.raises(fl.LibraryFilesMissing, match="source_records.csv could not be read from gs://bucket/empty/source_records.csv"):
        fl.data_dir()
