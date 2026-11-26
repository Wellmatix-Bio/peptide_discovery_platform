"""A run's proteins from the library: the records reach the pipeline's protein record, and the run's own allergen status covers only what the records do not."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from backend.worker_food.worker import load_run_proteins  # noqa: E402
from common.food_library import library  # noqa: E402
from common.protein_fasta import Substrate  # noqa: E402
from schemas.food_run_config import FoodRunConfig  # noqa: E402

L = library()


def config(tmp_path, fasta: str, records: str | None, **kw) -> FoodRunConfig:
    (tmp_path / "proteins.fasta").write_text(fasta, encoding="utf-8")
    if records:
        (tmp_path / "source_records.csv").write_text(records, encoding="utf-8")
    return FoodRunConfig(run_id="r", artifacts_dir=str(tmp_path), model_store="m", stages={}, fasta_path=str(tmp_path / "proteins.fasta"),
                         source_records_path=str(tmp_path / "source_records.csv") if records else "", **kw)


def test_library_proteins_arrive_with_family_flag_release_and_a_family_cluster(tmp_path):
    chosen = [next(p for p in L.proteins if p.food_source == "Casein"), next(p for p in L.proteins if p.allergen_source is False)]
    proteins, issues, notes = load_run_proteins(config(tmp_path, L.fasta_text(chosen), L.records_csv(chosen)))
    assert issues == [] and [p.substrate_id for p in proteins] == [c.substrate_id for c in chosen]
    for p, c in zip(proteins, chosen):
        assert (p.protein_family, p.allergen_source, p.uniprot_release, p.version, p.food_source) == (c.protein_family, c.allergen_source, c.uniprot_release, c.sequence_version, c.food_source)
        assert p.cluster_source == "family" and p.cluster == f"family:{c.food_source}: {c.protein_family}"  # family first, within its food source


def test_a_run_wide_allergen_status_does_not_override_a_records_flag(tmp_path):
    flagged_false = next(p for p in L.proteins if p.allergen_source is False)
    proteins, _, _ = load_run_proteins(config(tmp_path, L.fasta_text([flagged_false]), L.records_csv([flagged_false]), allergen_source=True))
    assert proteins[0].allergen_source is False  # the library's flag stands


def test_an_upload_that_matched_gets_the_library_data_and_one_that_did_not_keeps_what_was_supplied(tmp_path):
    p = next(x for x in L.proteins if x.food_source == "Whey")
    uploads = [Substrate(substrate_id="MY-1", accession="MY-1", precursor_start=1, precursor_end=len(p.sequence), sequence=p.sequence, food_source="Whey"),
               Substrate(substrate_id="NEW-2", accession="NEW-2", precursor_start=1, precursor_end=60, sequence="MKVLAAGIVGLLLAQPSAFAEQ" * 3, food_source="Whey")]
    matches = L.match_upload(uploads)
    fasta = "".join(f">{u.substrate_id} FS=Whey\n{u.sequence}\n" for u in uploads)
    proteins, issues, _ = load_run_proteins(config(tmp_path, fasta, L.matched_records_csv(uploads, matches), allergen_source=True))
    matched, other = proteins
    assert (matched.protein_family, matched.allergen_source, matched.uniprot_release, matched.cluster_source) == (p.protein_family, p.allergen_source, p.uniprot_release, "family")
    assert (other.protein_family, other.allergen_source, other.uniprot_release, other.cluster_source) == (None, True, None, "protein_id")  # the run's status, no library data
    assert [i.code for i in issues] == ["NO_SOURCE_RECORD"] and issues[0].substrate_id == "NEW-2:1-66"  # said, as a warning, not a rejection
