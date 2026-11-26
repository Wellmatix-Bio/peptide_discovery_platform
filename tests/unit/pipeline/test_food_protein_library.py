"""Food protein library: FASTA reading with QC, and the join to the data foundation's source records."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.food.protein_library.fasta import read_fasta  # noqa: E402
from pipeline.food.protein_library.protein import Protein, load_proteins  # noqa: E402

RECORDS = "accession,sequence_version,checksum,organism,taxon_id,protein_family,allergen_source_flag,retrieved_at,uniprot_release\n"


def write(tmp_path, fasta: str, records: str | None = None):
    f = tmp_path / "s.fasta"
    f.write_text(fasta, encoding="utf-8")
    r = None
    if records is not None:
        r = tmp_path / "source_records.csv"
        r.write_text(RECORDS + records, encoding="utf-8")
    return f, r


def test_fasta_with_bad_records_reports_each_one_and_drops_none_silently(tmp_path):
    f, _ = write(tmp_path, ">good\nAAKGG\n>bad\nAAXGG\n>good\nAAKGG\n>empty\n")
    subs, issues = read_fasta(f)
    assert [s.accession for s in subs] == ["good"]
    assert [(i.substrate_id, i.code) for i in issues] == [("bad:1-5", "NONSTANDARD_RESIDUES"), ("good:1-5", "DUPLICATE_ID"), ("empty:1-0", "EMPTY_SEQUENCE")]


def test_proteins_carry_the_source_record(tmp_path):
    f, r = write(tmp_path, ">P1:5-9 KIND=chain FS=Whey\nAAKGG\n", "P1,2,abc,Bos taurus,9913,beta-LG,True,2026-11-13,2026_04\n")
    (p,), issues = load_proteins(f, r)
    assert isinstance(p, Protein) and issues == []
    assert (p.substrate_id, p.accession, p.precursor_start, p.food_source) == ("P1:5-9", "P1", 5, "Whey")
    assert (p.version, p.checksum, p.organism, p.allergen_source, p.uniprot_release) == ("2", "abc", "Bos taurus", True, "2026_04")


def test_a_protein_with_no_source_record_has_an_unknown_allergen_flag_and_a_warning(tmp_path):
    f, r = write(tmp_path, ">P1\nAAKGG\n>P2\nAAKGG\n", "P1,1,x,o,1,fam,False,d,r\n")
    proteins, issues = load_proteins(f, r)
    assert [p.allergen_source for p in proteins] == [False, None]
    assert [(i.substrate_id, i.code, i.severity) for i in issues] == [("P2:1-5", "NO_SOURCE_RECORD", "warning")]


def test_without_a_source_table_nothing_is_guessed(tmp_path):
    f, _ = write(tmp_path, ">P1\nAAKGG\n")
    (p,), issues = load_proteins(f)
    assert (p.allergen_source, p.version, p.checksum, p.cluster, issues) == (None, None, None, None, [])
