"""Protein-family clusters: UniProt family, then sequence clustering (MMseqs2 through pymmseqs), then one cluster per protein; each peptide inherits its parent's cluster."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from food.pipeline.stages.digestion import FoodDigestion  # noqa: E402
from food.a_proteins import families as clusters  # noqa: E402
from food.a_proteins.protein import Protein  # noqa: E402
from food.c_discovery.ranking import shortlist  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402

HAVE_PYMMSEQS = importlib.util.find_spec("pymmseqs") is not None


def prot(sid, seq="MKVLAAGIVGLLLAQPSAFAEQ", **kw):
    acc = sid.split(":")[0]
    return Protein(substrate_id=sid, accession=acc, precursor_start=1, precursor_end=len(seq), sequence=seq, food_source="Whey", **kw)


@pytest.fixture
def no_pymmseqs(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymmseqs", None)  # an import of it now raises ImportError, whether or not it is installed
    monkeypatch.setitem(sys.modules, "pymmseqs.commands", None)


def stand_in(groups, seen=None):
    """A pymmseqs `easy_cluster` that writes MMseqs2's `<prefix>_cluster.tsv` for `groups` (lists of the short ids it is given, representative first)."""
    def easy_cluster(fasta, prefix, work, min_seq_id=0.0, **kw):
        if seen is not None:
            seen.append({"fasta": Path(fasta).read_text(), "min_seq_id": min_seq_id})
        Path(prefix + "_cluster.tsv").write_text("".join(f"{g[0]}\t{m}\n" for g in groups for m in g))
        return SimpleNamespace()

    return easy_cluster


def test_without_pymmseqs_every_protein_is_its_own_cluster_and_the_note_says_so(no_pymmseqs):
    out, notes = clusters.assign_clusters([prot("A1:1-22"), prot("B2:1-22")])
    assert [(p.cluster, p.cluster_source) for p in out] == [("protein:A1", "protein_id"), ("protein:B2", "protein_id")]
    assert notes == ["pymmseqs is not installed (it has no Windows build)"]  # said, not hidden


def test_two_chains_of_one_protein_share_a_cluster_when_nothing_else_groups_them(no_pymmseqs):
    out, _ = clusters.assign_clusters([prot("A1:1-22"), prot("A1:30-51")])
    assert out[0].cluster == out[1].cluster == "protein:A1"


def test_a_protein_family_is_used_before_sequence_clustering(no_pymmseqs):
    out, _ = clusters.assign_clusters([prot("B2:1-22", protein_family="Lipocalin family"), prot("C3:1-22")])
    assert [(p.cluster, p.cluster_source) for p in out] == [("family:Whey: Lipocalin family", "family"), ("protein:C3", "protein_id")]


def test_a_family_name_shared_by_two_food_sources_does_not_merge_them(no_pymmseqs):
    first = prot("W1:1-22", protein_family="Other seed storage proteins")
    pea = Protein(**{**vars(prot("P1:1-22", protein_family="Other seed storage proteins")), "food_source": "Pea"})
    out, _ = clusters.assign_clusters([first, pea])
    assert [p.cluster for p in out] == ["family:Whey: Other seed storage proteins", "family:Pea: Other seed storage proteins"]


def test_only_proteins_without_a_family_are_sent_to_the_tool_at_the_platform_identity():
    seen = []
    ps = [prot("A1:1-22"), prot("B2:1-22"), prot("D4:1-22", protein_family="F")]
    out, notes = clusters.assign_clusters(ps, easy_cluster=stand_in([["0", "1"]], seen))
    assert seen[0]["fasta"].count(">") == 2 and seen[0]["min_seq_id"] == clusters.IDENTITY == 0.7  # A1 and B2 only
    assert [(p.cluster, p.cluster_source) for p in out] == [("seq:A1:1-22", "sequence"), ("seq:A1:1-22", "sequence"), ("family:Whey: F", "family")]
    assert notes == ["pymmseqs (MMseqs2) at 70% identity"]


def test_a_cluster_of_several_members_and_a_singleton_come_back_as_they_are():
    out, _ = clusters.assign_clusters([prot("A1:1-22"), prot("B2:1-22"), prot("C3:1-22")], easy_cluster=stand_in([["0", "1"], ["2"]]))
    assert [p.cluster for p in out] == ["seq:A1:1-22", "seq:A1:1-22", "seq:C3:1-22"]


def test_a_failing_tool_falls_back_to_one_cluster_per_protein_and_says_so():
    def broken(*args, **kwargs):
        raise RuntimeError("out of memory")

    out, notes = clusters.assign_clusters([prot("A1:1-22"), prot("B2:1-22")], easy_cluster=broken)
    assert [p.cluster_source for p in out] == ["protein_id", "protein_id"] and notes == ["pymmseqs failed: RuntimeError: out of memory"]


def test_a_tool_that_writes_nothing_is_a_failure_not_a_silent_success():
    out, notes = clusters.assign_clusters([prot("A1:1-22")], easy_cluster=lambda fasta, prefix, work, **kw: None)  # no _cluster.tsv written
    assert out[0].cluster_source == "protein_id" and notes[0].startswith("pymmseqs failed:")


def test_peptides_inherit_their_parents_cluster_and_the_ranking_cap_sees_it(no_pymmseqs):
    ps, _ = clusters.assign_clusters([prot("A1:1-22", seq="AKVLEPKRGGKLLRVAGK"), prot("B2:1-22", seq="GGKLLRVAGKAKVLEPKR")])
    ctx = type("Ctx", (), {"proteins": ps})()
    out = FoodDigestion().run([], StageConfig(), ctx)
    assert out and all(c.parent_clusters for c in out)
    shared = [c for c in out if len(c.parent_clusters) == 2]
    assert shared and shared[0].parent_clusters == ["protein:A1", "protein:B2"]  # a peptide from both proteins keeps both
    assert all(shortlist.first_parent_cluster({"parent_clusters": ";".join(c.parent_clusters)}) != "" for c in out)  # no longer one unknown group


@pytest.mark.skipif(not HAVE_PYMMSEQS, reason="pymmseqs has Linux and macOS builds only; it runs in CI and in the worker image, which also runs this check at build")
def test_real_mmseqs2_groups_a_close_variant_with_its_parent_and_leaves_an_unrelated_protein_alone():
    from backend.worker_food.smoke import check_clustering

    check_clustering()  # a 98% identical variant joins its parent; an unrelated protein does not


def test_a_family_is_approved_only_when_none_of_its_proteins_has_a_qc_issue_not_even_a_warning():
    from food.a_proteins.fasta import QCIssue

    clean = prot("A:1-22", protein_family="beta-LG")
    warned = prot("B:1-22", protein_family="casein")
    rows = {r["family"]: r for r in clusters.approve_families([clean, warned], [QCIssue("B:1-22", "warning", "RANGE_LENGTH_MISMATCH", "x")])}
    assert rows["beta-LG"]["approved"] is True
    assert rows["casein"]["approved"] is False and "B:1-22 RANGE_LENGTH_MISMATCH" in rows["casein"]["reason"]


def test_proteins_with_no_family_are_one_row_family_not_established_and_never_approved():
    rows = clusters.approve_families([prot("X:1-22"), prot("Y:1-22")], [])
    assert len(rows) == 1 and rows[0]["family"] == clusters.FAMILY_NOT_ESTABLISHED
    assert rows[0]["approved"] is False and rows[0]["protein_ids"] == ["X:1-22", "Y:1-22"]


def test_the_same_family_name_in_two_food_sources_is_two_families():
    a, b = prot("A:1-22", protein_family="Other storage"), prot("B:1-22", protein_family="Other storage")
    b.food_source = "Soy"
    assert [r["food_source"] for r in clusters.approve_families([a, b], [])] == ["Whey", "Soy"]
