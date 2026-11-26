"""Food virtual digestion: deterministic Expasy-rule fragment enumeration and its provenance tags."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.food.digestion import DEFAULT_SETTINGS, Settings, peptide_id, read_measured_csv, reconcile, run  # noqa: E402
from pipeline.food.digestion.masses import mono_mass  # noqa: E402
from pipeline.food.protein_library.fasta import Substrate, read_fasta  # noqa: E402


def sub(seq: str, sid: str = "T1:1-0") -> Substrate:
    return Substrate(sid, "T1", 1, len(seq), seq)


def settings(**kw) -> Settings:
    base = {"name": "t", "pathways": [{"id": "tryp", "stages": [{"name": "i", "enzymes": ["trypsin"], "ph": 7.5}]}]}
    return Settings.from_dict({**base, **kw})


def test_trypsin_cuts_after_k_and_r_but_not_before_proline():
    _, peptides, _ = run([sub("AAAKPAAARGGG")], settings(max_missed_cleavages=0, min_length=1))
    assert {p["sequence"] for p in peptides} == {"AAAKPAAAR", "GGG"}


def test_missed_cleavages_add_longer_fragments():
    none = {p["sequence"] for p in run([sub("AAKGGRCC")], settings(max_missed_cleavages=0))[1]}
    one = {p["sequence"] for p in run([sub("AAKGGRCC")], settings(max_missed_cleavages=1))[1]}
    assert none < one and "AAKGGR" in one


def test_every_fragment_is_tagged_theoretical_with_positions():
    origins, _, _ = run([sub("AAKGGRCC")], settings())
    assert origins
    for o in origins:
        assert (o["origin_type"], o["evidence_class"]) == ("DIGESTION_DERIVED", "THEORETICAL")
        assert "AAKGGRCC"[o["start_in_product"] - 1 : o["end_in_product"]] == o["sequence"]


def test_limit_product_has_no_internal_site():
    origins, _, _ = run([sub("AAKGGRCC")], settings(max_missed_cleavages=1, min_length=1))
    assert {o["sequence"]: o["is_limit_product"] for o in origins}["AAKGGR"] is False
    assert {o["sequence"]: o["is_limit_product"] for o in origins}["AAK"] is True


def test_a_second_run_gives_identical_output():
    s = [sub("MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVK")]
    assert run(s, DEFAULT_SETTINGS) == run(s, DEFAULT_SETTINGS)


def test_peptide_id_depends_only_on_the_sequence():
    assert peptide_id("AAK") == peptide_id("AAK") != peptide_id("AAR")


def test_monoisotopic_mass_of_glycine_and_gly_gly():
    assert mono_mass("G") == pytest.approx(75.032029, abs=1e-5)
    assert mono_mass("GG") == pytest.approx(132.053493, abs=1e-5)


def test_unknown_enzyme_and_bad_lengths_are_refused():
    with pytest.raises(ValueError, match="unknown enzyme"):
        settings(pathways=[{"id": "x", "stages": [{"name": "s", "enzymes": ["nope"]}]}])
    with pytest.raises(ValueError, match="min_length"):
        settings(min_length=5, max_length=2)


def test_ph_outside_a_rules_range_warns():
    s = settings(pathways=[{"id": "x", "stages": [{"name": "g", "enzymes": ["trypsin"], "ph": 1.3}]}])
    assert s.ph_warnings()


def test_fasta_with_nonstandard_residues_is_reported_not_dropped_silently(tmp_path):
    f = tmp_path / "in.fa"
    f.write_text(">good\nAAKGG\n>bad\nAAXGG\n")
    subs, issues = read_fasta(f)
    assert [s.accession for s in subs] == ["good"]
    assert [(i.substrate_id, i.code) for i in issues] == [("bad:1-5", "NONSTANDARD_RESIDUES")]


def test_reconcile_marks_matches_and_what_the_run_could_not_have_made():
    subs = [sub("AAKGGRCC", "P1:1-8")]
    _, peptides, _ = run(subs, settings())
    measured = [{"sequence": s} for s in ("AAK", "GGGGG", "AAXK", "KGG")]
    status = {r["sequence"]: r["status"] for r in reconcile(peptides, subs, measured)}
    assert status == {
        "AAK": "MATCHED_THEORETICAL",
        "GGGGG": "NOT_IN_LIBRARY",
        "AAXK": "INVALID_SEQUENCE",
        "KGG": "OBSERVED_NOT_PREDICTED",  # inside the substrate, but trypsin cuts after K, so it is not a product
    }


def test_reconcile_matches_isoleucine_and_leucine_but_says_so():
    subs = [sub("AILKGG")]
    _, peptides, _ = run(subs, settings(min_length=1))
    (row,) = reconcile(peptides, subs, [{"sequence": "AIIK"}])
    assert (row["status"], row["il_ambiguous"]) == ("MATCHED_THEORETICAL", True)


def test_a_measured_match_is_observed_while_the_theoretical_peptide_stays_theoretical():
    subs = [sub("AAKGGRCC")]
    _, peptides, _ = run(subs, settings())
    (row,) = reconcile(peptides, subs, [{"sequence": "AAK"}])
    assert row["evidence_class"] == "OBSERVED"
    assert {p["evidence_class"] for p in peptides} == {"THEORETICAL"}


def test_measured_csv_without_a_sequence_column_is_refused(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("peptide" + chr(10) + "AAK" + chr(10))
    with pytest.raises(ValueError, match="sequence"):
        read_measured_csv(f)
    f.write_text("sequence,intensity" + chr(10) + "AAK,10" + chr(10))
    assert read_measured_csv(f) == [{"sequence": "AAK", "intensity": "10"}]


# The Phase 1 engine in Models/Food Peptide GLP1 Discovery, run once on these two proteins with DEFAULT_SETTINGS.
PARITY_PROTEINS = {
    "P02754:17-178": "LIVTQTMKGLDIQKVAGTWYSLAMAASDISLLDAQSAPLRVYVEELKPTPEGDLEILLQKWENGECAQKKIIAEKTKIPAVFKIDALNENKVLVLDTDYKKYLLFCMENSAEPEQSLACQCLVRTPEVDDEALEKFDKALKALPMHIRLSFNPTQLEEQCHI",
    "P02662:16-214": "RPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNELSKDIGSESTEDQAMEDIKQMEAESISSSEEIVPNSVEQKHIQKEDVPSERYLGYLEQLLRLKKYKVPQLEIVPNSAEERLHSMKEGIHAQQKEPMIGVNQELAYFYPELFRQFYQLDAYPSGAWYYVPLGTQYTDAPSFSDIPNPIGSENSEKTTMPLW",
}
PARITY_SHA256 = "1575bc93a88249a361ffd62891a28d66c9c208a686dd5af718325b313c2895c2"


def test_the_engine_reproduces_the_phase_1_output_exactly():
    """A refactor must not change which peptides come out, where they sit, or their masses.

    The hash covers every origin row, peptide row and cleavage-map row, so any difference fails it; there is no
    tolerance on masses. If the rules are changed on purpose, regenerate it from the engine and say why in the commit.
    """
    import hashlib
    import json

    subs = [Substrate(k, k.split(":")[0], int(k.split(":")[1].split("-")[0]), int(k.split("-")[1]), v) for k, v in PARITY_PROTEINS.items()]
    origins, peptides, cmap = run(subs, DEFAULT_SETTINGS)
    assert (len(origins), len(peptides), len(cmap)) == (926, 482, 370)
    assert hashlib.sha256(json.dumps([origins, peptides, cmap], sort_keys=True).encode()).hexdigest() == PARITY_SHA256


def test_the_inline_average_mass_equals_modlamp_for_every_peptide_of_a_real_digestion():
    import numpy as np
    from modlamp.descriptors import GlobalDescriptor

    from common.food_digestion.masses import avg_masses

    subs, _ = read_fasta(Path(__file__).resolve().parents[2] / "fixtures" / "food" / "phase1" / "digestion_substrates.fasta")
    seqs = sorted({o["sequence"] for o in run(subs[:6], DEFAULT_SETTINGS)[0]})[:4000] + ["A", "W" * 40, "ACDEFGHIKLMNPQRSTVWY"]
    g = GlobalDescriptor(seqs)
    g.calculate_MW(amide=False)
    assert avg_masses(seqs) == [float(x) for x in g.descriptor[:, 0]]  # identical, not close
    assert len(seqs) > 1000 and not np.isnan(avg_masses(seqs)).any()
