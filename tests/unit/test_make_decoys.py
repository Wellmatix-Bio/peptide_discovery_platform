import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_decoys.py"
spec = importlib.util.spec_from_file_location("make_decoys", SCRIPT)
make_decoys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_decoys)

POSITIVE = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"


def random_proteins(count=300, seed=0):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(make_decoys.AMINO), size=int(rng.integers(80, 400)))) for _ in range(count)]


def test_shuffles_keep_composition_and_differ_from_source():
    out = make_decoys.shuffles(POSITIVE, np.random.default_rng(0), 5)
    assert len(out) == 5
    for sequence in out:
        assert sorted(sequence) == sorted(POSITIVE)
        assert sequence != POSITIVE


def test_window_sampler_matches_length_charge_and_hydrophobicity():
    sampler = make_decoys.WindowSampler(random_proteins())
    charge, hydro = make_decoys.properties(POSITIVE)
    out = sampler.match(len(POSITIVE), charge, hydro, np.random.default_rng(0), 10)
    assert out
    for sequence in out:
        c, h = make_decoys.properties(sequence)
        assert len(sequence) == len(POSITIVE)
        assert abs(c - charge) <= make_decoys.CHARGE_TOL
        assert abs(h - hydro) <= make_decoys.HYDROPHOBICITY_TOL


def test_windows_never_span_two_proteins():
    proteins = ["A" * 10, "W" * 10]
    sampler = make_decoys.WindowSampler(proteins)
    for start in sampler.candidates(8, np.random.default_rng(0), draws=500):
        assert len(set(sampler.joined[start : start + 8])) == 1


def test_proteins_with_unusual_residues_are_dropped():
    assert make_decoys.WindowSampler(["ACDXK" * 10, "ACDEF" * 10]).proteins == ["ACDEF" * 10]


def test_build_is_deterministic_and_respects_identity_cutoff():
    benchmark = pd.DataFrame({"record_id": ["M1", "M2"], "sequence": [POSITIVE, "QRPRLSHKGPMPF"], "label": [1, 0]})
    proteins = random_proteins()
    first = make_decoys.build(benchmark, proteins, per_type=3, seed=1)
    second = make_decoys.build(benchmark, proteins, per_type=3, seed=1)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.matched_record_id) == {"M1"}  # only label-1 peptides get decoys
    assert (first.max_identity_to_benchmark < make_decoys.MAX_IDENTITY).all()
    assert first.sequence.is_unique
    assert set(first.type) <= {"shuffled", "protein_segment"}
