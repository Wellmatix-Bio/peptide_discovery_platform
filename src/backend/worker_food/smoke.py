"""Smoke check of the food worker image: pymmseqs and its MMseqs2 binary cluster for real.

Run at image build (`python -m backend.worker_food.smoke`), so a build whose clustering does not work fails there instead of every run quietly giving each protein its own cluster.
It needs pymmseqs, which has Linux and macOS builds only, so it cannot pass on Windows.
"""
from __future__ import annotations

import random

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def check_clustering() -> None:
    from pipeline.food.protein_library.clusters import assign_clusters
    from pipeline.food.protein_library.protein import Protein

    rng = random.Random(7)
    parent = "".join(rng.choice(RESIDUES) for _ in range(150))
    variant = list(parent)
    for i in (10, 60, 120):  # 3 substitutions in 150 residues: 98% identity
        variant[i] = "A" if variant[i] != "A" else "G"
    unrelated = "".join(rng.choice(RESIDUES) for _ in range(150))
    proteins = [Protein(substrate_id=f"P{i}:1-150", accession=f"P{i}", precursor_start=1, precursor_end=150, sequence=seq, food_source="Test")
                for i, seq in enumerate([parent, "".join(variant), unrelated], 1)]
    out, notes = assign_clusters(proteins)
    assert notes and notes[0].startswith("pymmseqs (MMseqs2)"), f"clustering did not run: {notes}"
    assert out[0].cluster == out[1].cluster, "a 98% identical variant did not join its parent"
    assert out[2].cluster != out[0].cluster, "an unrelated protein joined the family"
    assert all(p.cluster_source == "sequence" for p in out)


if __name__ == "__main__":
    check_clustering()
    print("food worker smoke check passed: MMseqs2 clustering works")
