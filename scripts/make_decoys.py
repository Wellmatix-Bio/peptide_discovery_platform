"""Builds the matched decoy set for the metabolic benchmark. Run once, before any pipeline run; the output is then fixed.

For each label-1 peptide it writes N shuffled decoys (same residues, new order) and N protein-segment decoys (a window of the same
length cut from a real protein, matched on net charge and mean hydrophobicity). Decoys too similar to any benchmark peptide are dropped.

    python -I scripts/make_decoys.py --benchmark <csv> --proteins <fasta.gz> --out data/raw/benchmark/decoys_metabolic.csv
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from common.similarity import best_identities  # noqa: E402

AMINO = "ACDEFGHIKLMNPQRSTVWY"
EISENBERG = dict(zip(AMINO, [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06, 0.64, -0.78, 0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26]))
CHARGE = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0}  # histidine counted as neutral; the match tolerance absorbs the approximation
SEED = 7
PER_TYPE = 5
CHARGE_TOL = 1.0
HYDROPHOBICITY_TOL = 0.1
MAX_IDENTITY = 0.5  # a decoy at or above this identity to any benchmark peptide is dropped
MAX_SHUFFLE_IDENTITY = 0.4  # a shuffle must also differ this much from the peptide it came from


def properties(sequence: str) -> tuple[float, float]:
    return sum(CHARGE.get(a, 0.0) for a in sequence), float(np.mean([EISENBERG[a] for a in sequence]))


def read_fasta(path: Path) -> list[str]:
    sequences, current = [], []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                current = []
            else:
                current.append(line.strip())
    if current:
        sequences.append("".join(current))
    return sequences


class WindowSampler:
    """Random same-length windows of real proteins, with their charge and hydrophobicity, ready for matching."""

    def __init__(self, proteins: list[str]) -> None:
        proteins = [p for p in proteins if set(p) <= set(AMINO)]  # proteins with unusual residues would give windows the models cannot score
        self.proteins = proteins
        self.lengths = np.array([len(p) for p in proteins])
        self.joined = "".join(proteins)
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.charge = np.concatenate([[0.0], np.cumsum([CHARGE.get(a, 0.0) for a in self.joined])])
        self.hydro = np.concatenate([[0.0], np.cumsum([EISENBERG[a] for a in self.joined])])

    def candidates(self, length: int, rng: np.random.Generator, draws: int = 200_000) -> np.ndarray:
        """Start offsets (into the joined text) of windows of `length` that lie inside one protein."""
        eligible = np.flatnonzero(self.lengths >= length)
        if len(eligible) == 0:
            return np.array([], dtype=int)
        protein = rng.choice(eligible, size=draws)
        offset = (rng.random(draws) * (self.lengths[protein] - length + 1)).astype(int)
        return self.starts[protein] + offset

    def match(self, length: int, charge: float, hydrophobicity: float, rng: np.random.Generator, want: int) -> list[str]:
        starts = self.candidates(length, rng)
        window_charge = self.charge[starts + length] - self.charge[starts]
        window_hydro = (self.hydro[starts + length] - self.hydro[starts]) / length
        ok = (np.abs(window_charge - charge) <= CHARGE_TOL) & (np.abs(window_hydro - hydrophobicity) <= HYDROPHOBICITY_TOL)
        found, seen = [], set()
        for start in starts[ok]:
            sequence = self.joined[start : start + length]
            if sequence not in seen:
                seen.add(sequence)
                found.append(sequence)
            if len(found) >= want:
                break
        return found


def shuffles(sequence: str, rng: np.random.Generator, want: int) -> list[str]:
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score, aligner.extend_gap_score = -10.0, -1.0
    found, seen = [], {sequence}
    for _ in range(500):
        candidate = "".join(rng.permutation(list(sequence)))
        if candidate in seen:
            continue
        identity = aligner.align(sequence, candidate)[0].counts().identities / len(sequence)
        if identity < MAX_SHUFFLE_IDENTITY:
            seen.add(candidate)
            found.append(candidate)
        if len(found) >= want:
            break
    return found


def build(benchmark: pd.DataFrame, proteins: list[str], per_type: int = PER_TYPE, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampler = WindowSampler(proteins)
    positives = benchmark[benchmark.label == 1]
    rows = []
    for record in positives.itertuples():
        charge, hydro = properties(record.sequence)
        made = {
            "shuffled": shuffles(record.sequence, rng, per_type * 2),
            "protein_segment": sampler.match(len(record.sequence), charge, hydro, rng, per_type * 4),
        }
        for kind, sequences in made.items():
            for sequence in sequences:
                rows.append({"type": kind, "matched_record_id": record.record_id, "sequence": sequence})
    pool = pd.DataFrame(rows)
    pool["max_identity_to_benchmark"] = best_identities(list(pool.sequence), list(benchmark.sequence))
    pool = pool[pool.max_identity_to_benchmark < MAX_IDENTITY]
    pool = pool.drop_duplicates("sequence")  # one decoy sequence is used once, under its first matched peptide
    kept = pool.groupby(["matched_record_id", "type"], sort=False).head(per_type).reset_index(drop=True)
    kept["length"] = kept.sequence.str.len()
    kept["net_charge"], kept["hydrophobicity"] = zip(*[properties(s) for s in kept.sequence])
    kept["decoy_id"] = [f"D{index + 1:04d}" for index in range(len(kept))]
    kept["seed"] = seed
    return kept[["decoy_id", "type", "matched_record_id", "sequence", "length", "net_charge", "hydrophobicity", "max_identity_to_benchmark", "seed"]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    benchmark = pd.read_csv(args.benchmark)
    benchmark["sequence"] = benchmark.sequence.astype(str).str.strip()
    benchmark = benchmark[benchmark.sequence.str.fullmatch(f"[{AMINO}]+")].reset_index(drop=True)
    decoys = build(benchmark, read_fasta(args.proteins))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    decoys.to_csv(args.out, index=False)

    short = decoys.groupby(["matched_record_id", "type"]).size().unstack(fill_value=0)
    positives = benchmark[benchmark.label == 1].record_id
    short = short.reindex(positives, fill_value=0).reindex(columns=["shuffled", "protein_segment"], fill_value=0)
    manifest = {
        "seed": SEED, "per_type": PER_TYPE, "charge_tolerance": CHARGE_TOL, "hydrophobicity_tolerance": HYDROPHOBICITY_TOL,
        "max_identity_to_benchmark": MAX_IDENTITY, "max_shuffle_identity": MAX_SHUFFLE_IDENTITY,
        "benchmark_sha256": sha256(args.benchmark), "proteins_sha256": sha256(args.proteins), "decoys_sha256": sha256(args.out),
        "decoys": len(decoys), "positives": int(len(positives)),
        "positives_short_of_target": {kind: [i for i, n in short[kind].items() if n < PER_TYPE] for kind in short.columns},
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(decoys.groupby("type").size().to_string())
    print("short of target:", manifest["positives_short_of_target"])


if __name__ == "__main__":
    main()
