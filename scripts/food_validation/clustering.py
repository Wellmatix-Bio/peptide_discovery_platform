"""Validation of protein-family assignment (src/pipeline/food/protein_library/clusters.py) against the data foundation's real CD-HIT output.

The data foundation clustered its 524 digestion substrates with CD-HIT at 100, 95, 90, 80, 70, 60 and 50% identity (`evaluation/digestion_substrates_c*.clstr`) and
classified each substrate into one of 21 curated protein families (`families` column of `digestion_substrates_clusters.csv`). This script:
1. parses the real `.clstr` files and checks the partition equals the table's cluster column at every level (this validates the reader used here, not the platform, which clusters with MMseqs2);
2. measures how well each identity level reproduces the curated families (homogeneity, completeness, adjusted Rand index);
3. shows what the per-cluster cap of the ranking allows one family to take at each level.
It does not execute MMseqs2.

    python scripts/food_validation/clustering.py [--evaluation-dir DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # model_store, imported by the pipeline package

from pipeline.food import settings  # noqa: E402
from pipeline.food.protein_library.clusters import IDENTITY  # noqa: E402

def _read_cdhit(path: Path) -> dict[str, str]:
    """member -> representative from CD-HIT's `.clstr` (the representative is the line ending in `*`). Reads the data foundation's reference files; the platform clusters with MMseqs2."""
    out, members, rep = {}, [], None
    for line in path.read_text(encoding="utf-8").splitlines() + [">Cluster end"]:
        if line.startswith(">Cluster"):
            out.update({m: rep for m in members if rep is not None})
            members, rep = [], None
            continue
        m = re.search(r">(.+?)\.\.\.", line)
        if m:
            members.append(m.group(1))
            if line.rstrip().endswith("*"):
                rep = m.group(1)
    return out


EVALUATION = ROOT.parent / "datasets" / "Food-Derived Metabolic Wellness Peptide" / "evaluation"
LEVELS = (100, 95, 90, 80, 70, 60, 50)
FINAL_COUNT, CAP = settings.FINAL_COUNT, settings.MAX_PER_PARENT_CLUSTER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "validation" / "food")
    args = parser.parse_args()
    table = pd.read_csv(args.evaluation_dir / "digestion_substrates_clusters.csv")
    families = table.families.tolist()
    sizes = Counter(families)
    result: dict = {"substrates": len(table), "families": len(sizes), "family_sizes": dict(sizes.most_common()), "platform_identity": IDENTITY, "levels": {}}
    for level in LEVELS:
        parsed = _read_cdhit(args.evaluation_dir / f"digestion_substrates_c{level}.clstr")
        rep_of = [parsed[i] for i in table.id]  # the parser's representative per substrate
        column = table[f"c{level}"].tolist()
        # same partition <=> the two labelings give an adjusted Rand index of exactly 1
        same = adjusted_rand_score(column, rep_of) == 1.0
        per_family = pd.DataFrame({"family": families, "cluster": rep_of}).groupby("family").cluster.nunique()
        result["levels"][level] = {
            "clusters": len(set(rep_of)), "parser_matches_table_column": bool(same),
            "homogeneity": round(homogeneity_score(families, rep_of), 3),  # 1: no cluster mixes families
            "completeness": round(completeness_score(families, rep_of), 3),  # 1: no family is split over clusters
            "adjusted_rand_index": round(adjusted_rand_score(families, rep_of), 3),
            "families_in_one_cluster": int((per_family == 1).sum()),
            "median_clusters_per_family": float(per_family.median()), "max_clusters_per_family": int(per_family.max()),
            "largest_family_clusters": {per_family.idxmax(): int(per_family.max())},
            "slots_one_family_could_take": min(FINAL_COUNT, CAP * int(per_family.max())),
            "families_that_could_fill_all_slots": int((per_family * CAP >= FINAL_COUNT).sum()),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "clustering_validation.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'identity':>8} {'clusters':>8} {'parser=table':>12} {'homog':>6} {'compl':>6} {'ARI':>6} {'fam in 1 cluster':>17} {'max cl/fam':>10} {'fam able to fill 25':>20}")
    for level, v in result["levels"].items():
        print(f"{level:>7}% {v['clusters']:>8} {str(v['parser_matches_table_column']):>12} {v['homogeneity']:>6} {v['completeness']:>6} {v['adjusted_rand_index']:>6} "
              f"{v['families_in_one_cluster']:>14}/{result['families']} {v['max_clusters_per_family']:>10} {v['families_that_could_fill_all_slots']:>20}")


if __name__ == "__main__":
    main()
