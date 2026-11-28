"""Benchmark of the platform's protease cleavage rules (src/food/b_digestion/rules.py) against experimentally observed cleavage sites.

The rules are the ones s03 (digestion) and s06 (gut stability) use. Each rule gives a yes/no call per internal bond, so the scores are classification metrics
at that one operating point: sensitivity, specificity, precision, MCC and balanced accuracy (= the AUROC of a binary call).

Data (Phase 1 `Model Validation`, read in place, nothing copied):
- MEROPS mass-spectrometry / N-terminal-sequencing cleavage sites with their parent proteins (trypsin 1, chymotrypsin A, pepsin A);
- porcine-pepsin peptidomics on yeast proteins (Baghalabadi et al., Proteomes 2021), the two label schemes Phase 1 built.
Labels are observed cleavages only, so an unlisted residue is a negative that may be an untested site: precision is understated. 95% intervals are a bootstrap
over substrates (1,000 resamples, seed 0).

    python scripts/food_validation/protease_rules.py [--validation-dir DIR] [--merops-dir DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = Path(r"C:\Users\KCC\OneDrive\Documents\AI Wound Healing Peptide Discovery\Models\Food Peptide GLP1 Discovery\Model Validation")
MEROPS = Path(r"C:\Users\KCC\Downloads\Cleavage site prediction eval\filtered_merops")
SEED, N_BOOT = 0, 1000

# MEROPS protease -> the platform rules that stand for it (the first is the one the pipeline uses).
MEROPS_RULES = {"trypsin 1": ["trypsin"], "chymotrypsin A": ["chymotrypsin_high", "chymotrypsin_low"], "pepsin A": ["pepsin_ph1.3", "pepsin_ph2.0"]}


def load_rules():
    """rules.py imported from its file: the package import would pull in the whole model stack, and the module needs only the standard library."""
    spec = importlib.util.spec_from_file_location("food_rules", ROOT / "src" / "pipeline" / "food" / "digestion" / "rules.py")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module  # dataclasses resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module.BUILTIN_ENZYMES


def read_fasta(path: Path) -> dict[str, str]:
    seqs, key = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            key = line[1:].split()[0]
            seqs[key] = ""
        elif key:
            seqs[key] += line.strip()
    return seqs


def counts(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """tp, fp, fn, tn."""
    return np.array([(pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum(), (~pred & (y == 0)).sum()], dtype=float)


def metrics(c: np.ndarray) -> dict:
    tp, fp, fn, tn = c
    sens, spec = tp / (tp + fn) if tp + fn else float("nan"), tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "balanced_accuracy": (sens + spec) / 2,
            "mcc": (tp * tn - fp * fn) / denom if denom else float("nan"),
            "prevalence": (tp + fn) / c.sum(), "cut_density": (tp + fp) / c.sum()}


def evaluate(per_substrate: list[np.ndarray]) -> dict:
    """Pooled metrics with a bootstrap interval over substrates."""
    table = np.array(per_substrate)
    point = metrics(table.sum(axis=0))
    rng = np.random.default_rng(SEED)
    boots = [metrics(table[rng.integers(0, len(table), len(table))].sum(axis=0)) for _ in range(N_BOOT)]
    out = {"substrates": len(table), "sites": int(table[:, 0].sum() + table[:, 2].sum()), "residues": int(table.sum()), **{k: round(float(v), 4) for k, v in point.items()}}
    for k in ("sensitivity", "specificity", "precision", "balanced_accuracy", "mcc"):
        v = np.array([b[k] for b in boots])
        out[k + "_ci95"] = [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    return out


def merops(rules: dict, validation: Path, merops_dir: Path) -> dict:
    sites = pd.read_csv(validation / "unizyme_unseen_sites_v1.csv")
    parents = read_fasta(merops_dir / "parent_proteins.fasta")
    out: dict = {}
    for protease, rule_names in MEROPS_RULES.items():
        for variant, flag in (("strict", "unseen_strict"), ("all", None)):
            rows = sites[sites.protease == protease]
            rows = rows[rows[flag]] if flag else rows
            skipped = {"no_sequence": 0, "site_outside_sequence": 0, "window_mismatch": 0}
            per_rule: dict[str, list] = {r: [] for r in rule_names}
            for acc, g in rows.groupby("substrate_uniprot"):
                seq = parents.get(acc)
                if not seq:
                    skipped["no_sequence"] += len(g)
                    continue
                pos = set()
                for r in g.itertuples():
                    p = int(r.cleavage_position)
                    window = seq[max(p - 4, 0):p + 4]
                    if not 1 <= p <= len(seq) - 1:
                        skipped["site_outside_sequence"] += 1
                    elif len(window) == 8 and all(a == b for a, b in zip(window, r.p4_p4prime) if b not in "-?XxUuBZ"):
                        pos.add(p)
                    else:
                        skipped["window_mismatch"] += 1  # the MEROPS P4-P4' window disagrees with the sequence: position not trusted
                if not pos:
                    continue
                y = np.zeros(len(seq) - 1, dtype=int)
                y[[p - 1 for p in pos]] = 1
                for name in rule_names:
                    pred = np.zeros(len(seq) - 1, dtype=bool)
                    pred[rules[name].sites(seq)] = True
                    per_rule[name].append(counts(pred, y))
            for name in rule_names:
                out[f"{protease} | {name} | {variant}"] = {**evaluate(per_rule[name]), "skipped_sites": skipped} if per_rule[name] else {"substrates": 0}
    return out


def yeast(rules: dict, validation: Path) -> dict:
    seqs = json.loads((validation / "pepsin_data" / "yeast_sequences.json").read_text(encoding="utf-8"))
    out: dict = {}
    for scheme in ("primary", "secondary"):
        d = pd.read_csv(validation / "pepsin_data" / f"pepsin_yeast_scores_{scheme}.csv")
        for name in MEROPS_RULES["pepsin A"]:
            per = []
            for acc, g in d.groupby("acc"):
                hit = np.zeros(len(seqs[acc]), dtype=bool)
                hit[rules[name].sites(seqs[acc])] = True
                per.append(counts(hit[g.pos.values - 1], g.y.values))
            out[f"pepsin A yeast {scheme} | {name}"] = evaluate(per)
    return out


def terminal_context(rules: dict) -> dict:
    """The most bonds each rule can cut in a peptide of a given length, using a sequence made of its best residue (all F, all K for trypsin).
    A rule that looks at P2, P1' or P2' cannot cut a bond whose context runs off the end of the peptide."""
    out = {}
    for name in sum(MEROPS_RULES.values(), []):
        best = "K" if name == "trypsin" else "F"
        out[name] = {n: {"cuttable_bonds": len(rules[name].sites(best * n)), "internal_bonds": n - 1} for n in (4, 6, 8, 10, 15, 20, 40)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, default=VALIDATION)
    parser.add_argument("--merops-dir", type=Path, default=MEROPS)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "validation" / "food")
    args = parser.parse_args()
    rules = load_rules()
    result = {"rules": {n: {"pattern": rules[n].pattern, "ph_range": rules[n].ph_range} for n in sum(MEROPS_RULES.values(), [])},
              "merops": merops(rules, args.validation_dir, args.merops_dir), "yeast_pepsin": yeast(rules, args.validation_dir),
              "terminal_context": terminal_context(rules)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protease_validation.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for group in ("merops", "yeast_pepsin"):
        for k, v in result[group].items():
            if v.get("substrates"):
                print(f"{k:55s} subs={v['substrates']:5d} sites={v['sites']:5d} sens={v['sensitivity']:.3f} spec={v['specificity']:.3f} prec={v['precision']:.3f} "
                      f"BA={v['balanced_accuracy']:.3f} {v['balanced_accuracy_ci95']} MCC={v['mcc']:.3f} density={v['cut_density']:.3f} prev={v['prevalence']:.4f}")
            else:
                print(f"{k:55s} no usable substrates")


if __name__ == "__main__":
    main()
