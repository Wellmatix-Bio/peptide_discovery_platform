"""Runs the screening benchmark (stages 5-11) and writes report.md plus CSVs.

    python -I scripts/run_screening_benchmark.py --out artifacts/benchmark/screening            # full run
    python -I scripts/run_screening_benchmark.py --audit-data-only                              # data audit, no models
    python -I scripts/run_screening_benchmark.py --check                                        # CI: fail on regression
    python -I scripts/run_screening_benchmark.py --update-baseline

Exit codes: 0 ok, 1 regression against the baseline, 2 the run could not complete, 3 no baseline to compare against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from screening_benchmark import report, runner  # noqa: E402

BASELINE = ROOT / "tests" / "benchmark" / "screening_baseline.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "benchmark" / "screening.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "benchmark" / "screening")
    ap.add_argument("--audit-data-only", action="store_true", help="load and audit the datasets, run no models")
    ap.add_argument("--audit-file", type=Path, help="rejections_for_audit.csv with expert_verdict filled in")
    ap.add_argument("--skip-stability", action="store_true")
    ap.add_argument("--skip-length-rerun", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare with the stored baseline; non-zero exit on regression")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--auc-tolerance", type=float, default=0.02)
    args = ap.parse_args()

    cfg = runner.load_config(args.config)
    if args.audit_data_only:
        peptides, audit = runner.load_peptides(cfg)
        args.out.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.out / "data_audit.csv", index=False)
        lengths = {b: tuple(cfg["branches"][b]["length_range"]) for b in runner.data.BRANCHES}
        print(runner.data.applicability_summary(peptides, lengths).to_string(index=False))
        print(audit.groupby(["source", "role", "reason"]).size().to_string())
        return 0

    try:
        outcome = runner.run_benchmark(
            cfg, runner.real_factories(args.out, seed=cfg["seed"]),
            runner.Options(args.skip_stability, args.skip_length_rerun, args.audit_file),
        )
    except Exception as exc:  # noqa: BLE001 — a missing weight file or GPU should be a clear non-zero exit
        print(f"benchmark could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"report: {report.write_outputs(outcome, args.out, cfg)}")

    if args.update_baseline:
        BASELINE.write_text(json.dumps(outcome.baseline_metrics, indent=2, sort_keys=True))
        print(f"baseline written: {BASELINE}")
        return 0
    if args.check:
        if not BASELINE.exists():
            print(f"no baseline at {BASELINE}; run with --update-baseline once the models are available", file=sys.stderr)
            return 3
        problems = runner.compare_to_baseline(json.loads(BASELINE.read_text()), outcome.baseline_metrics, args.auc_tolerance)
        for problem in problems:
            print(f"REGRESSION {problem}", file=sys.stderr)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
