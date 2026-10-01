"""Fail if the backend suite's result has moved from the recorded baseline.

The backend suite cannot be green: s4pred is committed as a gitlink with no .gitmodules, so
47 tests never collect, and 4 more fail on genuine disagreements between api.py and its tests
(docs/BASELINE.md). A CI job that simply required a zero exit code would therefore always fail,
and one that ignored the exit code would notice nothing.

So this compares the OUTCOME to the recorded numbers. Any movement in either direction fails --
a new failure is a regression, and a new pass means the baseline is stale and docs/BASELINE.md
needs updating. Both deserve a human's attention.

    DEV_MODE=true python .github/check_baseline.py
"""

from __future__ import annotations

import re
import subprocess
import sys

#: From docs/BASELINE.md. Update BOTH together, never just this.
EXPECTED = {"passed": 29, "failed": 4, "errors": 2}


def main() -> int:
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--continue-on-collection-errors",
         "--ignore=services", "--ignore=web"],
        capture_output=True,
        text=True,
    )
    tail = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
    got = {
        name: int(found.group(1)) if (found := re.search(rf"(\d+) {name}", tail)) else 0
        for name in ("passed", "failed", "errors")
    }
    print(f"baseline expected: {EXPECTED}")
    print(f"this run:          {got}")
    print(f"pytest summary:    {tail}")
    if got == EXPECTED:
        print("\nUnchanged from the recorded baseline.")
        return 0
    print(
        "\nThe backend suite's result MOVED.\n"
        "  More failures  -> a regression was introduced; find it before merging.\n"
        "  More passes    -> good news, but docs/BASELINE.md and EXPECTED here are now stale\n"
        "                    and must be updated in the same change.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
