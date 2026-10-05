"""Fail if the backend suite's result has moved from the recorded baseline.

The backend suite cannot be green: 4 tests fail on genuine disagreements between api.py and its
tests, and 3 need model weights that are gitignored and synced at run time (docs/BASELINE.md). A
CI job that simply required a zero exit code would therefore always fail, and one that ignored
the exit code would notice nothing.

So this compares the OUTCOME to the recorded numbers. Any movement in either direction fails --
a new failure is a regression, and a new pass means the baseline is stale and docs/BASELINE.md
needs updating. Both deserve a human's attention.

    DEV_MODE=true python .github/check_baseline.py
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys

#: From docs/BASELINE.md. Update BOTH together, never just this.
EXPECTED = {"passed": 105, "failed": 7, "errors": 0}

#: THE BASELINE IS A PROPERTY OF AN ENVIRONMENT, NOT JUST OF THE CODE. A module the suite imports
#: but that is not installed turns tests into collection ERRORS, and the count moves exactly as it
#: would for a real regression -- which is misleading in precisely the moment you need to trust it.
#: These are the imports the recorded numbers assume. They are checked first so a missing one is
#: reported as what it is, instead of being counted as a regression.
#:
#: This list was once deliberately short, on the grounds that "nothing here needs torch,
#: transformers, sklearn or xgboost, because the 47 tests that would are uncollectable anyway
#: (s4pred)". Making s4pred optional removed that barrier and those 47 tests now collect and
#: import the real ML stack -- so the short list no longer described the baseline's environment,
#: and a CI job installing only part of it reported 34/6/2 as a regression. The guard failed in
#: exactly the way it exists to prevent, because its premise had been changed out from under it.
#:
#: So: everything the collectable suite imports, deep enough to catch a partial install. Keep it
#: in step with the python job in .github/workflows/ci.yml.
ASSUMED_IMPORTS = (
    "fastapi",
    "pydantic",
    "yaml",
    "google.cloud.storage",
    "structlog",
    "pandas",
    "numpy",
    "torch",
    "transformers",
    "sklearn",
    "xgboost",
    "Bio",
    "modlamp",
    "deap",
    "freesasa",
    "joblib",
    "rapidfuzz",
)

#: The baseline also assumes MODEL WEIGHTS ARE ABSENT. They are gitignored and synced at run time,
#: so CI and a fresh clone never have them -- but a developer who has synced them locally will see
#: three stage-4 tests pass that cannot pass here, and the count will move. That is a better
#: environment, not a regression, and the message below says so rather than leaving someone to
#: guess. The two optional copyleft dependencies (propy3, s4pred) are likewise assumed ABSENT;
#: installing them also makes more tests runnable. See docs/LICENSING.md.
WEIGHTS_DIR = "model_store/model_weights"


def missing_imports() -> list[str]:
    absent = []
    for name in ASSUMED_IMPORTS:
        try:
            if importlib.util.find_spec(name) is None:
                absent.append(name)
        except (ImportError, ValueError):
            absent.append(name)
    return absent


def main() -> int:
    from pathlib import Path

    if Path(WEIGHTS_DIR).is_dir():
        print(
            f"{WEIGHTS_DIR} exists, but the recorded baseline was measured without model"
            " weights.\nTests that need them will pass here and cannot in CI, so the count will"
            " not match.\nThis is a richer environment, not a regression; compare against a"
            " clean checkout before concluding anything.",
            file=sys.stderr,
        )
        return 2

    absent = missing_imports()
    if absent:
        print(
            "This environment is missing modules the recorded baseline assumes:\n"
            f"  {', '.join(absent)}\n\n"
            "Those would be counted as collection ERRORS and look identical to a regression.\n"
            "Install them and run again; see the python job in .github/workflows/ci.yml.",
            file=sys.stderr,
        )
        return 2

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
