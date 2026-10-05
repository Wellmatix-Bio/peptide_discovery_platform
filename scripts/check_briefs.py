"""How many of data/briefs/*.json are valid API requests? Reports which, and why the rest fail.

The answer has been 0 or 1 for the life of this repository; see docs/BRIEF_VALIDITY.md. This
script exists so the number is measured rather than quoted, because the vocabularies in
src/schemas/stage_configs.py change and a figure in a document does not.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from schemas.stage_configs import BriefFields  # noqa: E402

BRIEFS = pathlib.Path(__file__).resolve().parents[1] / "data/briefs"


def main() -> int:
    files = sorted(BRIEFS.glob("*.json"))
    if not files:
        print(f"No briefs found in {BRIEFS}", file=sys.stderr)
        return 2

    valid, rejected = [], collections.Counter()
    for path in files:
        try:
            BriefFields.model_validate(json.loads(path.read_text(encoding="utf-8")))
            valid.append(path.name)
        except Exception as problem:  # noqa: BLE001 - any validation failure counts
            for line in str(problem).splitlines():
                if "Input should be" in line or "Field required" in line:
                    rejected[line.strip()[:90]] += 1

    print(f"{len(valid)} of {len(files)} briefs validate")
    for name in valid:
        print(f"  VALID  {name}")
    if rejected:
        print("\nMost frequent rejection reasons:")
        for reason, count in rejected.most_common(5):
            print(f"  {count:3}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
