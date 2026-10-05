"""Write, or check, the OpenAPI snapshot of the peptide job API.

    .venv/bin/python web/scripts/openapi.py peptide           # write web/openapi/peptide.json
    .venv/bin/python web/scripts/openapi.py peptide --check    # exit 1 if the snapshot is stale

Taken from the FastAPI app itself, NOT from the repository's openapi.yaml. The React types are
generated from this snapshot (`npm run api:types`), so a field renamed in api.py becomes a
TypeScript compile error rather than an `undefined` rendered in a browser -- and that only holds
if the snapshot reflects the code. openapi.yaml is hand-maintained and was already found to
disagree with the code (docs/BASELINE.md), so it cannot be the source of truth here.

Importing api.py does NOT contact Google: its Vertex client is built lazily inside job_client().
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGETS = {"peptide": "backend.api_e2e.api"}


def render(name: str) -> str:
    sys.path.insert(0, str(REPO / "src"))
    module = importlib.import_module(TARGETS[name])
    return json.dumps(module.app.openapi(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in TARGETS:
        print(f"usage: openapi.py {{{'|'.join(TARGETS)}}} [--check]", file=sys.stderr)
        return 2
    name, check = argv[0], "--check" in argv
    path = REPO / "web" / "openapi" / f"{name}.json"
    fresh = render(name)
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != fresh:
            print(
                f"{path} is stale: run  .venv/bin/python web/scripts/openapi.py {name}",
                file=sys.stderr,
            )
            return 1
        print(f"{path.name} is current")
        return 0
    path.write_text(fresh, encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
