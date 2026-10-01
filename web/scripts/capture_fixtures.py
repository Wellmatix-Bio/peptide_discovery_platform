"""Capture fixtures from the RUNNING job API, for the web app's tests.

    .venv/bin/python web/scripts/capture_fixtures.py --api http://127.0.0.1:8090

Why this exists: the spec requires result views to be tested against real responses captured from
a running API, never hand-written ones. A hand-written fixture encodes what the author BELIEVES
the API returns, so a test built on it passes while the app misrenders the real thing -- and this
API has already been found to differ from its own documentation more than once (docs/BASELINE.md).

What it does NOT do:
- It never submits a job. Every run it reads already exists; nothing here costs GPU time.
- It writes no credentials, bucket names, project ids or service accounts into the fixtures. The
  job resource name identifies a project, so it is REWRITTEN to a placeholder, and the few other
  fields that carry deployment identifiers go with it. The sequences and scores are the point;
  the infrastructure is not, and these files are committed.

Run it against an API that has working Application Default Credentials. In practice that is a
local one (`uvicorn backend.api_e2e.api:app --app-dir src`), not the compose container, which has
no credentials unless GOOGLE_CREDENTIALS_FILE is set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "src" / "test" / "fixtures"

#: Replaces the real project and location so a committed fixture names no infrastructure.
PLACEHOLDER_JOB = "projects/example-project/locations/us-central1/customJobs/{run_id}"
JOB_NAME = re.compile(r"projects/[^/\"]+/locations/[^/\"]+/customJobs/(\d+)")
#: gs:// paths carry the bucket name.
GCS_PATH = re.compile(r"gs://[^\"\s]+")


def fetch(api: str, path: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(f"{api}{path}", timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as problem:
        return problem.code, json.loads(problem.read() or b"null")


def scrub(value: object) -> object:
    """Replace deployment identifiers, in place, at any depth."""
    if isinstance(value, str):
        value = JOB_NAME.sub(lambda m: PLACEHOLDER_JOB.format(run_id=m.group(1)), value)
        return GCS_PATH.sub("gs://example-bucket/artifacts", value)
    if isinstance(value, list):
        return [scrub(one) for one in value]
    if isinstance(value, dict):
        return {key: scrub(one) for key, one in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8090")
    parser.add_argument("--project", required=True, help="GCP project id, for building job names")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("runs", nargs="+", help="numeric Vertex job ids of runs that already exist")
    args = parser.parse_args()

    FIXTURES.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    status_code, manifest = fetch(args.api, "/api/v1/models")
    if status_code == 200:
        (FIXTURES / "models.json").write_text(
            json.dumps(scrub(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append("models.json")

    status_code, health = fetch(args.api, "/healthz")
    if status_code == 200:
        (FIXTURES / "healthz.json").write_text(
            json.dumps(scrub(health), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append("healthz.json")

    for run_id in args.runs:
        job = f"projects/{args.project}/locations/{args.location}/customJobs/{run_id}"
        for kind in ("status", "results"):
            code, body = fetch(args.api, f"/api/v1/jobs/{job}/{kind}")
            if code != 200:
                print(f"  {run_id} {kind}: HTTP {code}, skipped", file=sys.stderr)
                continue
            name = f"{kind}-{run_id}.json"
            (FIXTURES / name).write_text(
                json.dumps(scrub(body), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            written.append(name)

    for name in written:
        print(f"wrote {FIXTURES / name}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
