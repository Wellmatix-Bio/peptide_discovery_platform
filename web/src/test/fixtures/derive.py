"""Build the `derived/` fixtures from the captured ones, by a stated transformation.

    .venv/bin/python web/src/test/fixtures/derive.py

Three UI behaviours no captured run exercises -- a withheld value, an unranked candidate, and a
run Vertex finished without the worker writing a result. Rather than hand-write a response and
pretend it is evidence, each derived file is produced here from a real one, records what it came
from, and says what was changed. See README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "derived"
SOURCE_RESULTS = "results-9223232270029029376.json"
SOURCE_STATUS = "status-9223232270029029376.json"


def write(name: str, body: dict, came_from: str, changed: str) -> None:
    body = dict(body)
    body["_derived_from"] = {"file": came_from, "changed": changed}
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote derived/{name}")


def main() -> None:
    results = json.loads((HERE / SOURCE_RESULTS).read_text(encoding="utf-8"))
    status = json.loads((HERE / SOURCE_STATUS).read_text(encoding="utf-8"))

    # 1. Withheld values, and an unranked candidate. The API's own counts are adjusted to match,
    #    because that is what it would report -- ranked_candidates counts ranking.status == ranked.
    partial = json.loads(json.dumps(results))
    candidates = partial["candidates"]
    for field in ("amp_probability", "hemolytic_activity_phc50", "solubility", "sequence"):
        candidates[0][field] = None
    candidates[0]["log_mic_um"] = {}
    candidates[1]["ranking"] = None
    partial["ranked_candidates"] = sum(1 for c in candidates if c["ranking"] is not None)
    partial["insufficient_evidence_candidates"] = len(candidates) - partial["ranked_candidates"]
    write(
        "results-with-withheld-and-unranked.json",
        partial,
        SOURCE_RESULTS,
        "candidate[0]: amp_probability, hemolytic_activity_phc50, solubility and sequence set to"
        " null and log_mic_um emptied; candidate[1].ranking set to null; ranked_candidates and"
        " insufficient_evidence_candidates recounted to match.",
    )

    # 2. Two candidates sharing a position.
    tied = json.loads(json.dumps(results))
    tied["candidates"][1]["ranking"] = tied["candidates"][0]["ranking"]
    write(
        "results-with-a-tie.json",
        tied,
        SOURCE_RESULTS,
        "candidate[1].ranking set to candidate[0]'s, so two candidates share one position.",
    )

    # 3. The case the whole run view exists to get right: Vertex finished, the worker never wrote
    #    results.json, so the API reports "pending" forever.
    abandoned = dict(status)
    abandoned["status"] = "pending"
    abandoned["stage"] = "pending"
    abandoned["vertex_state"] = "JOB_STATE_FAILED"
    abandoned["error"] = "The replica workerpool0-0 exited with a non-zero status of 1."
    write(
        "status-worker-died.json",
        abandoned,
        SOURCE_STATUS,
        "status and stage forced to 'pending' with vertex_state JOB_STATE_FAILED and a Vertex"
        " error message: a worker that died before writing results.json.",
    )

    # 4. The one the "abandoned" state exists for: Vertex SUCCEEDED, but the worker wrote no
    #    results.json, so the API reports "pending" FOREVER and nothing else ever contradicts it.
    #    (A Vertex FAILURE is a different and less dangerous case -- see status-worker-died.json,
    #    where Vertex says so plainly and carries an error message.)
    silent = dict(status)
    silent["status"] = "pending"
    silent["stage"] = "pending"
    silent["vertex_state"] = "JOB_STATE_SUCCEEDED"
    silent["error"] = None
    write(
        "status-succeeded-no-result.json",
        silent,
        SOURCE_STATUS,
        "status and stage forced to 'pending' while vertex_state stays JOB_STATE_SUCCEEDED and"
        " error is null: Vertex finished cleanly but no results.json was written, so the API"
        " reports pending indefinitely.",
    )

    # 5. Still running.
    running = dict(status)
    running["status"] = "running"
    running["stage"] = "s06_functional_models"
    running["vertex_state"] = "JOB_STATE_RUNNING"
    write(
        "status-running.json",
        running,
        SOURCE_STATUS,
        "status 'running' at stage s06_functional_models with vertex_state JOB_STATE_RUNNING.",
    )


if __name__ == "__main__":
    main()
