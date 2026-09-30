"""What a user ran, written BY OBSERVATION of responses the proxy forwarded.

- NO ENDPOINT CREATES A HISTORY ROW. A client cannot claim work it did not do: getting a row
  written requires causing the proxy to forward a request and receive an envelope, which IS
  submitting the run. It also cannot be forgotten if the tab closes.
- MATCH ON ENVELOPE SHAPE, NOT REQUEST PATH. Path matching means keeping a copy of the upstream
  route table and silently missing routes added later.
- RECORDING CANNOT CHANGE THE RESPONSE. `observe` swallows every failure, and its caller does not
  read a return value because there is none. Storage trouble never turns an accepted,
  already-submitted run into an error the caller would read as a rejection.

- A ROW IS NOT TERMINAL WHEN WRITTEN. This is the one structural difference from the synchronous
  service this was adapted from. A create response only means Vertex accepted the job; the run
  itself takes minutes to hours. So the create observation opens the row and later status and
  results observations update it, each moving observed_at. Nothing here polls: a run is only as
  current as the last response its owner passed through the proxy.

- VERTEX_STATE IS RECORDED SEPARATELY FROM STATUS, and deliberately so. The API derives `status`
  only from the worker's results.json, so a worker that dies without writing that file reports
  "pending" forever while Vertex says JOB_STATE_FAILED. Storing both is what lets the run view
  avoid telling somebody a dead run is still working. See docs/BASELINE.md.

Ownership (who may address a job id) is recorded here too, by the same rule and for the same
reason: it can only be learned from a response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .db import Database

log = logging.getLogger("accounts.history")

#: Vertex states that mean the job will not progress any further.
TERMINAL_VERTEX_STATES = frozenset(
    {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
)


def is_created_job(body: dict[str, Any]) -> bool:
    """The create envelope: CreateJobResponse (request_id, job_id, result_path)."""
    return (
        isinstance(body.get("job_id"), str)
        and isinstance(body.get("request_id"), str)
        and isinstance(body.get("result_path"), str)
    )


def is_job_status(body: dict[str, Any]) -> bool:
    """The status envelope: JobStatusResponse. `vertex_state` is the field that distinguishes it
    from the create envelope; `status` alone would not, since results carries no job_id."""
    return isinstance(body.get("job_id"), str) and isinstance(body.get("vertex_state"), str)


def is_job_results(body: dict[str, Any]) -> bool:
    """The results envelope: JobResultsResponse (run_id, n_final, ranked_candidates)."""
    return (
        isinstance(body.get("run_id"), str)
        and "n_final" in body
        and isinstance(body.get("ranked_candidates"), list)
    )


def _status_summary(body: dict[str, Any]) -> str:
    status = body.get("status") or "pending"
    vertex_state = body.get("vertex_state") or "unknown"
    stage = body.get("stage") or "pending"
    summary = f"{status} at {stage} (Vertex: {vertex_state})"
    # Say it in the summary too, not only in the fields, so a history list that shows nothing but
    # summaries still cannot imply a dead run is progressing.
    if status == "pending" and vertex_state in TERMINAL_VERTEX_STATES:
        summary += " -- reported pending, but Vertex has finished; the worker wrote no result"
    return summary


def _results_summary(body: dict[str, Any]) -> str:
    ranked = body.get("ranked_candidates") or []
    n_final = body.get("n_final")
    return f"{n_final if n_final is not None else '?'} final candidate(s), {len(ranked)} ranked"


def _kept(content: bytes, max_envelope_bytes: int) -> tuple[str | None, str | None]:
    if len(content) <= max_envelope_bytes:
        return content.decode("utf-8"), None
    return None, (
        f"response of {len(content)} bytes was not kept (limit {max_envelope_bytes});"
        " reopen the run to fetch it again"
    )


def observe(
    db: Database,
    *,
    user_id: int,
    service: str,
    status_code: int,
    content_type: str,
    content: bytes,
    max_envelope_bytes: int,
    request_id: str | None = None,
    run_name: str | None = None,
) -> None:
    """Record what was seen. Returns nothing and raises nothing, by construction."""
    try:
        if "json" not in content_type.lower() or not content:
            return
        body = json.loads(content)
        if not isinstance(body, dict):
            return

        if is_created_job(body) and status_code < 300:
            job_id = body["job_id"]
            # A later GET of one's own job is a read, not work: only the FIRST sighting of an id
            # becomes a history row, and that sighting is the create.
            if db.claim(job_id, user_id, service):
                db.add_history(
                    user_id=user_id,
                    service=service,
                    kind="peptide.run",
                    status_code=status_code,
                    summary="submitted, awaiting the worker's first progress write",
                    resource_id=job_id,
                    request_id=request_id or body.get("request_id"),
                    run_name=run_name,
                    status="pending",
                    stage="pending",
                )
        elif is_job_status(body) and status_code < 300:
            db.update_run(
                user_id=user_id,
                resource_id=body["job_id"],
                status=body.get("status"),
                stage=body.get("stage"),
                vertex_state=body.get("vertex_state"),
                summary=_status_summary(body),
            )
        elif is_job_results(body):
            # A run id is the numeric tail of the job resource name; the results envelope carries
            # only that, so the row is found by suffix rather than by the full name.
            resource_id = db.resource_id_for_run(user_id, body["run_id"])
            if resource_id is None:
                return
            # 200 is an answer; so is a refusal the API gives about a real run. Both are recorded,
            # but only a success is worth keeping a body for.
            envelope, note = (
                _kept(content, max_envelope_bytes) if status_code < 300 else (None, None)
            )
            db.update_run(
                user_id=user_id,
                resource_id=resource_id,
                summary=_results_summary(body) if status_code < 300 else None,
                envelope=envelope,
                envelope_note=note,
            )
    except Exception:  # noqa: BLE001 - recording must never reach the response
        log.exception("history: failed to record an observed response; the response is unaffected")
