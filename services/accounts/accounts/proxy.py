"""The passthrough to the peptide job API, and the per-user history read side.

DELIBERATELY DUMB: forward method, path, query, body and a small header allow-list; return the
response unchanged. No upstream schema is restated here, so nothing can drift from the API's
contract. `Authorization`, `Cookie` and `Host` are never forwarded - a backend with no auth
concept has no business receiving them.

THREE NARROW EXCEPTIONS, because "private per user" cannot be delivered by an upstream that has
no idea what a user is. Each is a SHAPE match, not a copy of the upstream route table, so a
route added upstream later is still covered:

1. Ownership. Any Vertex Custom Job resource name appearing in the path or the query must be
   owned by the caller, or the answer is 404 - the SAME 404 whether or not the job exists, so
   the proxy is not an existence oracle. This is what makes privacy real: the API's
   /status, /results and /cancel act on a job from its id alone, so without this check any
   signed-in user could read or cancel any other user's run.
2. History by observation - see history.py.
3. request_id is REPLACED, not merely prefixed. The API interpolates request_id straight into a
   GCS path (`<artifacts>/requests/<request_id>/config.json`) with no normalization, so a
   caller-chosen value can carry `..` out of the intended prefix, and two callers choosing the
   same value overwrite each other's staging config. This proxy therefore generates the value
   itself as `u<user_id>:<uuid4>` and keeps whatever the user typed as a local run name that
   never leaves this database. The prefix is stripped from responses on the way out.

Refused outright: the API's own interactive docs (`/docs`, `/redoc`, `/openapi.json`, all live
on the upstream by FastAPI default) and its root. The API has no global listing endpoint to
refuse; if one is ever added, add it to REFUSED.
"""

from __future__ import annotations

import json
import posixpath
import re
import uuid
from typing import Any
from urllib.parse import parse_qsl, quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from . import history
from .auth import current_user
from .db import User

SERVICES = ("peptide",)
#: The full Vertex Custom Job resource name, which is the only job id the API accepts
#: (api.py's job_name() rejects anything else with a 422). Matched anywhere in a path or a
#: query value rather than per segment, because unlike a `pred_...` style id this one spans
#: four segments and cannot be recognised a segment at a time.
JOB_ID = re.compile(r"projects/[^/]+/locations/[^/]+/customJobs/[0-9]+")
REQUEST_HEADERS = frozenset({"content-type", "accept", "accept-language"})
RESPONSE_HEADERS = frozenset({"content-type", "content-disposition", "cache-control"})
# Path prefixes (whole segments) that are never forwarded, per upstream.
REFUSED = {
    "peptide": ("", "docs", "redoc", "openapi.json"),
}
NOT_FORWARDED = "not available through this proxy: {service} /{path}"
NOT_YOURS = (
    "no job {job_id} under your account. Runs are private to the account that created them"
)
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]

# A second gated router. The gate is on the ROUTER, exactly as in auth.py: nothing registered
# here can be reached without a valid session.
router = APIRouter(dependencies=[Depends(current_user)], tags=["signed in"])


def namespace(user_id: int) -> str:
    return f"u{user_id}:"


def new_request_id(user_id: int) -> str:
    """Unique per submission and per user, and free of anything a GCS path would treat
    specially. See exception 3 in the module docstring."""
    return f"{namespace(user_id)}{uuid.uuid4().hex}"


def _refused(service: str, path: str) -> bool:
    for prefix in REFUSED[service]:
        if prefix == "" and path == "":
            return True
        if prefix and (path == prefix or path.startswith(prefix + "/")):
            return True
    return False


def _normalised(path: str) -> str:
    """Refuse traversal outright rather than resolving it: `..` has no honest use here."""
    parts = path.split("/")
    if any(part in (".", "..") for part in parts):
        raise HTTPException(400, "path segments '.' and '..' are not forwarded")
    return posixpath.normpath("/" + path).lstrip("/") if path else ""


def _job_ids(path: str, query: str) -> set[str]:
    found = set(JOB_ID.findall(path))
    for _, value in parse_qsl(query, keep_blank_values=True):
        found |= set(JOB_ID.findall(value))
    return found


def _rewritten_body(body: bytes, content_type: str, user_id: int) -> tuple[bytes, str | None, str | None]:
    """Replace a submitted request_id with a generated one.

    Returns the body to forward, the request_id sent upstream, and the name the user typed (to
    be kept locally). Anything that is not a JSON object is left for the upstream to refuse.
    """
    if not body or "json" not in content_type.lower():
        return body, None, None
    try:
        data = json.loads(body)
    except ValueError:
        return body, None, None
    if not isinstance(data, dict) or "request_id" not in data:
        return body, None, None
    typed = data["request_id"]
    run_name = typed.strip() if isinstance(typed, str) and typed.strip() else None
    generated = new_request_id(user_id)
    data["request_id"] = generated
    return json.dumps(data).encode("utf-8"), generated, run_name


def _strip(value: Any, prefix: str) -> Any:
    if isinstance(value, str):
        if value.startswith(prefix):
            value = value[len(prefix):]
        return value.replace(f"'{prefix}", "'").replace(f'"{prefix}', '"')
    if isinstance(value, list):
        return [_strip(one, prefix) for one in value]
    if isinstance(value, dict):
        return {key: _strip(one, prefix) for key, one in value.items()}
    return value


def _unnamespaced(content: bytes, content_type: str, user_id: int) -> bytes:
    """Remove this user's prefix. Bytes are passed through UNCHANGED unless the prefix is in
    them. The API echoes request_id in its create response and embeds it in result_path."""
    prefix = namespace(user_id)
    if prefix.encode("utf-8") not in content or "json" not in content_type.lower():
        return content
    try:
        return json.dumps(_strip(json.loads(content), prefix)).encode("utf-8")
    except ValueError:
        return content


@router.api_route("/api/{service}/{path:path}", methods=METHODS, include_in_schema=False)
async def forward(
    service: str, path: str, request: Request, user: User = Depends(current_user)
) -> Response:
    if service not in SERVICES:
        raise HTTPException(404, f"no upstream named {service!r}")
    path = _normalised(path)
    if _refused(service, path):
        raise HTTPException(404, NOT_FORWARDED.format(service=service, path=path))

    accounts = request.app.state.accounts
    query = request.url.query
    for job_id in sorted(_job_ids(path, query)):
        if not accounts.db.owns(user.id, job_id):
            raise HTTPException(404, NOT_YOURS.format(job_id=job_id))

    content_type = request.headers.get("content-type", "")
    body = await request.body()
    body, sent_request_id, run_name = _rewritten_body(body, content_type, user.id)
    headers = {k: v for k, v in request.headers.items() if k.lower() in REQUEST_HEADERS}

    client: httpx.AsyncClient = request.app.state.upstream_clients[service]
    try:
        upstream = await client.request(
            request.method,
            "/" + quote(path, safe="/:@-._~"),
            params=query or None,
            content=body or None,
            headers=headers,
        )
    except httpx.TimeoutException:
        raise HTTPException(504, f"the {service} service did not answer in time") from None
    except httpx.HTTPError as problem:
        raise HTTPException(
            502, f"the {service} service could not be reached ({type(problem).__name__})"
        ) from None

    response_type = upstream.headers.get("content-type", "")
    content = upstream.content
    # Observed BEFORE the prefix is stripped, so ownership and history see what upstream said.
    # No return value is read: recording cannot change what the caller receives.
    history.observe(
        accounts.db,
        user_id=user.id,
        service=service,
        status_code=upstream.status_code,
        content_type=response_type,
        content=content,
        request_id=sent_request_id,
        run_name=run_name,
        max_envelope_bytes=request.app.state.settings.max_kept_envelope_bytes,
    )
    content = _unnamespaced(content, response_type, user.id)
    return Response(
        content=content,
        status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items() if k.lower() in RESPONSE_HEADERS},
    )


# --- history, read side only. There is deliberately no route that CREATES a row. ------------------

#: Never returned to a client: the namespaced request_id is an internal detail, and echoing it
#: would put the `u<id>:` prefix the rest of this module strips back into a response.
PRIVATE_HISTORY_FIELDS = ("request_id", "user_id")

STALENESS = (
    "status, stage and vertex_state are as of observed_at, not as of this read. This service"
    " records what it saw pass through for your account; it does not poll the pipeline on your"
    " behalf. A run you stop watching keeps the last state you saw."
)


@router.get("/auth/history")
def list_history(
    request: Request,
    user: User = Depends(current_user),
    service: str | None = Query(default=None, pattern="^(peptide)$"),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    total, rows = request.app.state.accounts.db.history(
        user.id, service=service, limit=limit, offset=offset
    )
    for row in rows:
        row["has_envelope"] = bool(row["has_envelope"])
    return {
        "total": total,
        "returned": len(rows),
        "limit": limit,
        "offset": offset,
        "entries": rows,
        "note": (
            "Recorded by observing responses the proxy forwarded for this account. No endpoint"
            " creates an entry, so every entry is a run the API actually accepted."
        ),
        "staleness": STALENESS,
    }


@router.get("/auth/history/{entry_id}")
def history_entry(
    entry_id: int, request: Request, user: User = Depends(current_user)
) -> dict[str, Any]:
    row = request.app.state.accounts.db.history_entry(user.id, entry_id)
    if row is None:
        raise HTTPException(404, f"no history entry {entry_id} under your account")
    for field in PRIVATE_HISTORY_FIELDS:
        row.pop(field, None)
    envelope = row.pop("envelope")
    row["envelope"] = None if envelope is None else json.loads(envelope)
    row["staleness"] = STALENESS
    return row


@router.delete("/auth/history/{entry_id}")
def hide_history_entry(
    entry_id: int, request: Request, user: User = Depends(current_user)
) -> dict[str, Any]:
    """Remove a run from this account's history list.

    NOT a delete of the run: the run's config, audit log and results stay in GCS, which this
    service has no access to and no business touching. Callers must present this as "hide from
    my history" so nobody believes their data was erased. This is the one history route that
    writes, and it can only ever remove one of the caller's own rows -- it cannot create one.
    """
    if not request.app.state.accounts.db.hide_run(user.id, entry_id):
        raise HTTPException(404, f"no history entry {entry_id} under your account")
    return {
        "hidden": entry_id,
        "artifacts_deleted": False,
        "why": (
            "Only this account's history entry was removed. The run's artifacts in Cloud"
            " Storage are untouched; this service cannot delete them."
        ),
    }
