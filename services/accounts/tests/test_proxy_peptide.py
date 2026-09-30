"""The three privacy exceptions, against a fake job API that answers in the real envelopes.

A fake rather than the real API, because the real one submits a billable Vertex Custom Job. The
envelope SHAPES here are taken from src/backend/api_e2e/api.py's response models
(CreateJobResponse, JobStatusResponse, JobResultsResponse) -- if those change, these fakes drift
and the proxy's shape matching needs revisiting, which is the point.

What is being protected (see accounts/proxy.py's docstring):
1. Ownership -- /status, /results and /cancel act on a run from its job id alone, so without a
   check any signed-in user could read or cancel anyone's run.
2. History by observation -- no endpoint creates a row, and rows are MUTABLE because a run is
   asynchronous.
3. request_id is replaced, because the API interpolates it into a GCS path unsanitized.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from accounts.app import create_app
from accounts.proxy import namespace

from conftest import bearer, is_gated, register, routes_of

JOB = "projects/p-1/locations/us-central1/customJobs/456"
OTHER_JOB = "projects/p-1/locations/us-central1/customJobs/999"

api = FastAPI()
#: Every request_id the fake upstream was given, so a test can see what the proxy actually sent.
seen_request_ids: list[str] = []
#: Job ids the fake hands out, in order. The FIRST is JOB, so the common case reads plainly;
#: later submissions get distinct ids, which is what lets a test give the same caller one run it
#: owns and one it does not.
issued: list[str] = []


def _next_job_id() -> str:
    return JOB if not issued else f"projects/p-1/locations/us-central1/customJobs/{500 + len(issued)}"


@api.post("/api/v1/jobs/create")
async def create(request: Request) -> Response:
    body = json.loads(await request.body())
    seen_request_ids.append(body["request_id"])
    job_id = _next_job_id()
    issued.append(job_id)
    return Response(
        content=json.dumps(
            {
                "request_id": body["request_id"],
                "job_id": job_id,
                # The API builds this from the artifacts dir and the numeric job id; it does not
                # contain the request_id, but result_path for a staging config does.
                "result_path": "gs://bucket/artifacts/runs/456",
            }
        ),
        status_code=202,
        media_type="application/json",
    )


@api.get("/api/v1/jobs/{job_id:path}/status")
async def status(job_id: str) -> Response:
    # Mirrors the real behaviour documented in docs/BASELINE.md: `status` comes only from the
    # worker's results.json, so it stays "pending" even when Vertex has finished.
    body = {
        "job_id": job_id,
        "status": "pending",
        "stage": "pending",
        "vertex_state": "JOB_STATE_FAILED",
        "error": "worker exited with code 1",
    }
    return Response(content=json.dumps(body), status_code=200, media_type="application/json")


@api.get("/api/v1/jobs/{job_id:path}/results")
async def results(job_id: str) -> Response:
    body = {
        "run_id": job_id.rsplit("/", 1)[-1],
        "n_final": 2,
        "ranked_candidates": [{"id": "c1", "sequence": "KRWWKWIRW"}, {"id": "c2", "sequence": "GIGKFLK"}],
        "component_stats": {},
    }
    return Response(content=json.dumps(body), status_code=200, media_type="application/json")


@api.post("/api/v1/jobs/{job_id:path}/cancel")
async def cancel(job_id: str) -> Response:
    return Response(
        content=json.dumps({"cancelled": job_id}), status_code=200, media_type="application/json"
    )


@pytest.fixture
def proxied(settings):
    seen_request_ids.clear()
    issued.clear()
    app = create_app(settings, transports={"peptide": httpx.ASGITransport(app=api)})
    with TestClient(app) as client:
        token = register(client)["token"]
        yield client, token


def submit(client, token, name="my first run"):
    return client.post(
        "/api/peptide/api/v1/jobs/create",
        headers={**bearer(token), "Content-Type": "application/json"},
        json={"request_id": name, "stages": {}},
    )


# --- exception 3: request_id is replaced ---------------------------------------------------------


def test_the_request_id_sent_upstream_is_generated_not_the_users(proxied):
    client, token = proxied
    response = submit(client, token, name="my first run")
    assert response.status_code == 202, response.text
    sent = seen_request_ids[-1]
    assert sent != "my first run"
    assert sent.startswith(namespace(1))
    # Nothing a GCS path would treat specially, which is the whole reason this exists.
    assert "/" not in sent.removeprefix(namespace(1))
    assert ".." not in sent


def test_a_traversal_request_id_never_reaches_the_api(proxied):
    client, token = proxied
    assert submit(client, token, name="../../../etc/passwd").status_code == 202
    assert ".." not in seen_request_ids[-1]


def test_two_users_submitting_the_same_name_send_different_request_ids(proxied, settings):
    client, token = proxied
    submit(client, token, name="same name")
    second = register(client, email="grace@example.org")["token"]
    submit(client, second, name="same name")
    assert seen_request_ids[0] != seen_request_ids[1]


def test_the_users_own_namespace_prefix_is_stripped_from_the_response(proxied):
    client, token = proxied
    body = submit(client, token).json()
    assert not body["request_id"].startswith(namespace(1))


def test_resubmitting_the_same_name_still_gets_a_fresh_request_id(proxied):
    client, token = proxied
    submit(client, token, name="again")
    submit(client, token, name="again")
    assert seen_request_ids[0] != seen_request_ids[1]


# --- exception 1: ownership ----------------------------------------------------------------------


@pytest.mark.parametrize("suffix", ["status", "results"])
def test_a_foreign_job_is_404(proxied, suffix):
    client, token = proxied
    submit(client, token)  # user 1 owns JOB
    intruder = register(client, email="mallory@example.org")["token"]
    response = client.get(f"/api/peptide/api/v1/jobs/{JOB}/{suffix}", headers=bearer(intruder))
    assert response.status_code == 404


def test_a_foreign_job_and_an_unknown_job_are_indistinguishable(proxied):
    """The proxy must not be an existence oracle: same status AND same body either way."""
    client, token = proxied
    submit(client, token)
    intruder = register(client, email="mallory@example.org")["token"]
    foreign = client.get(f"/api/peptide/api/v1/jobs/{JOB}/status", headers=bearer(intruder))
    unknown = client.get(f"/api/peptide/api/v1/jobs/{OTHER_JOB}/status", headers=bearer(intruder))
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json()["detail"].replace(JOB, "X") == unknown.json()["detail"].replace(
        OTHER_JOB, "X"
    )


def test_cancelling_someone_elses_run_is_404(proxied):
    client, token = proxied
    submit(client, token)
    intruder = register(client, email="mallory@example.org")["token"]
    response = client.post(f"/api/peptide/api/v1/jobs/{JOB}/cancel", headers=bearer(intruder))
    assert response.status_code == 404


def test_the_owner_can_reach_their_own_run(proxied):
    client, token = proxied
    submit(client, token)
    assert client.get(
        f"/api/peptide/api/v1/jobs/{JOB}/status", headers=bearer(token)
    ).status_code == 200


def test_a_job_id_in_a_query_value_is_checked_too(proxied):
    """Isolated deliberately: the path carries a run the caller DOES own, so the only thing that
    can produce a 404 is the foreign id in the query. With both ids foreign, the path check
    answers first and this guard is never exercised at all."""
    client, token = proxied
    submit(client, token)  # user 1 owns JOB
    intruder = register(client, email="mallory@example.org")["token"]
    mine = submit(client, intruder).json()["job_id"]  # the intruder owns this one
    assert mine != JOB
    assert client.get(
        f"/api/peptide/api/v1/jobs/{mine}/status", headers=bearer(intruder)
    ).status_code == 200, "precondition: the caller can reach their own run"
    response = client.get(
        f"/api/peptide/api/v1/jobs/{mine}/status?copy_of={JOB}", headers=bearer(intruder)
    )
    assert response.status_code == 404


# --- exception 2: history by observation ---------------------------------------------------------


def test_no_history_row_exists_until_a_run_is_actually_submitted(proxied):
    client, token = proxied
    assert client.get("/auth/history", headers=bearer(token)).json()["total"] == 0
    submit(client, token)
    assert client.get("/auth/history", headers=bearer(token)).json()["total"] == 1


def test_the_row_keeps_the_typed_name_and_not_the_generated_id(proxied):
    client, token = proxied
    submit(client, token, name="diabetic foot ulcer, attempt 3")
    entry = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]
    assert entry["run_name"] == "diabetic foot ulcer, attempt 3"
    assert entry["resource_id"] == JOB
    # The namespaced request_id is internal and must not be echoed anywhere.
    assert "request_id" not in entry


def test_reading_a_run_again_does_not_add_a_second_row(proxied):
    client, token = proxied
    submit(client, token)
    client.get(f"/api/peptide/api/v1/jobs/{JOB}/status", headers=bearer(token))
    client.get(f"/api/peptide/api/v1/jobs/{JOB}/results", headers=bearer(token))
    assert client.get("/auth/history", headers=bearer(token)).json()["total"] == 1


def test_a_status_poll_updates_the_row_rather_than_leaving_it_at_submitted(proxied):
    """The structural difference from the synchronous original: rows are not terminal."""
    client, token = proxied
    submit(client, token)
    before = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]
    assert before["vertex_state"] is None
    client.get(f"/api/peptide/api/v1/jobs/{JOB}/status", headers=bearer(token))
    after = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]
    assert after["vertex_state"] == "JOB_STATE_FAILED"
    assert after["id"] == before["id"]


def test_a_run_vertex_has_failed_is_not_summarised_as_merely_pending(proxied):
    """The API reports status "pending" forever for a worker that died without writing
    results.json (docs/BASELINE.md). A history list showing only summaries must still not
    imply such a run is progressing."""
    client, token = proxied
    submit(client, token)
    client.get(f"/api/peptide/api/v1/jobs/{JOB}/status", headers=bearer(token))
    entry = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]
    assert entry["status"] == "pending"
    assert entry["vertex_state"] == "JOB_STATE_FAILED"
    assert "Vertex has finished" in entry["summary"]


def test_a_results_read_stores_the_body_for_reopening(proxied):
    client, token = proxied
    submit(client, token)
    client.get(f"/api/peptide/api/v1/jobs/{JOB}/results", headers=bearer(token))
    listed = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]
    assert listed["has_envelope"] is True
    entry = client.get(f"/auth/history/{listed['id']}", headers=bearer(token)).json()
    assert entry["envelope"]["n_final"] == 2
    assert len(entry["envelope"]["ranked_candidates"]) == 2


def test_a_later_status_poll_does_not_wipe_a_stored_results_body(proxied):
    client, token = proxied
    submit(client, token)
    client.get(f"/api/peptide/api/v1/jobs/{JOB}/results", headers=bearer(token))
    client.get(f"/api/peptide/api/v1/jobs/{JOB}/status", headers=bearer(token))
    listed = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]
    assert listed["has_envelope"] is True


def test_the_history_list_states_its_staleness_contract(proxied):
    client, token = proxied
    body = client.get("/auth/history", headers=bearer(token)).json()
    assert "observed_at" in body["staleness"]


def test_one_users_history_is_invisible_to_another(proxied):
    client, token = proxied
    submit(client, token)
    other = register(client, email="grace@example.org")["token"]
    assert client.get("/auth/history", headers=bearer(other)).json()["total"] == 0


def test_a_foreign_history_entry_is_404(proxied):
    client, token = proxied
    submit(client, token)
    mine = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]["id"]
    other = register(client, email="grace@example.org")["token"]
    assert client.get(f"/auth/history/{mine}", headers=bearer(other)).status_code == 404


def test_a_recorder_failure_cannot_change_the_response(proxied, monkeypatch):
    """Storage trouble must never turn an accepted, already-submitted run into an error."""
    from accounts import history

    monkeypatch.setattr(
        history, "is_created_job", lambda body: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    client, token = proxied
    response = submit(client, token)
    assert response.status_code == 202
    assert response.json()["job_id"] == JOB


# --- hiding a run ---------------------------------------------------------------------------------


def test_hiding_a_run_removes_the_row_and_says_artifacts_survive(proxied):
    client, token = proxied
    submit(client, token)
    entry_id = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]["id"]
    response = client.delete(f"/auth/history/{entry_id}", headers=bearer(token))
    assert response.status_code == 200
    assert response.json()["artifacts_deleted"] is False
    assert client.get("/auth/history", headers=bearer(token)).json()["total"] == 0


def test_hiding_someone_elses_run_is_404(proxied):
    client, token = proxied
    submit(client, token)
    entry_id = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]["id"]
    other = register(client, email="grace@example.org")["token"]
    assert client.delete(f"/auth/history/{entry_id}", headers=bearer(other)).status_code == 404
    assert client.get("/auth/history", headers=bearer(token)).json()["total"] == 1


def test_hiding_a_run_does_not_release_its_ownership(proxied):
    """Otherwise the next account to observe that job id would claim it.

    Asserted from the OWNER's side: an intruder is refused whether or not ownership was
    released, so checking the intruder proves nothing about this guard.
    """
    client, token = proxied
    submit(client, token)
    entry_id = client.get("/auth/history", headers=bearer(token)).json()["entries"][0]["id"]
    client.delete(f"/auth/history/{entry_id}", headers=bearer(token))
    assert client.get(
        f"/api/peptide/api/v1/jobs/{JOB}/status", headers=bearer(token)
    ).status_code == 200
    intruder = register(client, email="mallory@example.org")["token"]
    assert client.get(
        f"/api/peptide/api/v1/jobs/{JOB}/status", headers=bearer(intruder)
    ).status_code == 404


# --- the gate is on the router -------------------------------------------------------------------

PUBLIC = {
    ("GET", "/healthz"),
    ("GET", "/auth/captcha"),
    ("POST", "/auth/register"),
    ("POST", "/auth/login"),
    ("POST", "/auth/recover"),
    ("POST", "/auth/logout"),
}


def test_every_route_is_either_public_or_gated(settings):
    """Walks what FastAPI actually registered. Asserts a minimum count FIRST, because
    `app.routes` holds only opaque _IncludedRouter nodes on FastAPI 0.142 -- a walker that
    missed them would find nothing and this test would pass while checking nothing."""
    app = create_app(settings)
    routes = routes_of(app)
    assert len(routes) >= 12, f"the route walker found only {len(routes)}; it is not walking"
    for route in routes:
        for method in route.methods or []:
            if (method, route.path) in PUBLIC:
                assert not is_gated(route), f"{method} {route.path} is public but gated"
            else:
                assert is_gated(route), f"{method} {route.path} is neither public nor gated"


def test_the_public_list_matches_reality(settings):
    """Guards against PUBLIC above drifting into a stale exemption list."""
    app = create_app(settings)
    found = {
        (method, route.path)
        for route in routes_of(app)
        for method in route.methods or []
        if not is_gated(route)
    }
    assert found == PUBLIC


def test_every_gated_route_actually_refuses_without_a_session(settings):
    """Behavioural, not by introspection.

    test_every_route_is_either_public_or_gated inspects resolved dependencies, and every handler
    here ALSO takes `user: User = Depends(current_user)` in its signature to get the user object.
    That means removing the router-level gate does not change what introspection sees. Calling
    each route without a token does change it, so this is the test that would notice.
    """
    app = create_app(settings)
    with TestClient(app) as client:
        checked = 0
        for route in routes_of(app):
            for method in route.methods or []:
                if (method, route.path) in PUBLIC:
                    continue
                # Substitute something concrete for each path parameter.
                path = route.path.replace("{entry_id}", "1").replace(
                    "{service}", "peptide"
                ).replace("{path:path}", "api/v1/jobs/create")
                assert "{" not in path, f"unsubstituted parameter in {path}"
                response = client.request(method, path)
                assert response.status_code == 401, (
                    f"{method} {path} answered {response.status_code} with no session"
                )
                checked += 1
        assert checked >= 6, f"only {checked} gated routes were called; the walker is not walking"


@pytest.mark.parametrize("router_name", ["gated", "proxy"])
def test_a_new_route_on_a_gated_router_is_refused_without_asking_for_it(settings, router_name):
    """The point of gating at the ROUTER rather than per route.

    Every handler in this service also takes `user: User = Depends(current_user)` for access to
    the user object, and that parameter alone makes FastAPI refuse an anonymous call. So neither
    the introspection test nor the behavioural one above notices if the router-level dependency
    is removed -- both keep passing on the signatures alone.

    What the router-level gate actually protects is the route somebody adds LATER and forgets to
    gate. This registers exactly that: a handler taking no user, on the real routers, and
    requires it to be refused anyway. Remove `dependencies=[Depends(current_user)]` from either
    router and this is the test that fails.
    """
    from accounts import auth as auth_module
    from accounts import proxy as proxy_module

    router = {"gated": auth_module.gated, "proxy": proxy_module.router}[router_name]
    path = f"/probe-{router_name}-forgot-to-gate"

    @router.get(path)
    def _probe() -> dict[str, bool]:
        return {"reached": True}

    try:
        app = create_app(settings)
        with TestClient(app) as client:
            response = client.get(path)
        assert response.status_code == 401, (
            f"a route added to the {router_name} router with no user parameter answered"
            f" {response.status_code} anonymously: the router-level gate is not doing its job"
        )
    finally:
        router.routes[:] = [r for r in router.routes if getattr(r, "path", None) != path]
