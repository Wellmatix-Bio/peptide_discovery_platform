"""The passthrough against an ECHO upstream: what is forwarded, what is dropped, what comes back.

An echo rather than a real API, so every header and byte the upstream received is visible. The
real API's own shapes are exercised in test_proxy_peptide.py.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from accounts.app import create_app

from conftest import bearer, register

echo = FastAPI()


@echo.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def _echo(path: str, request: Request) -> Response:
    body = await request.body()
    return Response(
        content=(
            '{"method": "%s", "path": "/%s", "query": "%s", "headers": %s, "body": %s}'
            % (
                request.method,
                path,
                request.url.query,
                __import__("json").dumps(dict(request.headers)),
                __import__("json").dumps(body.decode("utf-8")),
            )
        ),
        status_code=299 if path == "odd-status" else 200,
        headers={
            "content-type": "application/json",
            "set-cookie": "upstream=should-not-reach-the-browser",
            "x-upstream-internal": "should-not-reach-the-browser",
        },
    )


@pytest.fixture
def proxied(settings):
    app = create_app(settings, transports={"peptide": httpx.ASGITransport(app=echo)})
    with TestClient(app) as client:
        token = register(client)["token"]
        yield client, token


def test_method_path_query_and_body_are_forwarded(proxied):
    client, token = proxied
    response = client.post(
        "/api/peptide/some/path?a=1&b=two", headers=bearer(token), content=b'{"x": 1}',
    )
    seen = response.json()
    assert seen["method"] == "POST"
    assert seen["path"] == "/some/path"
    assert seen["query"] == "a=1&b=two"
    assert seen["body"] == '{"x": 1}'


def test_authorization_cookie_and_host_are_never_forwarded(proxied):
    client, token = proxied
    headers = {
        **bearer(token),
        "Cookie": "session=secret",
        "X-Something-Else": "dropped too",
        "Content-Type": "application/json",
    }
    seen = client.post("/api/peptide/x", headers=headers, content=b"{}").json()["headers"]
    lowered = {key.lower(): value for key, value in seen.items()}
    assert "authorization" not in lowered
    assert "cookie" not in lowered
    assert "x-something-else" not in lowered
    # Host is the UPSTREAM's, set by the client library, never the caller's.
    assert lowered.get("host") != "testserver"
    assert lowered["content-type"] == "application/json"


def test_the_response_status_and_body_come_back_unchanged(proxied):
    client, token = proxied
    response = client.get("/api/peptide/odd-status", headers=bearer(token))
    assert response.status_code == 299
    assert response.json()["path"] == "/odd-status"


def test_upstream_headers_outside_the_allow_list_are_dropped(proxied):
    client, token = proxied
    response = client.get("/api/peptide/x", headers=bearer(token))
    assert "set-cookie" not in response.headers
    assert "x-upstream-internal" not in response.headers
    assert response.headers["content-type"] == "application/json"


def test_the_passthrough_refuses_without_a_session(proxied):
    client, _ = proxied
    assert client.get("/api/peptide/x").status_code == 401


@pytest.mark.parametrize("path", ["", "docs", "redoc", "openapi.json"])
def test_upstream_docs_and_root_are_not_forwarded(proxied, path):
    client, token = proxied
    assert client.get(f"/api/peptide/{path}", headers=bearer(token)).status_code == 404


def test_an_unknown_upstream_is_404(proxied):
    client, token = proxied
    assert client.get("/api/elsewhere/x", headers=bearer(token)).status_code == 404


def test_traversal_segments_are_not_forwarded(proxied):
    client, token = proxied
    # Sent raw so the test client does not normalise it away first.
    response = client.get("/api/peptide/a/%2E%2E/docs", headers=bearer(token))
    assert response.status_code in (400, 404)
    assert "path" not in response.json() or response.json().get("path") != "/docs"


def test_an_unreachable_upstream_is_502_not_a_crash(settings):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    app = create_app(settings, transports={"peptide": httpx.MockTransport(refuse)})
    with TestClient(app) as client:
        token = register(client)["token"]
        response = client.get("/api/peptide/health", headers=bearer(token))
    assert response.status_code == 502
    assert "peptide" in response.json()["detail"]


@pytest.mark.parametrize("method,path", [("POST", "/history"), ("POST", "/auth/history"), ("PUT", "/auth/history/1")])
def test_there_is_no_endpoint_that_creates_a_history_row(proxied, method, path):
    client, token = proxied
    response = client.request(method, path, headers=bearer(token), json={"kind": "invented"})
    assert response.status_code in (404, 405)
