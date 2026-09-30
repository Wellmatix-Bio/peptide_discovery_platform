"""Fixtures for the accounts service.

These tests live outside the repository's top-level `tests/` directory on purpose: that tree
tests the pipeline, needs its (currently broken) `pipeline` package to import, and is the
baseline the frontend work is measured against. Nothing here touches it. Run them with:

    .venv/bin/python -m pytest services/accounts/tests

`routes_of` below is not a convenience: see its docstring.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
SERVICE = HERE.parent
if str(SERVICE) not in sys.path:
    sys.path.insert(0, str(SERVICE))

from accounts.app import create_app  # noqa: E402
from accounts.config import Settings  # noqa: E402

KEY = b"k" * 48
PASSWORD = "correct horse battery"


def solve(svg: str) -> str:
    """Solve the captcha the way any bot can: read the SVG's text nodes.

    This is a TEST OF THE DOCUMENTATION as much as a helper. accounts/captcha.py says the captcha
    stops drive-by bots and nothing more; if this ever stops working, that sentence needs
    re-reading rather than this function needing cleverness.
    """
    return "".join(re.findall(r"<text[^>]*>([^<]+)</text>", svg))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, secret_key=KEY, key_from_environment=True)


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def captcha_fields(client: TestClient) -> dict[str, str]:
    challenge = client.get("/auth/captcha").json()
    return {"captcha_nonce": challenge["nonce"], "captcha_answer": solve(challenge["svg"])}


def register(client: TestClient, email: str = "ada@example.org", password: str = PASSWORD):
    response = client.post(
        "/auth/register", json={"email": email, "password": password, **captcha_fields(client)}
    )
    assert response.status_code == 201, response.text
    return response.json()


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def routes_of(app) -> list:
    """Every registered route, recursively.

    NOT a one-liner over `app.routes`. Since FastAPI 0.141 an `include_router` call leaves an
    opaque `_IncludedRouter` node in `app.routes` -- it exposes no `path`, so the obvious walker
    finds ZERO routes and every test built on it passes while checking nothing. This service runs
    FastAPI 0.142, where `app.routes` holds exactly 3 such nodes and no real route; the real ones
    hang off each node's `original_router`. Callers assert a minimum count for this reason.
    """
    found, seen = [], set()

    def walk(container) -> None:
        for route in getattr(container, "routes", []):
            if id(route) in seen:
                continue
            seen.add(id(route))
            inner = getattr(route, "original_router", None)
            if inner is not None:
                walk(inner)
            elif hasattr(route, "path"):
                found.append(route)
            else:
                walk(route)

    walk(app)
    return found


def is_gated(route) -> bool:
    """Whether a route carries the session dependency, by inspecting what FastAPI resolved
    rather than by matching its path against a list."""
    dependant = getattr(route, "dependant", None)
    names = {
        getattr(dependency.call, "__name__", "")
        for dependency in getattr(dependant, "dependencies", []) or []
    }
    return "current_user" in names
