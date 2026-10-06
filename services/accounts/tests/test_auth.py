"""The properties that matter, each written so that breaking it deliberately makes a test fail.

| Break this                                      | Fails                                         |
|-------------------------------------------------|-----------------------------------------------|
| remove spend_equal_work on the unknown email    | test_unknown_email_costs_the_same_as_a_wrong_password |
| stop comparing password_version                 | test_changing_the_password_revokes_old_tokens |
| return 401 on a wrong current password          | test_wrong_current_password_is_403_not_401    |
| register a route on the app instead of a router | test_every_route_is_either_public_or_gated    |
"""

from __future__ import annotations

import sqlite3
import statistics
import time

import pytest
from fastapi.routing import APIRoute

from accounts import auth, crypto
from accounts.config import KEY_ENV, StartupError, check_writable, signing_key

from conftest import KEY, PASSWORD, bearer, captcha_fields, register

PUBLIC = {
    ("GET", "/healthz"),
    ("GET", "/auth/captcha"),
    ("POST", "/auth/register"),
    ("POST", "/auth/login"),
    ("POST", "/auth/recover"),
    ("POST", "/auth/logout"),
}


# --- registration -----------------------------------------------------------------------------------


def test_registration_returns_a_session_and_a_recovery_code_once(client):
    body = register(client)
    assert body["user"]["email"] == "ada@example.org"
    assert len(crypto.normalise_recovery_code(body["recovery_code"])) == 16
    assert client.get("/auth/me", headers=bearer(body["token"])).status_code == 200


def test_the_recovery_code_is_stored_only_as_a_hash(client, settings):
    body = register(client)
    with sqlite3.connect(settings.database) as db:
        stored = db.execute("SELECT recovery_hash FROM users").fetchone()[0]
    assert crypto.normalise_recovery_code(body["recovery_code"]) not in stored
    assert stored == crypto.hash_recovery_code(body["recovery_code"])


def test_email_uniqueness_is_case_insensitive(client):
    register(client, "Ada@Example.org")
    response = client.post(
        "/auth/register",
        json={"email": "ada@EXAMPLE.ORG", "password": PASSWORD, **captcha_fields(client)},
    )
    assert response.status_code == 409


def test_email_uniqueness_is_enforced_by_the_schema_not_by_code(settings, client):
    """Go around the application entirely. The constraint must still hold."""
    register(client, "ada@example.org")
    with sqlite3.connect(settings.database) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES ('ADA@example.org', 'x', 'now')"
        )


def test_registration_without_a_solved_captcha_is_refused(client):
    challenge = client.get("/auth/captcha").json()
    response = client.post(
        "/auth/register",
        json={
            "email": "bot@example.org",
            "password": PASSWORD,
            "captcha_nonce": challenge["nonce"],
            "captcha_answer": "WRONG",
        },
    )
    assert response.status_code == 400


def test_short_passwords_are_refused(client):
    response = client.post(
        "/auth/register",
        json={"email": "ada@example.org", "password": "short", **captcha_fields(client)},
    )
    assert response.status_code == 422


# --- login: one answer, one cost -------------------------------------------------------------------


def test_unknown_email_and_wrong_password_get_the_same_answer(client):
    register(client)
    unknown = client.post("/auth/login", json={"email": "nobody@example.org", "password": PASSWORD})
    wrong = client.post("/auth/login", json={"email": "ada@example.org", "password": "not it at all"})
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json() == {"detail": auth.LOGIN_REFUSED}


def test_unknown_email_costs_the_same_as_a_wrong_password(client):
    """The message is half the control; the time is the other half.

    Without the equal-work call the unknown-email path returns in well under a millisecond and a
    real scrypt takes tens, so the ratio collapses to about 0.01. The threshold is loose on
    purpose: it is testing for the absence of a whole scrypt, not for a microsecond.

    MEASURED BY INTERLEAVING, and that matters. An earlier version timed all the known-email
    requests, then all the unknown-email ones, and compared medians. On a shared CI runner the
    machine's load changes between those two blocks, so the ratio measured the runner's mood as
    much as the code: one failure reported 40.1 ms against 109.0 ms, a 0.37 ratio, on a build
    where both paths provably run the same scrypt. Alternating the two within one loop means any
    drift hits both.

    MINIMUM, NOT MEDIAN. Noise can only make a timing longer, never shorter, so the smallest
    sample is the closest estimate of the true cost and the one least disturbed by a neighbour
    process.

    The threshold stays at half, because it has room: with the equal-work call removed the ratio
    is about 0.01, which is fifty times below the bar. It is testing for the absence of a whole
    scrypt, not for a microsecond.
    """
    register(client)

    def once(email: str) -> float:
        start = time.perf_counter()
        client.post("/auth/login", json={"email": email, "password": "not it at all"})
        return time.perf_counter() - start

    known_samples, unknown_samples = [], []
    for _ in range(9):
        known_samples.append(once("ada@example.org"))
        unknown_samples.append(once("nobody@example.org"))

    known, unknown = min(known_samples), min(unknown_samples)
    assert unknown > 0.5 * known, (
        f"unknown email {unknown*1000:.1f} ms vs known {known*1000:.1f} ms "
        f"(medians {statistics.median(unknown_samples)*1000:.1f} / "
        f"{statistics.median(known_samples)*1000:.1f} ms)"
    )


def test_login_is_case_insensitive_on_email(client):
    register(client, "ada@example.org")
    response = client.post("/auth/login", json={"email": "ADA@example.org", "password": PASSWORD})
    assert response.status_code == 200


# --- revocation -------------------------------------------------------------------------------------


def test_changing_the_password_revokes_old_tokens_and_issues_a_replacement(client):
    old = register(client)["token"]
    response = client.post(
        "/auth/change-password",
        headers=bearer(old),
        json={"current_password": PASSWORD, "new_password": "a different password"},
    )
    assert response.status_code == 200
    new = response.json()["token"]
    assert client.get("/auth/me", headers=bearer(old)).status_code == 401
    assert client.get("/auth/me", headers=bearer(new)).status_code == 200


def test_a_token_from_a_stale_password_version_is_refused(client, settings):
    """The same property as above without going through the route: the gate compares versions."""
    user = register(client)["user"]
    stale = crypto.issue_token(KEY, user["id"], password_version=0, ttl_seconds=60)
    assert client.get("/auth/me", headers=bearer(stale)).status_code == 401


def test_wrong_current_password_is_403_not_401(client):
    token = register(client)["token"]
    response = client.post(
        "/auth/change-password",
        headers=bearer(token),
        json={"current_password": "not the password", "new_password": "a different password"},
    )
    assert response.status_code == 403
    # ...and the session was not harmed by it.
    assert client.get("/auth/me", headers=bearer(token)).status_code == 200


def test_logout_admits_it_revokes_nothing(client):
    token = register(client)["token"]
    body = client.post("/auth/logout", headers=bearer(token)).json()
    assert body["logged_out"] is False
    assert "cannot revoke" in body["why"]
    assert client.get("/auth/me", headers=bearer(token)).status_code == 200


# --- recovery ---------------------------------------------------------------------------------------


def test_the_recovery_code_is_single_use_and_rotates(client):
    registered = register(client)
    code = registered["recovery_code"]
    first = client.post(
        "/auth/recover",
        json={"email": "ada@example.org", "recovery_code": code, "new_password": "recovered pass 1"},
    )
    assert first.status_code == 200
    assert first.json()["recovery_code"] != code
    again = client.post(
        "/auth/recover",
        json={"email": "ada@example.org", "recovery_code": code, "new_password": "recovered pass 2"},
    )
    assert again.status_code == 401
    # The old session is revoked; the new password works.
    assert client.get("/auth/me", headers=bearer(registered["token"])).status_code == 401
    login = client.post("/auth/login", json={"email": "ada@example.org", "password": "recovered pass 1"})
    assert login.status_code == 200


def test_recovery_refusals_are_one_message(client):
    register(client)
    unknown = client.post(
        "/auth/recover",
        json={"email": "nobody@example.org", "recovery_code": "AAAA-AAAA-AAAA-AAAA", "new_password": "whatever pass"},
    )
    wrong = client.post(
        "/auth/recover",
        json={"email": "ada@example.org", "recovery_code": "AAAA-AAAA-AAAA-AAAA", "new_password": "whatever pass"},
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()


# --- the gate ---------------------------------------------------------------------------------------


def _routes(app):
    """(method, path, gated?) for every route, whichever way this FastAPI stores inclusion.

    FastAPI 0.141 keeps an included router as one node holding `original_router`; older versions
    flatten it into APIRoutes carrying the router's dependencies. Both are read, so this does not
    pass vacuously on either - and the floor in the tests below says so if it ever does.
    """
    for route in app.routes:
        router = getattr(route, "original_router", None)
        if router is not None:
            router_gated = _gates(router.dependencies)
            for inner in router.routes:
                if isinstance(inner, APIRoute):
                    for method in inner.methods:
                        yield method, inner.path, router_gated or _gates(inner.dependencies)
        elif isinstance(route, APIRoute):
            for method in route.methods:
                yield method, route.path, _gates(route.dependencies)


def _gates(dependencies) -> bool:
    return any(getattr(dep, "dependency", None) is auth.current_user for dep in dependencies)


def test_every_route_is_either_public_or_gated(client):
    """A route that is neither was registered somewhere it should not have been."""
    seen = list(_routes(client.app))
    assert len(seen) >= len(PUBLIC) + 2, f"found only {seen}, so this checked nothing"
    for method, path, gated in seen:
        if (method, path) in PUBLIC:
            assert not gated, f"{method} {path} is meant to be public"
        else:
            assert gated, f"{method} {path} is not gated and not on the public list"


def test_every_gated_route_refuses_without_a_token(client):
    checked = 0
    for method, path, gated in _routes(client.app):
        if not gated:
            continue
        concrete = path.replace("{path:path}", "anything").replace("{path}", "anything")
        response = client.request(method, concrete)
        assert response.status_code == 401, f"{method} {path} answered {response.status_code}"
        checked += 1
    assert checked >= 2, "found nothing gated to check, so this checked nothing"


def test_public_routes_stay_public(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/auth/captcha").status_code == 200
    assert client.post("/auth/logout").status_code == 200


@pytest.mark.parametrize(
    "token",
    [
        "",
        "garbage",
        "a.b",
        "a.b.c",
        "eyJ1aWQiOjF9.AAAA",
        crypto.issue_token(b"x" * 48, 1, 1, 60),  # signed with another key
        crypto.issue_token(KEY, 1, 1, -1),  # expired
        crypto.issue_token(KEY, 999, 1, 60),  # a user who does not exist
    ],
)
def test_garbage_tokens_are_refused(client, token):
    register(client)
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


def test_a_tampered_payload_is_refused(client):
    token = register(client)["token"]
    payload, signature = token.split(".")
    forged = crypto._b64(crypto._unb64(payload).replace(b'"uid":1', b'"uid":2'))
    assert client.get("/auth/me", headers=bearer(f"{forged}.{signature}")).status_code == 401


def test_a_non_bearer_scheme_is_refused(client):
    token = register(client)["token"]
    assert client.get("/auth/me", headers={"Authorization": f"Basic {token}"}).status_code == 401


# --- startup ----------------------------------------------------------------------------------------


def test_a_missing_key_is_random_per_process_never_a_constant():
    first, from_env_1 = signing_key({})
    second, from_env_2 = signing_key({})
    assert not from_env_1 and not from_env_2
    assert first != second
    assert len(first) >= 32


def test_a_short_key_is_refused():
    with pytest.raises(StartupError, match="at least"):
        signing_key({KEY_ENV: "short"})


def test_an_unwritable_data_directory_names_the_fix(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        if (locked / "x").exists() or _can_write(locked):
            pytest.skip("running as a user that can write anywhere (root)")
        with pytest.raises(StartupError) as refused:
            check_writable(locked)
        message = str(refused.value)
        assert "chown" in message and "uid" in message and "owned by" in message
    finally:
        locked.chmod(0o700)


def _can_write(directory) -> bool:
    try:
        (directory / "probe").write_text("x")
        (directory / "probe").unlink()
        return True
    except OSError:
        return False
