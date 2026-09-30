"""Rate limits on login, registration and recovery (accounts/ratelimit.py).

Small limits here so each property is reachable in a handful of requests; the production numbers
are the Policy defaults and are asserted separately, because they are a decision someone made.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from accounts import ratelimit
from accounts.app import create_app
from accounts.config import Settings
from accounts.ratelimit import Limit, Policy

from conftest import KEY, PASSWORD, captcha_fields, register

SMALL = Policy(
    login_per_ip=Limit("login-ip", 6, 60),
    login_failures_per_email=Limit("login-email", 3, 60),
    register_per_ip=Limit("register-ip", 3, 60),
    recover_per_ip=Limit("recover-ip", 2, 60),
)


def make(tmp_path, header: str | None = None) -> TestClient:
    settings = Settings(
        data_dir=tmp_path, secret_key=KEY, key_from_environment=True,
        rate_policy=SMALL, client_ip_header=header,
    )
    return TestClient(create_app(settings))


@pytest.fixture
def client(tmp_path):
    with make(tmp_path) as test_client:
        yield test_client


def login(client, email, password="not the password", ip=None):
    headers = {"X-Forwarded-For": ip} if ip else {}
    return client.post("/auth/login", json={"email": email, "password": password}, headers=headers)


# --- the decided numbers ----------------------------------------------------------------------------


def test_the_production_policy_is_the_one_that_was_decided():
    policy = Policy()
    assert (policy.login_per_ip.max_events, policy.login_per_ip.window_seconds) == (30, 900)
    assert (policy.login_failures_per_email.max_events, policy.login_failures_per_email.window_seconds) == (10, 900)
    assert (policy.register_per_ip.max_events, policy.register_per_ip.window_seconds) == (5, 3600)
    assert (policy.recover_per_ip.max_events, policy.recover_per_ip.window_seconds) == (5, 3600)


# --- login ------------------------------------------------------------------------------------------


def test_failed_logins_for_one_email_are_limited_with_retry_after(client):
    register(client)
    for _ in range(3):
        assert login(client, "ada@example.org").status_code == 401
    limited = login(client, "ada@example.org")
    assert limited.status_code == 429
    assert 1 <= int(limited.headers["Retry-After"]) <= 60
    # Even the right password waits: that is what slowing a guesser means.
    assert login(client, "ada@example.org", PASSWORD).status_code == 429


def test_a_limited_unknown_email_looks_exactly_like_a_limited_account(tmp_path):
    """Otherwise the 429 is the enumeration oracle the login message was written to avoid.

    Every attempt from its own address, so the per-address limit cannot produce the 429s and hide
    whether the per-email counter ran for the unknown email. It did hide it in the first version.
    """
    with make(tmp_path, header="x-forwarded-for") as client:
        register(client)
        addresses = (f"203.0.113.{i}" for i in range(100))
        for email in ("ada@example.org", "nobody@example.org"):
            for _ in range(3):
                assert login(client, email, ip=next(addresses)).status_code == 401
        known = login(client, "ada@example.org", ip=next(addresses))
        unknown = login(client, "nobody@example.org", ip=next(addresses))
        assert known.status_code == unknown.status_code == 429
        assert known.json() == unknown.json()


def test_the_email_limit_ignores_case_and_spaces(client):
    for email in ("Ada@Example.org", " ada@example.org", "ADA@EXAMPLE.ORG"):
        login(client, email)
    assert login(client, "ada@example.org").status_code == 429


def test_successful_logins_do_not_use_up_the_failure_allowance(client):
    register(client)
    for _ in range(5):
        assert login(client, "ada@example.org", PASSWORD).status_code == 200


def test_login_attempts_per_address_are_limited_across_emails(client):
    statuses = [login(client, f"user{i}@example.org").status_code for i in range(8)]
    assert statuses[:6] == [401] * 6 and statuses[6:] == [429, 429], statuses


def test_the_email_limit_holds_across_addresses(tmp_path):
    """A botnet spreading guesses for one account over many addresses is still slowed."""
    with make(tmp_path, header="x-forwarded-for") as client:
        register(client)
        for i in range(3):
            assert login(client, "ada@example.org", ip=f"203.0.113.{i}").status_code == 401
        assert login(client, "ada@example.org", ip="203.0.113.99").status_code == 429


def test_nothing_is_locked_once_the_window_passes(client, monkeypatch):
    register(client)
    for _ in range(4):
        login(client, "ada@example.org")
    assert login(client, "ada@example.org", PASSWORD).status_code == 429
    later = ratelimit.time.time() + 61
    monkeypatch.setattr(ratelimit.time, "time", lambda: later)
    assert login(client, "ada@example.org", PASSWORD).status_code == 200


# --- register and recover ---------------------------------------------------------------------------


def test_registration_is_limited_per_address_and_counts_captcha_failures(client):
    statuses = [
        client.post(
            "/auth/register",
            json={"email": f"b{i}@example.org", "password": PASSWORD, "captcha_nonce": "x", "captcha_answer": "y"},
        ).status_code
        for i in range(5)
    ]
    assert statuses == [400, 400, 400, 429, 429], statuses


def test_recovery_is_limited_per_address(client):
    body = {"email": "ada@example.org", "recovery_code": "AAAA-AAAA-AAAA-AAAA", "new_password": "a new password"}
    statuses = [client.post("/auth/recover", json=body).status_code for _ in range(4)]
    assert statuses == [401, 401, 429, 429], statuses


# --- the client address -----------------------------------------------------------------------------


def test_a_forwarded_header_is_ignored_unless_configured(client):
    """Unconfigured, a client choosing its own X-Forwarded-For would get a fresh allowance each time."""
    statuses = [
        client.post(
            "/auth/register",
            headers={"X-Forwarded-For": f"198.51.100.{i}"},
            json={"email": f"s{i}@example.org", "password": PASSWORD, "captcha_nonce": "x", "captcha_answer": "y"},
        ).status_code
        for i in range(4)
    ]
    assert statuses[-1] == 429, statuses


def test_a_configured_header_separates_clients(tmp_path):
    with make(tmp_path, header="x-forwarded-for") as client:
        for i in range(4):
            response = client.post(
                "/auth/register",
                headers={"X-Forwarded-For": f"198.51.100.{i}"},
                json={"email": f"s{i}@example.org", "password": PASSWORD, **captcha_fields(client)},
            )
            assert response.status_code == 201, response.text


# --- storage ---------------------------------------------------------------------------------------


def test_no_email_is_stored_in_clear(client, tmp_path):
    login(client, "secret.person@example.org")
    with sqlite3.connect(tmp_path / "accounts.sqlite3") as db:
        buckets = [row[0] for row in db.execute("SELECT bucket FROM rate_counts")]
    assert buckets and not any("example.org" in bucket for bucket in buckets), buckets


def test_counters_live_in_sqlite_and_no_module_level_structure_holds_them():
    mutable = {
        name: type(value).__name__
        for name, value in vars(ratelimit).items()
        if isinstance(value, (dict, list, set, bytearray)) and not name.startswith("__")
    }
    assert mutable == {}, f"ratelimit.py holds module-level mutable state: {mutable}"


def test_expiring_one_limit_does_not_delete_another_limits_running_window(client, monkeypatch):
    """Cleanup is by each row's own expiry. By window start alone, the 60 s login window's cleanup
    would have deleted a longer window that was still running."""
    long_limit = Limit("long", 1, 3600)
    db = client.app.state.accounts.db
    ratelimit.hit(db, long_limit, "subject")
    later = ratelimit.time.time() + 120
    monkeypatch.setattr(ratelimit.time, "time", lambda: later)
    login(client, "anyone@example.org")  # counts in a fresh short window, cleaning up on the way
    with pytest.raises(Exception) as limited:
        ratelimit.hit(db, long_limit, "subject")
    assert getattr(limited.value, "status_code", None) == 429
