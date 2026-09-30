"""The captcha: what it is worth, single-use nonces, and shared storage.

The shared-storage test asserts the PROPERTY (state is in SQLite; no module-level structure holds
it), not the behaviour. A single-process test client cannot reproduce the multi-worker failure,
so "it works" would pass throughout the bug. The multi-worker test at the bottom reproduces it
for real, against two uvicorn workers.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from accounts import captcha

from conftest import PASSWORD, SERVICE, solve


def test_anyone_who_reads_the_text_nodes_solves_it(client):
    """Keeps the documentation honest: this is all the captcha is worth."""
    challenge = client.get("/auth/captcha").json()
    answer = solve(challenge["svg"])
    assert len(answer) == captcha.LENGTH
    assert captcha.check(client.app.state.accounts.db, challenge["nonce"], answer)


def test_a_nonce_is_spent_by_a_wrong_answer(client):
    db = client.app.state.accounts.db
    challenge = client.get("/auth/captcha").json()
    assert not captcha.check(db, challenge["nonce"], "WRONG")
    # The right answer, second, is too late: otherwise a 1-in-N guess becomes a certainty.
    assert not captcha.check(db, challenge["nonce"], solve(challenge["svg"]))


def test_a_nonce_is_spent_by_a_right_answer(client):
    db = client.app.state.accounts.db
    challenge = client.get("/auth/captcha").json()
    answer = solve(challenge["svg"])
    assert captcha.check(db, challenge["nonce"], answer)
    assert not captcha.check(db, challenge["nonce"], answer)


def test_an_unknown_nonce_is_refused(client):
    assert not captcha.check(client.app.state.accounts.db, "never-issued", "ABCDE")


def test_captcha_state_lives_in_sqlite(client, settings):
    challenge = client.get("/auth/captcha").json()
    with sqlite3.connect(settings.database) as db:
        rows = db.execute("SELECT nonce, spent FROM captchas").fetchall()
    assert (challenge["nonce"], 0) in rows


def test_no_module_level_structure_can_hold_captcha_state():
    """The property. Put the store back in a dict and this fails, whether or not anything breaks."""
    mutable = {
        name: type(value).__name__
        for name, value in vars(captcha).items()
        if isinstance(value, (dict, list, set, bytearray)) and not name.startswith("__")
    }
    assert mutable == {}, f"captcha.py holds module-level mutable state: {mutable}"


# --- the real multi-worker run ------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def two_workers(tmp_path: Path):
    port = _free_port()
    env = dict(
        os.environ,
        ACCOUNTS_SECRET_KEY="t" * 48,
        ACCOUNTS_DATA_DIR=str(tmp_path),
        WEB_CONCURRENCY="2",
        # As in deploy/compose.yaml, so each registration below can come from its own address.
        ACCOUNTS_CLIENT_IP_HEADER="x-forwarded-for",
        PYTHONPATH=str(SERVICE),
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "accounts.app:build", "--factory",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=SERVICE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(base + "/healthz", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        process.kill()
        pytest.fail("the two-worker server did not start:\n" + process.stdout.read().decode())
    yield base
    process.terminate()
    process.wait(timeout=10)


def test_captcha_and_tokens_work_across_two_workers(two_workers):
    """A challenge issued by one worker must validate on the other, and a token likewise.

    Fresh connections per request so the kernel spreads them across both workers. With the store
    in process memory, roughly half of these registrations fail.
    """
    for index in range(12):
        with httpx.Client(base_url=two_workers, timeout=10) as issue:
            challenge = issue.get("/auth/captcha").json()
        with httpx.Client(base_url=two_workers, timeout=10) as answer:
            response = answer.post(
                "/auth/register",
                headers={"X-Forwarded-For": f"203.0.113.{index}"},
                json={
                    "email": f"worker{index}@example.org",
                    "password": PASSWORD,
                    "captcha_nonce": challenge["nonce"],
                    "captcha_answer": solve(challenge["svg"]),
                },
            )
        assert response.status_code == 201, response.text
        token = response.json()["token"]
        with httpx.Client(base_url=two_workers, timeout=10) as check:
            assert check.get(
                "/auth/me", headers={"Authorization": f"Bearer {token}"}
            ).status_code == 200


def test_the_registration_limit_is_shared_by_both_workers(two_workers):
    """Five per address per hour, counted in SQLite. With per-process counters, two workers would
    allow ten, and fresh connections spread across both would see it."""
    statuses = []
    for _ in range(8):
        with httpx.Client(base_url=two_workers, timeout=10) as client:
            statuses.append(
                client.post(
                    "/auth/register",
                    headers={"X-Forwarded-For": "198.51.100.7"},
                    json={"email": "x@example.org", "password": PASSWORD, "captcha_nonce": "n", "captcha_answer": "a"},
                ).status_code
            )
    assert statuses[:5] == [400] * 5, statuses  # counted, then refused for the bad captcha
    assert statuses[5:] == [429] * 3, statuses
