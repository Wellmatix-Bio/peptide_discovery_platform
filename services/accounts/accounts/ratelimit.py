"""Rate limits on the credential routes. Decided 29 September 2026 by the owner.

WHAT IS COUNTED, and the numbers are POLICY, chosen rather than measured - nobody knows a
legitimate caller's peak yet (LIMITATIONS 7G.4):

- login:    every attempt per client IP, 30 per 15 minutes; and FAILED attempts per email,
            10 per 15 minutes. The per-email limit is what slows a botnet spreading guesses for
            one account across many addresses.
- register: every attempt per client IP, 5 per hour.
- recover:  every attempt per client IP, 5 per hour.

Over a limit the answer is 429 with Retry-After. NOTHING IS EVER LOCKED: a lock would let anyone
who knows an email address lock its owner out. An attacker is slowed, not the user stopped.

THE PER-EMAIL COUNTER RUNS FOR EMAILS THAT DO NOT EXIST TOO, and a limited request is refused
before any account is looked up. So a 429 says nothing about whether an account exists; otherwise
the limit itself would be the enumeration oracle the login message was written to avoid.

Counters live in SQLite (db.rate_counts), never in process memory, for the same reason the
captcha store does: a module-level dict passes every single-process test and gives each worker
its own separate allowance. Emails are stored as an HMAC under the signing key, never in clear -
the table would otherwise collect every address anybody typed, including ones that are not
accounts.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from .db import Database


@dataclass(frozen=True)
class Limit:
    name: str
    max_events: int
    window_seconds: int


@dataclass(frozen=True)
class Policy:
    login_per_ip: Limit = Limit("login-ip", 30, 15 * 60)
    login_failures_per_email: Limit = Limit("login-email", 10, 15 * 60)
    register_per_ip: Limit = Limit("register-ip", 5, 60 * 60)
    recover_per_ip: Limit = Limit("recover-ip", 5, 60 * 60)


LIMITED = "too many attempts; try again in {minutes} minute(s)"


def client_ip(request: Request, header: str | None) -> str:
    """The caller's address.

    `header` is trusted ONLY when configured (ACCOUNTS_CLIENT_IP_HEADER), and must only be
    configured when every request reaches this service through a proxy that OVERWRITES that header
    - deploy/nginx.conf sets X-Forwarded-For to $remote_addr. Unconfigured, a client could send
    any X-Forwarded-For and pick a fresh allowance per request, so the header is ignored and the
    socket peer is used.
    """
    if header:
        value = request.headers.get(header, "")
        # If a chain arrives anyway, the rightmost entry is the one the nearest proxy wrote.
        candidate = value.split(",")[-1].strip()
        if candidate:
            return candidate
    return request.client.host if request.client else "unknown"


def email_key(key: bytes, email: str) -> str:
    normalised = email.strip().lower()
    return hmac.new(key, normalised.encode("utf-8"), hashlib.sha256).hexdigest()


def _window(limit: Limit, now: float) -> tuple[int, int]:
    start = int(now // limit.window_seconds) * limit.window_seconds
    return start, start + limit.window_seconds


def _refuse(limit: Limit, now: float) -> HTTPException:
    _, end = _window(limit, now)
    retry = max(1, math.ceil(end - now))
    return HTTPException(
        429,
        LIMITED.format(minutes=max(1, math.ceil(retry / 60))),
        headers={"Retry-After": str(retry)},
    )


def hit(db: Database, limit: Limit, subject: str) -> None:
    """Count this attempt, and refuse it if it is over the limit."""
    now = time.time()
    start, end = _window(limit, now)
    count = db.rate_hit(f"{limit.name}:{subject}", start, end, now)
    if count > limit.max_events:
        raise _refuse(limit, now)


def check(db: Database, limit: Limit, subject: str) -> None:
    """Refuse if the limit is already reached, WITHOUT counting this attempt."""
    now = time.time()
    start, _ = _window(limit, now)
    if db.rate_count(f"{limit.name}:{subject}", start) >= limit.max_events:
        raise _refuse(limit, now)


def record(db: Database, limit: Limit, subject: str) -> None:
    """Count an event after the fact - a failed login - without refusing anything."""
    now = time.time()
    start, end = _window(limit, now)
    db.rate_hit(f"{limit.name}:{subject}", start, end, now)
