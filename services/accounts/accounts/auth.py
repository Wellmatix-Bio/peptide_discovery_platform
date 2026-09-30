"""Registration, login, recovery, password change - and the gate.

GATED AT THE ROUTER, NOT PER ROUTE. `public` is a separate, ungated router. `gated` carries the
authentication dependency on the router itself, so a route is protected by virtue of where it was
registered. There is no exemption list to forget to add to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from . import captcha, crypto, ratelimit
from .db import Database, EmailTaken, User

# ONE message for every failed credential check. Two different answers turn the login form into
# an account-enumeration oracle; so do two different response TIMES, which is what
# crypto.spend_equal_work is for.
LOGIN_REFUSED = "email or password not recognised"
RECOVERY_REFUSED = "email or recovery code not recognised"
CAPTCHA_REFUSED = (
    "the captcha answer was wrong, expired or already used. Each challenge can be tried once;"
    " request a new one"
)
SIGNED_OUT = "not signed in, or the session has expired or been revoked; sign in again"
LOGOUT_EXPLAINED = (
    "Sessions are stateless signed tokens, so the server cannot revoke one individually. Your"
    " client has discarded its copy; any other copy stays valid until it expires. To revoke every"
    " session, change your password."
)

EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}\.[^@\s]{1,63}$")
PASSWORD_MIN = 10
PASSWORD_MAX = 256


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=PASSWORD_MAX)


class Registration(Credentials):
    captcha_nonce: str = Field(max_length=128)
    captcha_answer: str = Field(max_length=32)


class Recovery(BaseModel):
    email: str = Field(max_length=254)
    recovery_code: str = Field(max_length=64)
    new_password: str = Field(max_length=PASSWORD_MAX)


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=PASSWORD_MAX)
    new_password: str = Field(max_length=PASSWORD_MAX)


class UserOut(BaseModel):
    id: int
    email: str
    created_at: str


class Session(BaseModel):
    token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class SessionWithRecovery(Session):
    recovery_code: str = Field(
        description="Shown ONCE. Only its hash is stored, so it cannot be shown or sent again."
    )


@dataclass(frozen=True)
class Context:
    """What the routes need, attached to app.state by the app factory."""

    db: Database
    key: bytes
    token_ttl: int
    captcha_ttl: int
    client_ip_header: str | None = None
    rate: ratelimit.Policy = ratelimit.Policy()


def context(request: Request) -> Context:
    return request.app.state.accounts


def _user_out(user: User) -> UserOut:
    return UserOut(id=user.id, email=user.email, created_at=user.created_at)


def _session(ctx: Context, user: User, password_version: int | None = None) -> Session:
    version = user.password_version if password_version is None else password_version
    return Session(
        token=crypto.issue_token(ctx.key, user.id, version, ctx.token_ttl),
        expires_in=ctx.token_ttl,
        user=_user_out(user),
    )


def _password_problem(password: str) -> str | None:
    if len(password) < PASSWORD_MIN:
        return f"the password must be at least {PASSWORD_MIN} characters"
    if not password.strip():
        return "the password cannot be only whitespace"
    return None


# --- the gate --------------------------------------------------------------------------------------


def current_user(request: Request) -> User:
    """The ONE authentication check. Every route on `gated` runs it; nothing else does.

    Refused: no header, a header that is not Bearer, a token that fails its signature, has
    expired, names a user that does not exist, or carries a password_version that is no longer
    current. The last is revocation, and it is the only revocation there is.
    """
    ctx = context(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, SIGNED_OUT, headers={"WWW-Authenticate": "Bearer"}
        )
    try:
        claims = crypto.read_token(ctx.key, token.strip())
    except crypto.BadToken:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, SIGNED_OUT, headers={"WWW-Authenticate": "Bearer"}
        ) from None
    user = ctx.db.user_by_id(claims.user_id)
    if user is None or user.password_version != claims.password_version:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, SIGNED_OUT, headers={"WWW-Authenticate": "Bearer"}
        )
    request.state.user = user
    return user


public = APIRouter(tags=["public"])
gated = APIRouter(dependencies=[Depends(current_user)], tags=["signed in"])


# --- public ----------------------------------------------------------------------------------------


@public.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@public.get("/auth/captcha")
def new_captcha(request: Request) -> dict[str, object]:
    ctx = context(request)
    challenge = captcha.issue(ctx.db, ctx.captcha_ttl)
    return {"nonce": challenge.nonce, "svg": challenge.svg, "expires_in": challenge.expires_in}


@public.post("/auth/register", status_code=201, response_model=SessionWithRecovery)
def register(body: Registration, request: Request) -> SessionWithRecovery:
    ctx = context(request)
    # Counted before anything else, captcha included: a wrong captcha is an attempt too.
    ratelimit.hit(ctx.db, ctx.rate.register_per_ip, ratelimit.client_ip(request, ctx.client_ip_header))
    # Captcha first, so a bot learns nothing about emails or password rules without solving it.
    if not captcha.check(ctx.db, body.captcha_nonce, body.captcha_answer):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, CAPTCHA_REFUSED)
    email = body.email.strip()
    if not EMAIL.match(email):
        raise HTTPException(422, "that does not look like an email address")
    problem = _password_problem(body.password)
    if problem:
        raise HTTPException(422, problem)
    code = crypto.new_recovery_code()
    try:
        user = ctx.db.create_user(
            email, crypto.hash_password(body.password), crypto.hash_recovery_code(code)
        )
    except EmailTaken:
        # Registration DOES reveal whether an email is registered: there is no mail path to hide
        # it behind. The captcha makes that expensive to ask in bulk, and no more than that.
        raise HTTPException(409, "an account with this email already exists") from None
    session = _session(ctx, user)
    return SessionWithRecovery(**session.model_dump(), recovery_code=code)


@public.post("/auth/login", response_model=Session)
def login(body: Credentials, request: Request) -> Session:
    ctx = context(request)
    ratelimit.hit(ctx.db, ctx.rate.login_per_ip, ratelimit.client_ip(request, ctx.client_ip_header))
    # Checked BEFORE the account is looked up, and keyed on the email as typed, so a 429 is the
    # same answer for an address that is an account and one that is not.
    email_subject = ratelimit.email_key(ctx.key, body.email)
    ratelimit.check(ctx.db, ctx.rate.login_failures_per_email, email_subject)
    user = ctx.db.user_by_email(body.email.strip())
    if user is None:
        # EQUAL WORK before refusing. Remove this line and tests/test_auth.py's timing test fails.
        crypto.spend_equal_work(body.password)
        ratelimit.record(ctx.db, ctx.rate.login_failures_per_email, email_subject)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, LOGIN_REFUSED)
    if not crypto.verify_password(body.password, user.password_hash):
        ratelimit.record(ctx.db, ctx.rate.login_failures_per_email, email_subject)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, LOGIN_REFUSED)
    return _session(ctx, user)


@public.post("/auth/recover", response_model=SessionWithRecovery)
def recover(body: Recovery, request: Request) -> SessionWithRecovery:
    """Spend the recovery code, set a new password, revoke every session, issue a NEW code."""
    ctx = context(request)
    ratelimit.hit(ctx.db, ctx.rate.recover_per_ip, ratelimit.client_ip(request, ctx.client_ip_header))
    problem = _password_problem(body.new_password)
    if problem:
        raise HTTPException(422, problem)
    user = ctx.db.user_by_email(body.email.strip())
    # The success path hashes a new password; every refusal path does the same work first.
    new_hash = crypto.hash_password(body.new_password)
    if user is None or not crypto.recovery_code_matches(body.recovery_code, user.recovery_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, RECOVERY_REFUSED)
    new_code = crypto.new_recovery_code()
    version = ctx.db.consume_recovery_code(
        user.id,
        crypto.hash_recovery_code(body.recovery_code),
        new_hash,
        crypto.hash_recovery_code(new_code),
    )
    if version is None:  # raced with another use of the same code
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, RECOVERY_REFUSED)
    session = _session(ctx, user, version)
    return SessionWithRecovery(**session.model_dump(), recovery_code=new_code)


@public.post("/auth/logout")
def logout() -> dict[str, object]:
    """Client-side, and it says so. Claiming a server-side logout would be worse than admitting it."""
    return {"logged_out": False, "why": LOGOUT_EXPLAINED}


# --- gated -----------------------------------------------------------------------------------------


@gated.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> UserOut:
    return _user_out(user)


@gated.post("/auth/change-password", response_model=Session)
def change_password(
    body: PasswordChange, request: Request, user: User = Depends(current_user)
) -> Session:
    ctx = context(request)
    if not crypto.verify_password(body.current_password, user.password_hash):
        # 403, NEVER 401. The session is fine; the proof was wrong. A 401 is read by the client
        # as a dead session and signs the user out in the middle of what they were doing.
        raise HTTPException(status.HTTP_403_FORBIDDEN, "the current password is not correct")
    problem = _password_problem(body.new_password)
    if problem:
        raise HTTPException(422, problem)
    version = ctx.db.set_password(user.id, crypto.hash_password(body.new_password))
    # A replacement token, or the user is signed out by their own action: the bump above has just
    # revoked the token they sent this request with.
    return _session(ctx, user, version)
