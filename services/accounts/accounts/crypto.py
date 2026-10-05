"""Password hashing, recovery codes and session tokens. Standard library only.

No passlib, no python-jose, no authlib: everything security-relevant here is auditable in one
sitting and carries no third-party dependency to track.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass

# scrypt, memory-hard. n=2^14, r=8 needs 16 MiB, inside hashlib's default 32 MiB ceiling.
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_DKLEN
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(derived)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time. Parameters are read from the stored string, so they can be raised later."""
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_unb64(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(_unb64(expected)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, _unb64(expected))


# A real hash of a password nobody knows, made once per process. Verifying against it costs
# exactly what verifying a real account costs, which is the point of it.
_EQUAL_WORK_HASH = hash_password(secrets.token_urlsafe(24))


def spend_equal_work(password: str) -> None:
    """Do the work a real verification does, and discard the answer.

    Called on every path that refuses WITHOUT having verified a password - above all the
    unknown-email path. A dictionary miss returns in microseconds and a real scrypt takes tens of
    milliseconds; that gap is measurable over a network and turns the login form into an
    account-enumeration oracle even when the message is identical.
    """
    verify_password(password, _EQUAL_WORK_HASH)


# --- recovery codes --------------------------------------------------------------------------------

RECOVERY_BYTES = 10  # 80 bits -> 16 base32 characters


def new_recovery_code() -> str:
    """16 base32 characters shown as XXXX-XXXX-XXXX-XXXX. Shown once; only its hash is kept."""
    raw = base64.b32encode(secrets.token_bytes(RECOVERY_BYTES)).decode("ascii").rstrip("=")
    return "-".join(raw[i : i + 4] for i in range(0, len(raw), 4))


def normalise_recovery_code(code: str) -> str:
    return "".join(ch for ch in code.upper() if ch.isalnum())


def hash_recovery_code(code: str) -> str:
    # 80 random bits cannot be brute-forced offline, so a fast hash is enough; the slow hash is
    # for passwords, which people choose.
    return hashlib.sha256(normalise_recovery_code(code).encode("ascii")).hexdigest()


def recovery_code_matches(code: str, stored: str | None) -> bool:
    if not stored:
        return False
    return hmac.compare_digest(hash_recovery_code(code), stored)


# --- session tokens ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Claims:
    user_id: int
    password_version: int
    issued_at: int
    expires_at: int


class BadToken(ValueError):
    """Anything wrong with a token. Deliberately one class: the caller gets one answer."""


def issue_token(key: bytes, user_id: int, password_version: int, ttl_seconds: int) -> str:
    now = int(time.time())
    payload = json.dumps(
        {"uid": user_id, "pv": password_version, "iat": now, "exp": now + ttl_seconds},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    signature = hmac.new(key, payload, hashlib.sha256).digest()
    return f"{_b64(payload)}.{_b64(signature)}"


def read_token(key: bytes, token: str) -> Claims:
    """Verify the signature FIRST, then parse. An unsigned payload is never interpreted."""
    try:
        encoded_payload, encoded_signature = token.split(".")
        payload = _unb64(encoded_payload)
        signature = _unb64(encoded_signature)
    except (ValueError, TypeError) as problem:
        raise BadToken("malformed") from problem
    expected = hmac.new(key, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise BadToken("signature")
    try:
        body = json.loads(payload)
        claims = Claims(
            user_id=int(body["uid"]),
            password_version=int(body["pv"]),
            issued_at=int(body["iat"]),
            expires_at=int(body["exp"]),
        )
    except (ValueError, TypeError, KeyError) as problem:
        raise BadToken("payload") from problem
    if claims.expires_at <= int(time.time()):
        raise BadToken("expired")
    return claims
