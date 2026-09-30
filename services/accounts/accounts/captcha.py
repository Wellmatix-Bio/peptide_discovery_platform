"""A self-hosted SVG captcha. Say plainly what it is worth.

IT IS TEXT. The answer is drawn with SVG <text> nodes, and anybody who parses the SVG's text
nodes solves it - `tests/test_captcha.py` does exactly that, so this paragraph cannot quietly
become an overstatement. It stops drive-by bot registration and nothing more. If more is needed,
the answer is a real provider, not a cleverer SVG.

THE NONCE IS SINGLE-USE WHETHER THE ANSWER WAS RIGHT OR WRONG, and it is marked spent BEFORE the
answer is compared. It lives in SQLite, never in a module-level structure: this module holds no
state at all, and a test asserts that property.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

from .db import Database

# No 0/O, 1/I/L, 5/S, 2/Z: a captcha people misread is a captcha people abandon.
ALPHABET = "ABCDEFGHJKMNPQRTUVWXY346789"
LENGTH = 5


@dataclass(frozen=True)
class Challenge:
    nonce: str
    svg: str
    expires_in: int


def _answer_hash(nonce: str, answer: str) -> str:
    # Bound to the nonce, so one answer hash tells nothing about another challenge.
    normalised = "".join(answer.split()).upper()
    return hashlib.sha256(f"{nonce}:{normalised}".encode("utf-8")).hexdigest()


def _render(text: str) -> str:
    width, height = 170, 56
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}" role="img" aria-label="captcha">',
        f'<rect width="{width}" height="{height}" rx="8" fill="#eaf3f7"/>',
    ]
    for _ in range(6):
        x1, y1 = secrets.randbelow(width), secrets.randbelow(height)
        x2, y2 = secrets.randbelow(width), secrets.randbelow(height)
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#9fb6c3" stroke-width="1.2"/>'
        )
    for index, char in enumerate(text):
        x = 18 + index * 29 + secrets.randbelow(6)
        y = 38 + secrets.randbelow(8) - 4
        angle = secrets.randbelow(40) - 20
        parts.append(
            f'<text x="{x}" y="{y}" transform="rotate({angle} {x} {y})" font-size="28"'
            f' font-family="monospace" font-weight="700" fill="#153c55">{char}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def issue(db: Database, ttl_seconds: int) -> Challenge:
    nonce = secrets.token_urlsafe(24)
    text = "".join(secrets.choice(ALPHABET) for _ in range(LENGTH))
    now = time.time()
    db.store_captcha(nonce, _answer_hash(nonce, text), now + ttl_seconds, now)
    return Challenge(nonce=nonce, svg=_render(text), expires_in=ttl_seconds)


def check(db: Database, nonce: str, answer: str) -> bool:
    """Spend first, compare second. An unknown, expired or already-spent nonce is simply False."""
    stored = db.spend_captcha(nonce, time.time())
    if stored is None:
        return False
    return hmac.compare_digest(stored, _answer_hash(nonce, answer))
