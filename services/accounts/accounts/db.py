"""SQLite storage. ONE INSTANCE ONLY.

Several workers in one container share this file safely (WAL, busy timeout). Replicas on
separate hosts or volumes will corrupt or split the data. Say so before somebody scales it.

ALL shared state lives here - accounts, captcha nonces - and none in process memory. A
module-level dict works in every single-process test and fails behind gunicorn/uvicorn workers,
where a challenge issued by one worker is validated by another.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Uniqueness in the SCHEMA, case-insensitively. A code path added later cannot forget a
    -- database constraint the way it can forget an application-level check.
    email            TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash    TEXT    NOT NULL,
    -- Embedded in every token. Bumping it is the ONLY way to revoke a stateless token.
    password_version INTEGER NOT NULL DEFAULT 1,
    recovery_hash    TEXT,
    created_at       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS captchas (
    nonce       TEXT    PRIMARY KEY,
    answer_hash TEXT    NOT NULL,
    expires_at  REAL    NOT NULL,
    spent       INTEGER NOT NULL DEFAULT 0
);

-- Who owns an upstream resource id. Written ONLY by observing a response the proxy forwarded,
-- never by a request. First observer wins and is never replaced (PRIMARY KEY + INSERT OR IGNORE).
CREATE TABLE IF NOT EXISTS owned (
    resource_id TEXT    PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    service     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);

-- What a user ran, by observation. There is NO endpoint that creates a row here.
--
-- UNLIKE the synchronous service this was adapted from, a row here is NOT terminal when
-- written. A peptide run is submitted, then takes minutes to hours; the row is created from
-- the create response (status "pending") and updated whenever this user's later polling
-- passes a status or results response through the proxy. Hence the staleness contract:
-- status/stage/vertex_state are as of observed_at, NOT as of the read. A run whose owner
-- stops polling stays at whatever was last seen. The run view says so.
CREATE TABLE IF NOT EXISTS history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id),
    service       TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    -- The full Vertex Custom Job resource name, which is the only id the API accepts back.
    resource_id   TEXT    UNIQUE,
    -- The namespaced request_id the proxy generated and sent upstream (u<user_id>:<uuid>),
    -- and the name this user typed for the run, which never leaves this database.
    request_id    TEXT,
    run_name      TEXT,
    status_code   INTEGER NOT NULL,
    summary       TEXT    NOT NULL,
    -- Mutable, by observation. status/stage come from the worker's results.json;
    -- vertex_state is the real Vertex CustomJob state and is the ONLY field that tells the
    -- truth about a worker that died without writing results.json (see docs/BASELINE.md).
    status        TEXT,
    stage         TEXT,
    vertex_state  TEXT,
    -- The results body, kept so a finished run reopens without another upstream read.
    envelope      TEXT,
    envelope_note TEXT,
    created_at    TEXT    NOT NULL,
    observed_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS history_by_user ON history (user_id, id DESC);

-- Fixed-window counters for rate limiting. In SQLite, like every other piece of shared state, so
-- the limit holds across workers and restarts. `bucket` never holds a raw email: see ratelimit.py.
CREATE TABLE IF NOT EXISTS rate_counts (
    bucket       TEXT    NOT NULL,
    window_start INTEGER NOT NULL,
    count        INTEGER NOT NULL,
    -- Per row, because the limits have different windows: cleaning up by window_start alone
    -- would let a 15-minute limit delete an hour-long window that is still running.
    expires_at   INTEGER NOT NULL,
    PRIMARY KEY (bucket, window_start)
);
"""


def _like_escaped(value: str) -> str:
    """Neutralise LIKE wildcards in a value that came from a response body."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class User:
    id: int
    email: str
    password_hash: str
    password_version: int
    recovery_hash: str | None
    created_at: str


class EmailTaken(Exception):
    pass


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """A connection per use. Short-lived by design: nothing holds a lock between requests."""
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    # --- users -----------------------------------------------------------------------------------

    def create_user(self, email: str, password_hash: str, recovery_hash: str) -> User:
        try:
            with self.connect() as db:
                cursor = db.execute(
                    "INSERT INTO users (email, password_hash, recovery_hash, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (email, password_hash, recovery_hash, now_iso()),
                )
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError as clash:
            raise EmailTaken(email) from clash
        user = self.user_by_id(int(user_id))
        assert user is not None
        return user

    def user_by_email(self, email: str) -> User | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return None if row is None else User(**dict(row))

    def user_by_id(self, user_id: int) -> User | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return None if row is None else User(**dict(row))

    def set_password(self, user_id: int, password_hash: str) -> int:
        """Replace the hash AND bump the version, which revokes every token issued before."""
        with self.connect() as db:
            db.execute(
                "UPDATE users SET password_hash = ?, password_version = password_version + 1"
                " WHERE id = ?",
                (password_hash, user_id),
            )
            row = db.execute(
                "SELECT password_version FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return int(row["password_version"])

    def consume_recovery_code(
        self, user_id: int, expected_hash: str, password_hash: str, new_recovery_hash: str
    ) -> int | None:
        """Swap the code, the password and the version in ONE statement guarded by the old hash.

        Single-use by construction: a second use of the same code matches no row, even if two
        requests race.
        """
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE users SET password_hash = ?, recovery_hash = ?,"
                " password_version = password_version + 1"
                " WHERE id = ? AND recovery_hash = ?",
                (password_hash, new_recovery_hash, user_id, expected_hash),
            )
            if cursor.rowcount != 1:
                return None
            row = db.execute(
                "SELECT password_version FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return int(row["password_version"])

    # --- ownership and history --------------------------------------------------------------------

    def claim(self, resource_id: str, user_id: int, service: str) -> bool:
        """True if this call recorded the ownership; False if somebody (anybody) already had it."""
        with self.connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO owned (resource_id, user_id, service, created_at)"
                " VALUES (?, ?, ?, ?)",
                (resource_id, user_id, service, now_iso()),
            )
        return cursor.rowcount == 1

    def owns(self, user_id: int, resource_id: str) -> bool:
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM owned WHERE resource_id = ? AND user_id = ?", (resource_id, user_id)
            ).fetchone()
        return row is not None

    def add_history(
        self,
        *,
        user_id: int,
        service: str,
        kind: str,
        status_code: int,
        summary: str,
        resource_id: str | None = None,
        request_id: str | None = None,
        run_name: str | None = None,
        status: str | None = None,
        stage: str | None = None,
        vertex_state: str | None = None,
        envelope: str | None = None,
        envelope_note: str | None = None,
    ) -> None:
        stamp = now_iso()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO history (user_id, service, kind, resource_id,"
                " request_id, run_name, status_code, summary, status, stage, vertex_state,"
                " envelope, envelope_note, created_at, observed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, service, kind, resource_id, request_id, run_name, status_code,
                 summary, status, stage, vertex_state, envelope, envelope_note, stamp, stamp),
            )

    def update_run(
        self,
        *,
        user_id: int,
        resource_id: str,
        status: str | None = None,
        stage: str | None = None,
        vertex_state: str | None = None,
        summary: str | None = None,
        envelope: str | None = None,
        envelope_note: str | None = None,
    ) -> None:
        """Refresh a run row from an observed status or results response.

        Scoped to `user_id` as well as the id, so observing a response can only ever touch the
        observer's own row. COALESCE keeps whatever a call does not carry -- a status poll must
        not wipe a results body already stored, and a results read must not blank the stage.
        observed_at always moves, which is what makes the staleness contract readable.
        """
        with self.connect() as db:
            db.execute(
                "UPDATE history SET status = COALESCE(?, status), stage = COALESCE(?, stage),"
                " vertex_state = COALESCE(?, vertex_state), summary = COALESCE(?, summary),"
                " envelope = COALESCE(?, envelope),"
                " envelope_note = COALESCE(?, envelope_note), observed_at = ?"
                " WHERE resource_id = ? AND user_id = ?",
                (status, stage, vertex_state, summary, envelope, envelope_note, now_iso(),
                 resource_id, user_id),
            )

    def resource_id_for_run(self, user_id: int, run_id: str) -> str | None:
        """The full job resource name this user owns whose numeric tail is `run_id`.

        The results envelope carries only the numeric run id (api.py derives it from the job
        name), so a results observation has to find its row from that. Scoped to the caller and
        matched on the exact `/customJobs/<run_id>` tail, so one user's poll cannot reach
        another's row and `456` cannot match `123456`.
        """
        with self.connect() as db:
            row = db.execute(
                "SELECT resource_id FROM history WHERE user_id = ?"
                " AND resource_id LIKE ? ESCAPE '\\' LIMIT 1",
                (user_id, f"%/customJobs/{_like_escaped(run_id)}"),
            ).fetchone()
        return None if row is None else str(row["resource_id"])

    def hide_run(self, user_id: int, entry_id: int) -> bool:
        """Remove a run from this user's history list. The run's GCS artifacts are NOT touched
        -- this service cannot delete them -- so every caller must say "hide", never "delete".
        Ownership in `owned` is deliberately left in place: forgetting it would let the id be
        claimed by whoever next observes it."""
        with self.connect() as db:
            cursor = db.execute(
                "DELETE FROM history WHERE id = ? AND user_id = ?", (entry_id, user_id)
            )
        return cursor.rowcount == 1

    def history(
        self, user_id: int, *, service: str | None, limit: int, offset: int
    ) -> tuple[int, list[dict]]:
        where, args = "user_id = ?", [user_id]
        if service:
            where += " AND service = ?"
            args.append(service)
        with self.connect() as db:
            total = db.execute(f"SELECT COUNT(*) FROM history WHERE {where}", args).fetchone()[0]
            rows = db.execute(
                "SELECT id, service, kind, resource_id, run_name, status_code, summary,"
                " status, stage, vertex_state,"
                " envelope IS NOT NULL AS has_envelope, envelope_note, created_at, observed_at"
                f" FROM history WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return int(total), [dict(row) for row in rows]

    def history_entry(self, user_id: int, entry_id: int) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM history WHERE id = ? AND user_id = ?", (entry_id, user_id)
            ).fetchone()
        return None if row is None else dict(row)

    # --- rate limiting -----------------------------------------------------------------------------

    def rate_hit(self, bucket: str, window_start: int, expires_at: int, now: float) -> int:
        """Count one event in the window and return the new total, atomically.

        One UPSERT with RETURNING, so two workers counting at the same instant cannot both read
        the old value. Expired windows - of any limit - are deleted on the way.
        """
        with self.connect() as db:
            db.execute("DELETE FROM rate_counts WHERE expires_at <= ?", (now,))
            row = db.execute(
                "INSERT INTO rate_counts (bucket, window_start, count, expires_at) VALUES (?, ?, 1, ?)"
                " ON CONFLICT (bucket, window_start) DO UPDATE SET count = count + 1"
                " RETURNING count",
                (bucket, window_start, expires_at),
            ).fetchone()
        return int(row["count"])

    def rate_count(self, bucket: str, window_start: int) -> int:
        with self.connect() as db:
            row = db.execute(
                "SELECT count FROM rate_counts WHERE bucket = ? AND window_start = ?",
                (bucket, window_start),
            ).fetchone()
        return 0 if row is None else int(row["count"])

    # --- captcha ---------------------------------------------------------------------------------

    def store_captcha(self, nonce: str, answer_hash: str, expires_at: float, now: float) -> None:
        with self.connect() as db:
            # Housekeeping on issue, so an attacker requesting challenges cannot grow the table
            # without bound: anything expired or spent is removed.
            db.execute("DELETE FROM captchas WHERE expires_at < ? OR spent = 1", (now,))
            db.execute(
                "INSERT INTO captchas (nonce, answer_hash, expires_at) VALUES (?, ?, ?)",
                (nonce, answer_hash, expires_at),
            )

    def spend_captcha(self, nonce: str, now: float) -> str | None:
        """Mark the nonce spent and return its answer hash - BEFORE anybody checks the answer.

        Spent whether the answer turns out right or wrong. Otherwise a 1-in-N guess, retried
        against the same nonce, becomes a certainty.
        """
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE captchas SET spent = 1 WHERE nonce = ? AND spent = 0 AND expires_at >= ?",
                (nonce, now),
            )
            if cursor.rowcount != 1:
                return None
            row = db.execute(
                "SELECT answer_hash FROM captchas WHERE nonce = ?", (nonce,)
            ).fetchone()
        return None if row is None else str(row["answer_hash"])
