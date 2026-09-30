# accounts

Email/password accounts, sessions and an authenticating proxy in front of the
peptide job API (`src/backend/api_e2e/api.py`), which has no authentication of
its own and gains none.

**Demo-scoped.** To be deleted, not grown, when a platform identity system
replaces it. Before adding anything here, ask whether it is worth building twice.

```
browser ──► web (nginx) ──► accounts ──► job API (no host port, no auth)
                 /auth/*   register, login, recovery, captcha
                 /api/peptide/*   the passthrough
```

## Run it

```bash
ACCOUNTS_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))') PEPTIDE_UPSTREAM=http://127.0.0.1:8080 ../../.venv/bin/python -m uvicorn accounts.app:build --factory --port 8081
```

Tests (they need no credentials and never reach GCP):

```bash
.venv/bin/python -m pytest -q services/accounts/tests
```

## What protects what

**The proxy is the only route to the job API, and that — not the web app — is what
makes "the backend has no auth" safe.** The API must publish no host port.
*Protected by the frontend is not protection.*

Security primitives are **standard library only**: `hashlib.scrypt` (n=2¹⁴, r=8,
p=1), `hmac.compare_digest`, `secrets`, `sqlite3`. No passlib, no python-jose, no
ORM.

Three narrow exceptions to the dumb passthrough exist because the API has no
concept of a user. Each is a shape match, not a copy of its route table:

1. **Ownership.** A Vertex job resource name in the path or query must belong to
   the caller, else 404 — the same 404 for a foreign run and a nonexistent one.
   Without this, any signed-in user could read or cancel any other user's run,
   because `/status`, `/results` and `/cancel` act on a job from its id alone.
2. **History by observation.** No endpoint creates a history row; rows are written
   only by observing responses the proxy forwarded.
3. **`request_id` is replaced, not prefixed.** The API interpolates it into a GCS
   path with no normalization, so a caller-chosen value can carry `..` out of the
   intended prefix and two callers can overwrite each other's staging config. The
   proxy sends `u<user_id>:<uuid4>` and keeps the name you typed locally.

## Say it plainly

- **The captcha is text in an SVG and a parser solves it.**
  `tests/conftest.py`'s `solve()` is that parser, and the test suite uses it. It
  stops drive-by bots and nothing more.
- **Logout revokes nothing server-side.** Sessions are stateless signed tokens.
  `/auth/logout` says so in its own response. Changing the password bumps
  `password_version`, which is the only real revocation.
- **SQLite means one instance.** Several uvicorn workers in one container are
  fine (WAL, busy timeout, all shared state in the database). Several VMs or
  replicas will split or corrupt the data.
- **No account is ever locked.** Over a rate limit is 429 with `Retry-After`;
  locking would let anyone who knows an email lock its owner out.
- **History rows go stale.** A run takes minutes to hours, and this service does
  not poll on your behalf — a row is only as current as the last response its
  owner passed through the proxy. Every history response carries a `staleness`
  note saying so.
- **`vertex_state` is the only truthful completion signal.** The API derives
  `status` solely from the worker's `results.json`, so a worker that dies without
  writing it reports `"pending"` forever. Rows record both, and a run Vertex has
  finished while `status` is still `"pending"` is called out in its summary.
- **Hiding a run does not delete it.** `DELETE /auth/history/{id}` removes only
  this account's history entry; the run's artifacts stay in Cloud Storage, which
  this service cannot touch. Never label it "delete".

## Ownership is not released when a run is hidden

Deliberate. Forgetting ownership would let the next account that observes that
job id claim it.

## Files

| File | Holds |
|---|---|
| `crypto.py` | scrypt hashing, token signing, equal-work |
| `captcha.py` | self-hosted SVG challenge |
| `ratelimit.py` | fixed-window counters, HMAC'd email buckets |
| `db.py` | all shared state: users, captchas, ownership, history, counters |
| `auth.py` | the public and gated routers |
| `proxy.py` | the passthrough and the history read side |
| `history.py` | recording by observation |
| `config.py` | settings and the startup checks that fail loudly |

Ported from the wmxccs project's accounts service; `crypto.py`, `captcha.py`,
`ratelimit.py` and `auth.py` are substantially unchanged. `proxy.py` and
`history.py` were rewritten, because that API was synchronous and this one is not.
