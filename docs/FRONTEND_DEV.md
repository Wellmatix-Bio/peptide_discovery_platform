# Running and testing the web app locally

Three ways, in increasing order of how much of the real system they involve. Pick the first one
that answers your question — the later ones need more running and tell you less about the code.

Everything here is run from `web/`, which needs **Node 22** (the version CI uses) and nothing else.

```bash
cd /home/ajit/Documents/peptide_discovery_platform/web && npm ci --no-audit --no-fund
```

---

## 1. The tests — the fastest and the most informative

These render the real components against real captured API responses, in jsdom. **No server, no
backend, no sign-in.** This is where a change to a page is actually verified, and it is what CI
runs.

```bash
cd /home/ajit/Documents/peptide_discovery_platform/web && npm test
```

Watch one file while you edit it:

```bash
cd /home/ajit/Documents/peptide_discovery_platform/web && npx vitest src/test/submitrun.test.tsx
```

What the suites cover, so you know which one to look at:

| File | What it holds you to |
|---|---|
| `runview.test.tsx` | The run page against responses **captured from the running API** (`src/test/fixtures/`) — the API's own counts, no invented ordering, withheld values rendered as "No value" and never as 0 |
| `runview-pipelines.test.tsx` | Metabolic and cosmetic results render *their own* fields, the GIPR flag is visible before anything is expanded, and an untagged response reads as wound healing |
| `submitrun.test.tsx` | What **bytes reach the API**: the right `pipeline` tag and the right stage names per pipeline, checks that refuse a bad brief, and the JSON tab taking its pipeline from the pasted text |
| `examples.test.ts` | Every example brief the form offers is valid against the OpenAPI snapshot — the repo's own `data/briefs/` are not, which is why the app ships its own |
| `runstate.test.ts` | The TypeScript `runState()` table. `tests/unit/test_run_state_parity.py` reads this same table and runs the Python implementation against it |
| `client.test.ts`, `workspace.test.ts`, `design.test.ts` | Fetch wrapper, local workspace state, and the design system comparison |

Typecheck and the production build, both of which CI gates on:

```bash
cd /home/ajit/Documents/peptide_discovery_platform/web && npm run typecheck && npm run build
```

---

## 2. The dev server, with nothing behind it

```bash
cd /home/ajit/Documents/peptide_discovery_platform/web && npm run dev
```

Vite serves on <http://localhost:5173>. **You will land on the sign-in page and you will not get
past it** — the app confirms its session against the accounts service, and nothing is running.
That is the correct behaviour, not a bug, and it is why the tests above are the real verification
route for a page.

Useful for: CSS, layout, responsive behaviour, and anything on the sign-in and registration pages
themselves.

Not useful for: the run form, the run view, history, or anything that needs a session. Use the
tests, or stack 3.

---

## 3. The whole stack, in Docker

This is the only way to click through the app as a signed-in user. It runs the web app, the
accounts service and the job API behind nginx on one origin.

```bash
cd /home/ajit/Documents/peptide_discovery_platform && cp deploy/.env.example deploy/.env
```

Then edit `deploy/.env`. The minimum to get the app up and let you sign in:

- `ACCOUNTS_SECRET_KEY` — any 32+ byte string for local use
- the `VERTEX_*` and `WORKER_*` values can stay as placeholders **if you are not submitting runs**

```bash
cd /home/ajit/Documents/peptide_discovery_platform && docker compose -f deploy/compose.yaml up --build
```

The app is on <http://localhost:8088> (`WEB_PORT` in `deploy/.env` changes it; it binds
loopback only unless you set `WEB_BIND`). Register an account in the UI — the accounts service holds
its own users, so there is no external identity provider to configure.

What works without any Google Cloud setup: registration, sign-in, the account pages, the submit
form including all three pipelines' validation, and history (which will be empty).

What does **not**: submitting a run. `POST /api/v1/jobs/create` reads its settings through
`required()` and answers **503** naming the first one missing. The job API's `/healthz` tells you
exactly which, per pipeline — but it is **not reachable from your machine**: the API publishes no
host port, and the accounts proxy requires a session, so ask the container directly:

```bash
cd /home/ajit/Documents/peptide_discovery_platform && docker compose -f deploy/compose.yaml exec api python -c "import urllib.request,json;print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8080/healthz')),indent=2))"
```

`http://localhost:8088/healthz` is a different thing — nginx routes that to the **accounts**
service's own health check, not the job API's.

To actually submit a run you need a real `VERTEX_CLOUD_PROJECT`, a bucket for
`VERTEX_ARTIFACTS_DIR`, a pushed worker image and a seed file — `docs/DEPLOYMENT.md` has that
path, and each run starts a billed GPU job.

---

## Changing the API and the types

The web app's types are **generated** from a snapshot of the API, and CI fails on any drift. After
changing a response model in `src/backend/api_e2e/api.py`:

```bash
cd /home/ajit/Documents/peptide_discovery_platform && .venv/bin/python web/scripts/openapi.py peptide
```

```bash
cd /home/ajit/Documents/peptide_discovery_platform/web && npm run api:types && npm run typecheck
```

The first regenerates `web/openapi/peptide.json`; the second regenerates
`web/src/api/peptide.gen.ts` from it. Never edit either by hand. A field you renamed server-side
shows up here as a compile error, which is the point.

Two separate CI gates check this, and they fail for different reasons:

- `openapi.py peptide --check` — the snapshot is stale relative to `api.py`
- `npm run api:types` then `git diff --exit-code` — the generated types are stale relative to the
  snapshot

## Before you push

What CI will run, in the order it runs it:

```bash
cd /home/ajit/Documents/peptide_discovery_platform && .venv/bin/python web/scripts/openapi.py peptide --check && cd web && npm run api:types && git diff --exit-code -- src/api/peptide.gen.ts && npm run typecheck && npm test && npm run build
```

The backend side is a separate job; `CONTRIBUTING.md` and `docs/BASELINE.md` cover it, and the
short version is that the suite **cannot** be green — it is compared to a recorded baseline, and a
new *pass* fails CI just as a new failure does.
