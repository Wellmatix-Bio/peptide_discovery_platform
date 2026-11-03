# DevOps handover

What a platform team normally asks for before taking this on, and where each item actually stands.
Every row was checked against the code on the date below, not against intent. Where something is
partly done, the row says which part.

Checked: 2026-11-03, against `main`.

| Requirement | Status |
|---|---|
| **Stateless service, runs from a Dockerfile** | **Done.** `deploy/api.Dockerfile` — non-root (uid 10001), no volumes, no writable state. Configuration is entirely environment variables, read through `required()` at request time; nothing is baked into the image and nothing is read from a local file. The image holds code and `model_store/*/model_card.json` only — **no model weights**, which are synced to the worker from `VERTEX_MODEL_STORE` at run time. |
| **Honours the platform's port** | **Done.** `CMD` uses `${PORT:-8080}`, so Cloud Run's injected `$PORT` is used and compose gets 8080. |
| **Container healthcheck** | **Done.** `HEALTHCHECK` hits `/healthz`, which reports per-pipeline readiness: which of the three pipelines would answer 503 and exactly which settings are missing. |
| **Vertex status and cancel by job name** | **Done.** `GET /api/v1/jobs/{job_name}/status` and `POST /api/v1/jobs/{job_name}/cancel`, both taking the full Vertex resource name (`projects/…/locations/…/customJobs/…`; the route is a `:path` so the slashes are fine). Status returns the worker's self-report **and** the live Vertex state **and** `run_state`, which reconciles the two — see the row below. |
| **Outputs in Cloud Storage, no local storage** | **Done in the deployed path.** Everything goes through `src/common/storage.py`, which writes `gs://` URIs. It falls back to local paths **only when `DEV_MODE` is set**, which is for running the pipeline without a cloud account; `DEV_MODE` must not be set in a deployment. CI sets it deliberately and says so. |
| **Request ID echoed, results retrievable by it** | **Partly done.** `request_id` is required on create, echoed in the create response, used as the GCS staging prefix (`<artifacts>/requests/<request_id>`), and attached to the Vertex job as the `request-id` label. **There is no `GET /api/v1/requests/{id}`** — results are fetched by Vertex job name. A caller that wants its own ID as the lookup key must keep the ID → job-name mapping, which is what the accounts service does: it rewrites `request_id` to `u<user id>:<uuid>`, keeps the caller's text as the run name, and serves `GET /auth/history` per account. If the platform team needs lookup by their own ID, that endpoint is the gap to close. |
| **Under one hour per request** | **Not established, and not measured here.** No end-to-end wall-clock figure has been recorded in this repository for any of the three pipelines, so there is no number to hand over. What is known: stage 7 (ESMFold) dominates, the candidate count is the main driver, and both are controllable from the request — the web form exposes "candidates to generate", and any stage can be turned off. A run's duration therefore depends on the brief, not only on the machine. **Treat this row as an open measurement, not a yes.** |
| **Auth under the platform's own system** | **Not done. No seam exists yet.** The job API has *no authentication of its own* — deliberately: `deploy/compose.yaml` publishes no host port for it, and `services/accounts/` is the only route in. But the accounts service is a **self-contained** identity system (its own user table, bcrypt passwords, bearer tokens, captcha, rate limiting), not an adapter. There is no `AUTH_MODE`, no OIDC/JWT verification against an external issuer, and no trusted-header mode. Wiring this to an existing IdP means adding one of those to `services/accounts/accounts/auth.py` — the token check is already a single FastAPI dependency, so it is a contained change, but it is a change, not configuration. |
| **Reusable frontend** | **Partly.** `web/` is a complete React + TypeScript + Vite application on one origin with no CORS, and its API layer is generated from the OpenAPI snapshot, so it is reusable *as an app*. It is **not** packaged as an embeddable component: styling is one global `web/src/styles.css` imported in `main.tsx`, not scoped or namespaced, and routing assumes it owns the page. Dropping it inside another product's shell would need the CSS scoped first. |
| **Images built reproducibly in CI** | **Partly.** The `images` job in `.github/workflows/ci.yml` runs `docker compose build`, which builds the API, accounts and web images. It builds **none of the three worker images** — `src/backend/worker_*/Dockerfile` are built and pushed by hand. A broken worker Dockerfile is therefore discovered when a Vertex job fails, after submission. This is the largest remaining CI gap. |
| **Secrets kept out of the image and the repo** | **Done.** `.env*` is gitignored except the examples; `deploy/cloudrun/env.sh` likewise. `docs/DEPLOYMENT.md` uses Secret Manager for the VM path. `docs/HISTORY_SCAN.md` records a pre-publication scan of the git history for secrets and what it found. |
| **One place that lists every setting** | **Done.** `.env.example`, `deploy/.env.example` and `deploy/cloudrun/env.sh.example` carry every variable with a comment, including the per-pipeline pairs. `/healthz` reports which are missing at run time. |

## The two rows to argue about first

**Auth.** Everything else here is deployment mechanics. This one is a code change, and the shape
of it depends on the platform team's answer: OIDC token verification, a trusted header from their
gateway, or keeping the accounts service and federating it. The seam is small — one FastAPI
dependency — but someone has to pick.

**The hour.** Nobody has measured a full run end to end and written the number down. Before
promising a time, run each pipeline once at the candidate count the platform expects and record
it; the figure will be specific to that count, and that is the honest way to quote it.

## Per-pipeline deployment state

All three pipelines are routed by one endpoint, `POST /api/v1/jobs/create`, discriminated by a
`pipeline` field. Each needs **two** settings of its own, and both are hard requirements — setting
only the image yields a 503 that names the *seed file*, which reads like a different fault:

| Pipeline | Image setting | Seed-file setting |
|---|---|---|
| `wound_healing` | `WORKER_IMAGE_URI` | `SEED_CANDIDATES_FILE` |
| `metabolic` | `WORKER_METABOLIC_IMAGE_URI` | `SEED_CANDIDATES_FILE_METABOLIC` |
| `cosmetic` | `WORKER_COSMETIC_IMAGE_URI` | `SEED_CANDIDATES_FILE_COSMETIC` |

All six are present in `deploy/compose.yaml`, `deploy/cloudrun/deploy.sh` and both example env
files. The three worker images themselves still have to be built and pushed by hand.

## Known limitations a platform team should be told about, not discover

- `status` from the worker can be stale; a dead worker leaves it reading `"running"` forever. Show
  **`run_state`**, which the API reconciles against the live Vertex state — the rule is one named
  function, `run_state()` in `src/backend/api_e2e/api.py`, mirrored in TypeScript and held to the
  same table by `tests/unit/test_run_state_parity.py`.
- GIPR potency predictions are returned and **flagged as not usable for ranking** (leave-one-cluster-out
  R² is −0.108). They still count toward the gates and the ranking score. See `docs/LIMITATIONS.md`.
- The backend test suite cannot be green: seven tests need model weights that are gitignored and
  synced at run time. CI compares the outcome to a recorded baseline instead — `docs/BASELINE.md`.
