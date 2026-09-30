# Frontend + accounts: decisions record

Build spec: `FRONTEND_SPEC_TEMPLATE.md` (derived from the wmxccs CCS/glycan project).
Reference implementation on disk: `/home/ajit/Documents/wmxccs`.
Branch: `frontend-accounts`. Backend (`src/`, `model_store/`) is not modified except
where §0.1 below explicitly waives that.

## §0 Decisions (owner: ajit@wellmatix.co, 2026-09-30)

| # | Question | Decision |
|---|---|---|
| 0.1 | Model provenance, absent from the API | **Add read-only endpoints to the backend**: `GET /healthz` and a model-manifest endpoint serving each predictor's name/version/weight digest. This is a deliberate, scoped waiver of "do not modify the backend" — additive and read-only only. |
| 0.2 | Who may register | **Open registration**, captcha-gated (§4.2), no allowlist. |
| 0.3 | Per-user job quotas | **None.** See the warning below. |
| 0.4 | `request_id` | **Proxy generates it.** The user supplies a friendly run name stored only in the accounts DB; the proxy sends `request_id = u<user_id>:<uuid>` and strips the prefix on the way out. Closes both the cross-user collision and the path-traversal exposure described in §0.7. |
| 0.5 | Design | Theme tokens, colours and CSS taken from `/home/ajit/Documents/wmxccs/web/src/styles.css`; UI designed to cover every endpoint, with hints, autofill and example briefs. |
| 0.6 | Visibility | Results and history are **private per user** (§5 ownership + history-by-observation apply in full). |
| 0.7 | Run duration | **Unknown / varies by brief.** Design for the slow case: reload-safe run pages, slow polling, history as a primary surface, "safe to close this tab". |
| 0.8 | Fixtures | Captured from the **live GCP deployment** (§7), not hand-written. |
| 0.9 | Scope now | **Phases 0–4**, verified under local `docker compose`. Phase 5 (GCP deploy) delivered as a runbook only. |
| 0.10 | Job actions in the UI | **All four**: cancel (ownership-checked, confirm dialog), re-run/duplicate a brief, hide from my history, export results JSON. |

## Cost warning (arising from 0.2 + 0.3)

Every `POST /api/v1/jobs/create` starts a billable Vertex AI Custom Job on an
NVIDIA L4 (`g2-standard-4`). With open registration and no per-user quota, any
person who can reach the web app and register can start GPU jobs that bill to
the Vertex project. **Set a hard billing cap / budget alert on the Vertex project
before this stack is reachable from the internet.** This is recorded as an
accepted risk, not an oversight.

## Backend findings that shaped the design

1. **No model identity in any response.** `ModelRegistry` (`src/common/model_registry.py`)
   is an explicit no-op stub; only 6 of 16 `model_store/*/` directories carry a
   `model_card.json`. Hence decision 0.1.
2. **`request_id` reaches a GCS path unsanitized.** `src/backend/api_e2e/api.py`
   builds `<artifacts>/requests/<request_id>/config.json`, and `storage.join`
   (`src/common/storage.py`) is string concatenation with no normalization, so
   `..` segments escape the intended prefix and two callers sharing an id
   overwrite each other's staging config. Hence decision 0.4.
3. **Results are addressable by job id alone.** `/results` and `/cancel` read and
   act on GCS purely from the Vertex job resource name, so §5.1 ownership is
   what makes per-user privacy real.
4. **FastAPI's `/docs`, `/redoc`, `/openapi.json` are live** on the API and must be
   refused by the proxy (§5). There are no global list endpoints to refuse.
5. **The API is asynchronous.** Create returns 202 with a Vertex resource name;
   status and results are polled from GCS artifacts. There is no synchronous
   predict call, so §7's rules apply to stored run results.
6. **The `Brief` schema is small and enum-driven** (`Literal` types in
   `src/schemas/stage_configs.py`), and `data/briefs/` holds 32 example briefs —
   so §6.3's "build the form from the API's validators" is fully achievable.

## Baseline (§2 phase 0)

- Local venv is Python 3.12.13 with the **CPU** PyTorch wheel
  (`--extra-index-url https://download.pytorch.org/whl/cpu`) rather than the
  CUDA 12.8 wheel the worker Dockerfiles select. The baseline suite does not
  need a GPU; the deployed worker image is unaffected.
- Model version at baseline: **none exists** — that is the subject of decision 0.1.
  The baseline is therefore the backend test suite result alone, recorded in
  `docs/BASELINE.md`.

## Later decisions

| # | Question | Decision |
|---|---|---|
| 0.11 | Real Vertex runs for fixtures | **Reuse past artifacts first.** Look in `VERTEX_ARTIFACTS_DIR` for completed runs and derive §7 fixtures from those; ask before submitting any fresh billable job. |
| 0.12 | Where the provenance endpoints land | **Their own commit** on this branch, isolated for independent review or revert. Backend suite re-run either side. |
| 0.13 | Missing `example_request.json` | **Reconstruct** from `API_SPECIFICATION.md` + `src/schemas/stage_configs.py`. Second scoped waiver of "do not modify the backend" (a new file under `src/`, test-input data only). |
| 0.14 | Broken `s4pred` submodule | **Leave untouched**, exclude its 47 tests from the baseline, file for the backend owner. GPL-3.0 makes vendoring a licensing call that is not mine. |
| 0.15 | `docs/` was gitignored | **Un-ignore it.** The `docs/` line is removed from `.gitignore`; `!docs/**/*.pdf` is kept so tracked PDF reports behave exactly as before. |
| 0.16 | 7 stale `test_api_e2e.py` failures | **Fix them, in their own commit.** Test-side only: set the three machine env vars in the fixture, mock `storage.exists`, drop the nonexistent `config_path`. No `api.py` change. Restores the spec's before/after guard. |
