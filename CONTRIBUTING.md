# Contributing

Thanks for looking. This file covers getting set up, what the tests expect, and the few things
about this repository that will waste your afternoon if nobody tells you.

---

## Setup

Python 3.12, Node 22 for the web app, Docker only if you want the full stack.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
```

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt pytest
```

The CPU PyTorch index matters: the default wheel is the CUDA build and several GB. The worker
image selects the CUDA wheel itself; nothing about local development needs it.

Run the pipeline:

```bash
python main.py configs/runs/test_run.yaml
```

---

## Tests

Three suites, run separately.

```bash
DEV_MODE=true .venv/bin/python -m pytest -q --continue-on-collection-errors --ignore=services --ignore=web
```

```bash
.venv/bin/python -m pytest -q services/accounts/tests
```

```bash
cd web && npm ci && npm test
```

### The backend suite does not pass, on purpose

It is **82 passed / 7 failed / 0 errors**, and that exact result is the baseline. CI compares
against it with `.github/check_baseline.py`, which fails on *any* movement — more failures mean a
regression, more passes mean the baseline is stale and must be updated in the same change.

The 7 are documented in [docs/BASELINE.md](docs/BASELINE.md): 4 are stale `test_api_e2e.py`
expectations that disagree with the shipped API, and 3 need model weights that a fresh clone does
not have.

**A count only means something against a stated environment.** The checker asserts its assumptions
before counting — required modules present, model weights absent, the optional copyleft packages
absent — so a dependency gap reports itself instead of looking like a regression. If you see it
complain about your environment, it is telling you the truth.

### `DEV_MODE=true` is not optional for the backend suite

`tests/integration/test_worker_e2e.py` needs it. Nothing documented this before; it does now.

---

## Things that will trip you up

**`s4pred` does not resolve.** It is committed as a git submodule pointer with no `.gitmodules`
entry, so `git submodule update --init` cannot fetch it. It is optional — stage 5's
secondary-structure screen reports itself unavailable — and the pipeline imports fine without it.
If you need that screen, obtain s4pred from upstream and place it in
`src/pipeline/s05_physchem_screening/s4pred/`, accepting its GPL-3.0 terms.

**Two dependencies are copyleft and deliberately absent.** `propy3` (GPL-2.0-only) and `s4pred`
(GPL-3.0). Do not add either to `requirements.txt` or to `pyproject.toml`'s core dependencies —
`tests/test_licensing.py` fails if you do, and for good reason: it would make every built image a
combined work under that licence. [docs/LICENSING.md](docs/LICENSING.md) has the detail.

**Model weights are not in the repository** and several predictors cannot run without them.

**Port 8080 is the job API's default** and is a popular port. Check what is listening before you
blame the code; a proxy pointed at the wrong service fails in confusing ways.

---

## House rules for changes

**A check that did not run is never reported as a pass.** This is the rule the codebase bends
hardest to keep. If a model is unavailable, a screen is skipped, or a value is withheld, say so —
`not_screened`, `available: false`, `"No value"` — never a default, a dash, or a zero. A stage-8
bug where a null aggregation score read `else "pass"` is exactly what this rule exists to prevent.

**Do not compute in the UI what the API did not compute.** Every number the web app shows is a
field of a response. Where the API's own counts disagree with the data, show both and reconcile
neither.

**Prefer the API's own words.** Error text, refusal reasons and caveats come from the service that
produced them, rendered rather than paraphrased.

**Say what is not known.** Seven predictors ship no model card, and the UI says which. That is
better than a page implying all fifteen are documented.

---

## Tests for new work

New behaviour needs a test that fails without it. For anything load-bearing — a privacy boundary,
a safety screen, an honesty rule — **break it deliberately and confirm the test fails**. Two
harnesses already do this and are worth reading as examples:

- `services/accounts/tests/break_guards.sh` sabotages the accounts service 17 ways and fails if
  any sabotage goes undetected.
- `tests/test_licensing.py` guards the licence posture; each check was verified by breaking it.

Fixtures for result views are **captured from a running API**, never hand-written — see
`web/src/test/fixtures/README.md`. A hand-written fixture encodes what its author believed the API
returns, so the test passes while the app misrenders the real thing. Capturing is what revealed
that `stage` reads `pending` on a succeeded run.

---

## Commits and pull requests

Explain **what changed, why, and how you verified it**. If you found something while working that
you are not fixing, say so rather than leaving it for the next person to rediscover.

CI runs three jobs: the Python suites and the baseline check, the web typecheck/test/build with
generated types regenerated and compared, and a build of all three images.

---

## Reporting something

- **Security** — do not open a public issue. See [SECURITY.md](SECURITY.md).
- **A bug** — what you ran, what happened, what you expected. The run's `vertex_state` and
  `status` both, if it involves a job.
- **A question** — GitHub Discussions.
