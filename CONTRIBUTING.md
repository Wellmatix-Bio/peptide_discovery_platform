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

It is **310 passed / 3 failed / 0 errors**, and that exact result is the baseline. CI compares
against it with `.github/check_baseline.py`, which fails on *any* movement — more failures mean a
regression, more passes mean the baseline is stale and must be updated in the same change.

The 3 are documented in [docs/BASELINE.md](docs/BASELINE.md): all need model weights that a fresh
clone does not have, so they fail for the environment rather than for the code.

**A count only means something against a stated environment.** The checker asserts its assumptions
before counting — required modules present, model weights absent, the optional copyleft packages
absent — so a dependency gap reports itself instead of looking like a regression. If you see it
complain about your environment, it is telling you the truth.

### `DEV_MODE=true` is not optional for the backend suite

`tests/integration/test_worker_e2e.py` needs it. Nothing documented this before; it does now.

---

## Things that will trip you up

**`s4pred` is not in the repository.** It used to be committed as a submodule pointer with no
`.gitmodules` entry, so `git submodule update --init` could not fetch it and a fresh clone got an
empty directory. The pointer has been removed; the path is gitignored instead.

It is optional. Stage 5's secondary-structure screen reports itself unavailable, the pipeline
imports fine, and nothing else in stage 5 changes. **Set it up only if you need that screen** —
see [Setting up s4pred](#setting-up-s4pred-optional-gpl-30) below.

**Two dependencies are copyleft and deliberately absent.** `propy3` (GPL-2.0-only) and `s4pred`
(GPL-3.0). Do not add either to `requirements.txt` or to `pyproject.toml`'s core dependencies —
`tests/test_licensing.py` fails if you do, and for good reason: it would make every built image a
combined work under that licence. [docs/LICENSING.md](docs/LICENSING.md) has the detail.

**Model weights are not in the repository** and several predictors cannot run without them.

**TypeScript is pinned below 6 by a peer dependency, not by preference.**
`openapi-typescript` generates `web/src/api/peptide.gen.ts`, and every published version of it —
7.13.0 is the latest — declares `peer typescript@"^5.x"`. Bumping TypeScript past 5 makes
`npm ci` fail with `ERESOLVE`, which breaks the web job *and* the image build, because
`deploy/web.Dockerfile` runs `npm ci` too. A grouped Dependabot PR raising vite, vitest and
TypeScript together fails for this reason alone and cannot be rebased green. Wait for
`openapi-typescript` to support TypeScript 7, or drop the generator first.

**Port 8080 is the job API's default** and is a popular port. Check what is listening before you
blame the code; a proxy pointed at the wrong service fails in confusing ways.

---

## Setting up s4pred (optional, GPL-3.0)

Needed only for stage 5's secondary-structure screen. **Doing this makes your installation a
combined work with GPL-3.0 code**, which constrains what you may redistribute — that is your
decision to make, and `docs/LICENSING.md` explains the consequence.

Upstream is [psipred/s4pred](https://github.com/psipred/s4pred). The code must land at
`src/pipeline/s05_physchem_screening/s4pred/`, because `stage.py` puts that exact directory on
`sys.path` and imports `network.S4PRED` and `utilities.aas2int` from it.

**Download it as a plain directory, not a `git clone`.** That path is already a gitlink in this
repository's index, and a clone puts a nested `.git` there, which makes `git status` report
`M src/pipeline/s05_physchem_screening/s4pred` forever — one careless `git add -A` then commits a
new gitlink SHA. Extracting a tarball leaves the tree clean. (Both behaviours were checked, not
assumed.) If you clone anyway, delete the nested `.git` afterwards.

```bash
cd src/pipeline/s05_physchem_screening && curl -sL https://github.com/psipred/s4pred/archive/refs/heads/main.tar.gz | tar -xz && mv s4pred-main s4pred
```

Then the weights — five ensembled files, ~430 MB unpacked, which the loader expects as
`s4pred/weights/weights_1.pt` … `weights_5.pt`:

```bash
cd src/pipeline/s05_physchem_screening/s4pred && curl -O http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz && tar -xvzf weights.tar.gz
```

That URL is **plain HTTP**, so verify what you got before loading it. Upstream publishes the
archive's MD5 as `e04ad7d10b61551f7e07a86b65bb88dc`:

```bash
md5sum src/pipeline/s05_physchem_screening/s4pred/weights.tar.gz
```

**Confirm it worked** — the flag is the one the code actually branches on, so this is the check
that matters rather than the files being present:

```bash
PYTHONPATH=src .venv/bin/python -c "from pipeline.s05_physchem_screening.stage import S4PRED_AVAILABLE as a, S4PRED_UNAVAILABLE_REASON as r; print(a); print(r or 'the screen will run')"
```

`True` means stage 5 will run the screen and report `available: true`. `False` prints the reason,
which names what failed to import.

Two things to expect afterwards:

- **The backend baseline moves.** The recorded 82/7/0 assumes s4pred is *absent* (`docs/BASELINE.md`).
  A richer environment is not a regression, and `.github/check_baseline.py` says so rather than
  reporting one.
- **Do not add it to `requirements.txt` or `pyproject.toml`'s core dependencies.**
  `tests/test_licensing.py` fails if you do, because that would make every built image a combined
  work under GPL-3.0.

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

**Say what is not known.** Every predictor ships a model card now, but the UI reports
`model_card_status` rather than mere presence, so a card that only records its own absence could
never read as documentation. That is
better than a page implying all fifteen are documented.

---

## Tests for new work

New behaviour needs a test that fails without it. For anything load-bearing — a privacy boundary,
a safety screen, an honesty rule — **break it deliberately and confirm the test fails**. Two
harnesses already do this and are worth reading as examples:

- `services/accounts/tests/break_guards.sh` sabotages the accounts service 17 ways and fails if
  any sabotage goes undetected.
- `tests/test_licensing.py` guards the licence posture; each check was verified by breaking it.

The design-system tests compare `web/src/styles.css` against the project it was copied from. That
project is not in this repository, so they also carry `web/src/test/design-manifest.json` — the
source's rules, committed — and fall back to it when the original is not on disk. Without that,
31 of the 36 tests skipped everywhere except one machine, CI included. Point at a local copy with
`WMXCCS_STYLES=/path/to/styles.css`, and regenerate the manifest with
`node web/scripts/design_manifest.mjs` when the reference design system changes.

Fixtures for result views are **captured from a running API**, never hand-written — see
`web/src/test/fixtures/README.md`. A hand-written fixture encodes what its author believed the API
returns, so the test passes while the app misrenders the real thing. Capturing is what revealed
that `stage` reads `pending` on a succeeded run.

---

## Opening a pull request

You do not need write access. Fork, branch, push to your fork, open a PR.

```bash
gh repo fork Wellmatix-Bio/peptide_discovery_platform --clone
```

```bash
git checkout -b short-description-of-the-change
```

Branch from `main` and keep one change per PR. If you find a second thing worth fixing, say so in
the PR and let somebody decide whether it belongs in the same change — see **Say what you found**
below.

### What has to pass

Three CI jobs are **required**; a PR cannot merge until all three are green:

| Job | What it does |
|---|---|
| `python` | The backend suite against its recorded baseline, plus the accounts suite, plus a check that the OpenAPI snapshot matches `api.py` |
| `web` | Regenerates the TypeScript types and fails on any diff, then typecheck, tests and build |
| `images` | Builds all three container images |

Run them locally first; all three are in [Tests](#tests) above. Two traps worth knowing:

- **`npm run typecheck` can pass locally while CI fails.** `tsc -b` reuses an incremental cache
  and will not revisit a regenerated file. Use `npx tsc -b --noEmit --force` to reproduce CI.
- **If you change `api.py` — even a docstring — regenerate the snapshot.** FastAPI puts docstrings
  in the schema, so a comment-only edit changes the contract file. `python web/scripts/openapi.py
  peptide` then `cd web && npm run api:types`. This has broken `main` twice.

### What review expects

- **One approval**, and conversations resolved before merge.
- **Code Owners review** is required for some paths. `LICENSE`, `NOTICE`, `docs/LICENSING.md` and
  `tests/test_licensing.py` are called out in `.github/CODEOWNERS` because a change there can
  quietly alter what the project may be distributed under.
- **Merge commits only.** Squash and rebase are disabled, so your individual commits survive —
  write them accordingly.
- **Your branch must be up to date** with `main` before merging. GitHub offers an "Update branch"
  button.

### Writing the commit and the PR

Explain **what changed, why, and how you verified it**. "How you verified it" is not a formality
here: for anything load-bearing, the expectation is that you broke it deliberately and watched a
test fail — see [Tests for new work](#tests-for-new-work).

**Say what you found and are not fixing.** A PR that mentions an unrelated problem it noticed is
worth more than one that silently leaves it. If a test you wrote surprised you, or passed when it
should not have, that belongs in the message too.

### First time here?

Issues labelled **`good first issue`** and **`help wanted`** are the ones most likely to be
self-contained. If something in this file is wrong or out of date, that is also a real
contribution — several sections exist because somebody lost an afternoon to the thing they
describe.

---

## Reporting something

- **Security** — do not open a public issue. See [SECURITY.md](SECURITY.md).
- **A bug** — what you ran, what happened, what you expected. The run's `vertex_state` and
  `status` both, if it involves a job.
- **A question** — GitHub Discussions.
