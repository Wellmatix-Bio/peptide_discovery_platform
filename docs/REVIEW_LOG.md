# Review log

Queries raised in review, what was found, and where the answer landed. One entry per query, kept
because the interesting part of a review is usually not the fix but **why the problem was not
caught**, and that is the part nobody writes down.

Verified against `main` at the commit that merged PR #28.

---

## 2026-10-03 — Review of the frontend/accounts branch

> *"I checked the current peptide repo. The s4pred setup is incomplete for a fresh checkout, which
> is also noted in the decisions log. Could you include the setup instructions with your
> restructuring? I also found a mismatch in saved result-history handling: it expects a list where
> the API returns a count. There's a possible late-response issue when switching directly between
> runs too. Reproduced in isolation, but not yet in the browser."*

**All three were real. Two were worse than described.**

### Q1 — s4pred setup is incomplete for a fresh checkout

**Status: resolved** — `CONTRIBUTING.md`, *Setting up s4pred (optional, GPL-3.0)*.

Correct, and the first answer given was not an answer. Making the import optional stopped a missing
`s4pred` from breaking the pipeline, but the documentation still said only *"obtain it from its
upstream project and place it there yourself"* — no source, no weights, no way to tell whether it
had worked. That restates the problem.

The section now gives the upstream project ([psipred/s4pred](https://github.com/psipred/s4pred),
GPL-3.0), the exact path the loader requires and why, the weights download with upstream's
published MD5, and a command printing the flag the code actually branches on rather than leaving
someone to infer success from files being present.

Two details found by testing rather than reasoning:

- **Download it as a plain directory, not a `git clone`.** That path is a gitlink in this
  repository's index, so a clone leaves a nested `.git` and `git status` reports the path modified
  from then on — where one `git add -A` commits a new gitlink SHA. Reproduced in a scratch
  repository. The instruction would have been wrong if it had been written from reasoning.
- **Setting it up moves the recorded test baseline**, since the baseline assumes `s4pred` is
  absent. Stated so it is not mistaken for a regression.

### Q2 — result-history expects a list where the API returns a count

**Status: resolved** — `services/accounts/accounts/history.py`.

Correct, and **it was not only the summary**. `is_job_results()` required `ranked_candidates` to be
a **list**. The API returns it as an integer count, with the list in `candidates`. The matcher
therefore returned `False` for *every* real results response: no envelope was ever stored,
`has_envelope` stayed permanently false, and a finished run's summary never advanced past
*"submitted, awaiting the worker's first progress write"*. `_results_summary` compounded it by
calling `len()` on the integer, and `observe()` swallows exceptions by design, so it failed
silently.

**Why it was not caught: the fake upstream in `test_proxy_peptide.py` returned the same wrong
shape.** The fake encoded the misunderstanding it existed to catch, so the suite agreed with the
bug. One of the assertions did too.

`services/accounts/tests/test_envelope_shapes.py` (11 tests) therefore runs against **captured**
fixtures — real responses from a running deployment, which cannot agree with a misreading — and
carries a guard so it cannot pass vacuously. Restoring the old matcher fails three of its tests.

Confirmed end-to-end against **live** API responses once credentials were working: the three
envelope matchers are mutually exclusive on real data, and the summary renders from the integer
counts.

### Q3 — late response when switching directly between runs

**Status: resolved** — `web/src/runs/RunView.tsx`, guarded by
`web/src/test/runview-switch.test.tsx` (4 tests).

Correct. `poll()` called `setStatus`/`setResults` unconditionally, checked its `live` flag only
*after* returning, never passed the `AbortSignal` that `jobStatus()` accepts, and did not reset
state on a param change. Switching between runs could land one run's response under another's
heading. Fixed with a `showing` ref plus the signal.

**Worth recording: the first test written for this passed with the guard removed**, because
aborting already covers the unmount case. The reported case is a param change *without* unmount,
which needed its own test. A guard test that passes against the sabotage is not a guard.

---

## What the review changed beyond the three queries

Chasing Q1 and Q2 surfaced defects nobody had asked about:

- **Stage 5 raised `KeyError('flag')` whenever `s4pred` was unavailable** — the default on a fresh
  clone — so the stage failed on every candidate. Four further screens across stages 5 and 8
  recorded a model that never ran as a **pass**. `cleavage_stability` was the worst: a hard
  *reject* threshold that never ran cleared the candidate.
- **Stage 7 dropped pathways the model could not call**, so "could not determine" rendered as "not
  relevant". Now reported as `undetermined_pathways`.

Both are the same mistake in different places, and both had a guard that should have caught them.
`test_licensing.py`'s unavailable-screen test asserts against **source text** for the single line
that had been fixed, so it passed throughout. **A guard written around one instance of a mistake
does not cover the mistake.**

---

## Still open from this round

| Item | Where |
|---|---|
| `pathway_engagement_min_probability` defaults to 0.5, so the activated and inhibited bands meet and a 0.501 coin flip reads as a definite call. A modelling decision, not made here. | `docs/BASELINE.md` |
| Captured web fixtures predate the pathway-predictor v2 API and still carry `engaged_pathways`. Needs a live run; hand-editing is forbidden for the reason Q2 demonstrates. | `web/src/test/fixtures/README.md` |
| `status` still reports `"pending"` for a dead worker. Now pinned by a test that documents the defect. | `docs/BASELINE.md` |
| 7 of 15 predictors ship no model card; the licensing has had no lawyer review. | `docs/LICENSING.md` |
