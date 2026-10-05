# Fixtures

**Captured from a running job API, not written by hand.** A hand-written fixture encodes what its
author believed the API returns, so a test built on one passes while the app misrenders the real
thing. This API has already been found to differ from its own documentation more than once
(`docs/BASELINE.md`), which is exactly why this rule exists.

## How these were produced

```bash
.venv/bin/python -m uvicorn backend.api_e2e.api:app --host 127.0.0.1 --port 8090 --app-dir src
```

```bash
.venv/bin/python web/scripts/capture_fixtures.py --api http://127.0.0.1:8090 --project <PROJECT> --location <LOCATION> 156676979574177792 2046288866976989184 9223232270029029376
```

**No job was submitted.** All three runs already existed in the artifacts bucket; capturing them
cost no GPU time. The API must have working Application Default Credentials — a local process, not
the compose container, which has none unless `GOOGLE_CREDENTIALS_FILE` is set.

The capture script rewrites the project, location and bucket to placeholders, so no committed
fixture names real infrastructure. `models.json` and `healthz.json` come from the two provenance
endpoints.

| File | What |
|---|---|
| `results-156676979574177792.json` | A succeeded run, 11 candidates |
| `results-2046117909033719520.json` | A succeeded run, 16 candidates, **the current stage-11 shape** — real `evidence_coverage`, `final_score` and `missing_modules` |
| `results-2046288866976989184.json` | Vertex says `JOB_STATE_SUCCEEDED`; the API says `running` at `s05_physchem_screening`, with no candidates |
| `results-9223232270029029376.json` | A succeeded run, 33 candidates |
| `status-*.json` | The matching status responses |
| `models.json` | 15 predictors, 8 with model cards |
| `healthz.json` | A fully configured API |

## What the real responses revealed

Both of these were found by capturing rather than assuming, and both are now covered by tests.

**`stage` is `"pending"` on every succeeded run.** It does not end up naming the last stage that
ran. So the stage field is only ever meaningful mid-run, and a completed run shows nothing useful
in it — the UI must not present it as "where the run finished".

**`pmbic` returns 13 pathogens, while a brief may only request 3.** Live output includes
*Candida albicans*, *Klebsiella pneumoniae*, *Acinetobacter baumannii* and ten others. This is the
contradiction recorded in `docs/BRIEF_VALIDITY.md` confirmed from real data: the biofilm model
scores species the request schema refuses to accept as input. Note also that these keys are
space-separated (`"Escherichia coli"`), while the brief's vocabulary is underscored
(`Escherichia_coli`) — the same organism is spelled two ways across one API.

## The `derived/` directory

Three UI behaviours cannot be exercised by any run captured so far, because no real run has yet
produced them: a **withheld value**, an **unranked candidate**, and a run Vertex has finished
while the worker wrote no result. Every captured run has all fields populated, every candidate
ranked, and `insufficient_evidence_candidates: 0`.

So `derived/` holds files **built from the captured ones by a documented transformation**
(`derive.py`), never typed from imagination. Each says in its own `_derived_from` field what it
came from and what was changed. They are clearly separated from the captured fixtures so nobody
mistakes one for evidence of what the API does.

If a real run ever produces an unranked candidate or a withheld score, capture it and delete the
corresponding derived file.


## Re-captured 2026-10-05, after the pathway-predictor v2 API change

The committed fixtures had gone stale: they carried `engaged_pathways`, a candidate field the API
**no longer emits**. Re-captured through the current API from the same runs, plus one new one.

**No run in the bucket carries the v2 pathway shape.** All 22 have `engaged_pathways` in their
stored artifacts, including the newest (2 October) and one named `local-test-worker-v2`, which
turns out to be worker v2 rather than pathway v2. So `activated_pathways`, `inhibited_pathways`
and `undetermined_pathways` read `null` in every fixture here — which is **correct** for a run
produced before that predictor shipped, and is what a user reading their own historical run will
see.

Covering the new pathway fields needs a **new pipeline run through a rebuilt worker image**.
Nothing here can substitute for it, and hand-writing one is exactly what this file forbids.

`results-2046117909033719520.json` was added because it does carry the current **ranking** shape.
Without it nothing exercised `evidence_coverage`, `final_score` or `missing_modules` — and the run
view ignored all three, so a rank computed from part of its intended evidence was displayed
exactly like one computed from all of it.

### One capture changed meaning, and it is worth knowing why

`results-2046288866976989184.json` used to be a succeeded run with 11 candidates. Re-capturing it
now returns **`status: running`, `stage: s05_physchem_screening`, no candidates** — while
`vertex_state` is still `JOB_STATE_SUCCEEDED` and `candidates_final.json` in the bucket still
holds all 11.

Its `results.json` was overwritten by a later partial write. **That is worth someone's attention
beyond the fixtures**: a finished run's results can be replaced by a subsequent write to the same
run directory, and the API then reports the run as unfinished forever. It is kept as a fixture
because it is a real instance of the `status` defect in `docs/BASELINE.md`, observed rather than
constructed.
