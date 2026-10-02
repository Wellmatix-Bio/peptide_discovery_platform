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
| `results-2046288866976989184.json` | A succeeded run, 11 candidates |
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
