# The example briefs do not validate against the API

Measured 2026-09-30 against `BriefFields` in `src/schemas/stage_configs.py`, the
schema `POST /api/v1/jobs/create` actually enforces.

## Result

**0 of 32 briefs in `data/briefs/` are valid API requests.** Not one.

Reproduce:

```bash
.venv/bin/python -c "import sys,json,pathlib; sys.path.insert(0,'src'); from schemas.stage_configs import BriefFields; print(sum(1 for f in pathlib.Path('data/briefs').glob('*.json') if not BriefFields.model_validate(json.loads(f.read_text()))), 'valid')"
```

One of the 32 — `TC-20_edge_case_invalid_inputs.json` — is *meant* to be invalid
(`teleportation`, `max_length: -5`). The other 31 are not.

## What is rejected

| Field | Allowed | Distinct values rejected |
|---|---|---|
| `wound_context` | 15 | **35** |
| `desired_functions` | 9 | **20** |
| `pathogens` | 3 | **12** |

Most frequent rejections:

- `desired_functions`: `re_epithelialization` (9 briefs), `antibiofilm` (4),
  `fibroblast_proliferation` (4), `hemostatic` (4), `scar_reduction` (3)
- `pathogens`: `MRSA` (5), `Acinetobacter_baumannii` (2), `Candida_albicans` (2)
- `wound_context`: `contaminated` (3), `moist` (3), `inflamed` (2),
  `high_protease` (2), `bleeding` (2)
- `dosing_interval_hours` is required but null or absent in 3 briefs
- `min_length`/`max_length` must be 6–50; `TC-21` (tetrapeptide, 4) and `TC-31`
  (pentapeptide, 5) fall below it. Both are modelled on published short-peptide
  work, so the bound excludes a real class of candidate.

## Two contradictions inside the schema itself

These are not the test data being sloppy. `src/schemas/stage_configs.py` disagrees
with itself:

1. **`Stage6Thresholds.max_mbic` ships a default threshold for `Candida_albicans`**,
   while `Pathogens` forbids a client from asking for it. The same file both
   configures a pathogen and refuses to accept it as input.
2. **`Stage4Params.tags` accepts the literal `"<ANTIBIOFILM>"`**, and the project
   ships `model_store/mbic_predictor_v1` (biofilm-inhibition potency), yet
   `antibiofilm` is not a permitted `desired_function`. You can ask the generator
   to produce antibiofilm peptides but cannot state antibiofilm as a goal.

The `Pathogens` docstring explains the narrowing — it is scoped to the 3 species
`mic_predictor_v1` supports, the stricter of the two models sharing the field, and
notes MBIC recognises 13. So the narrowing is deliberate; what appears unintended
is that the rest of the file was not narrowed with it.

## Consequence for the web app

"Load an example brief" cannot be built from `data/briefs/` as it stands: every
file would produce a 422. The UI therefore ships its own small set of examples
that are valid by construction, generated from the enums in
`web/openapi/peptide.json`, and the real briefs are not offered until the
vocabularies agree. Whichever way that is resolved, no example is offered to a
user unless it validates.

## Consequence for the project

Worth someone's attention beyond the frontend: the 32 curated cases are the
project's own description of what it is for — diabetic foot ulcers, burns,
biofilm, haemostasis, MRSA. As deployed, the API accepts none of them. Either the
briefs describe an intended scope the request schema has not caught up with, or
the schema is correct and the briefs are aspirational. The frontend cannot tell
which, and has assumed neither.

---

## Re-measured after PR #6 (2026-10-05)

**1 of 33 briefs validates** — `TC-33_denovo_anti_inflammatory_immunomodulatory_shortlist.json`,
added by that PR. It is the first example brief in the repository that is a usable API request.

The other 32 still fail, and the count of allowed `wound_context` values went **15 → 12**: PR #6
removed `clean`, `high_exudate` and `radiation_induced` from the vocabulary, deliberately, with
the reasoning recorded inline in `src/schemas/stage_configs.py` (`high_exudate` affects dressing
choice rather than any model the pipeline runs; `radiation_induced` had the thinnest mappings).

That makes two more briefs invalid than before — `clean` appears in 5 and `high_exudate` in 5 —
but it is not the main cause. **26 of 33 briefs use at least one context the API has never
accepted**, including `contaminated`, `moist`, `inflamed`, `high_protease` and `bleeding`. The
narrowing made a long-standing mismatch slightly worse; it did not create it.

Reproduce the count:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_briefs.py
```

The three example briefs shipped in the web app (`web/src/runs/examples.ts`) **were** affected and
have been corrected; they are validated by the generated types at compile time, which is how the
breakage surfaced.
