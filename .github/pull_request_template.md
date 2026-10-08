## What changed, and why

<!-- The why matters more than the what; the diff already shows the what. -->

## How you verified it

<!-- Commands and their results, not "tested locally". If a suite moved, say by how much. -->

## Checklist

- [ ] New behaviour has a test that fails without the change
- [ ] For anything load-bearing — a privacy boundary, a safety screen, an honesty rule — I broke
      the guard deliberately and confirmed the test fails
- [ ] Backend baseline unchanged, or `docs/BASELINE.md` **and** `.github/check_baseline.py`
      updated together in this PR
- [ ] No copyleft dependency added to `requirements.txt` or pyproject's core dependencies
      (`tests/test_licensing.py` enforces this — see `docs/LICENSING.md`)
- [ ] A check that cannot run reports itself as not run, never as a pass
- [ ] If `api.py` changed **at all, including a docstring**: OpenAPI snapshot regenerated
      (`python web/scripts/openapi.py peptide`) **and** types rebuilt
      (`cd web && npm run api:types`). FastAPI puts docstrings in the schema, so a comment-only
      edit changes the contract file — this has broken `main` twice
- [ ] Web typechecked with `npx tsc -b --noEmit --force`, not just `npm run typecheck`
      (`tsc -b` reuses an incremental cache and will not revisit a regenerated file)
- [ ] Docs updated if behaviour or setup changed

## Anything you found but did not fix

<!-- Say it here rather than leaving it for the next person to rediscover. -->
