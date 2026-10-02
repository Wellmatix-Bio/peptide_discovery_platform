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
- [ ] Docs updated if behaviour or setup changed

## Anything you found but did not fix

<!-- Say it here rather than leaving it for the next person to rediscover. -->
