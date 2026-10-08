# Maintaining this repository

For whoever has merge rights. [CONTRIBUTING.md](../CONTRIBUTING.md) is for people sending
changes; this is the other side — what arrives on its own, and what to do with it.

---

## The baseline check will be the first thing that confuses you

The backend suite **does not pass, on purpose**: 310 passed / 3 failed / 0 errors, and
`.github/check_baseline.py` fails CI on *any* movement, including more passes.

- **More failures** → a regression. Find it.
- **More passes** → usually good, but `EXPECTED` and [BASELINE.md](BASELINE.md) are now stale and
  must be updated **in the same PR**.

**A count only means something against a stated environment.** The checker asserts its
assumptions before counting — required modules importable, model weights absent, the copyleft
optionals absent — so a dependency gap reports itself rather than looking like a regression. If
it complains about your environment, believe it: that message exists because a CI dependency gap
was once mistaken for a code regression.

## Dependabot

**Version updates are paused** (`open-pull-requests-limit: 0` in `.github/dependabot.yml`).
**Security updates are not** — they come from repository settings and will still open PRs.

When a security PR arrives:

1. **Is the package a `devDependency`?** Then it does not ship. `deploy/web.Dockerfile` copies
   only vite's `dist/` output into nginx. A "critical" in a build-time package is a far smaller
   problem than the label suggests. `npm audit --omit=dev` is the number that matters for users.
2. **Is the fix a major version?** Then it is a migration, not a bump. Treat it as its own piece
   of work, not a merge button.
3. **Does it pull in a known conflict?** `vitest` and `vite` upgrades currently drag in
   TypeScript 7, which `openapi-typescript` does not support at any published version. Those PRs
   cannot be made green; close them with a pointer to `CONTRIBUTING.md` rather than leaving them
   to rot.

To resume version updates, restore each limit to the value noted beside it in `dependabot.yml`
and read the footnote about `ignore` suppressing security updates before adopting the obvious
major-version filter.

## When CI goes red on `main`

It has happened three times, and twice for the same reason. In order of likelihood:

1. **The OpenAPI snapshot is stale.** Someone changed `api.py` — *including a docstring*, since
   FastAPI puts those in the schema — without running `python web/scripts/openapi.py peptide`.
   The `python` job checks this before the baseline, so it is the first step to fail.
2. **The baseline moved.** See above.
3. **A dependency upgrade broke a guard.** Read the failing test before assuming the test is
   wrong. When react-router 7 made `runview-switch` fail, the test was right and the app had a
   real mismatch.

You can push to `main` directly — you are on the ruleset's bypass list. That exists so a red CI
job cannot lock you out. It is not a general-purpose shortcut; using it routinely means the
ruleset only constrains other people.

## You cannot approve your own pull request

The ruleset requires one approving review, and GitHub does not let an author approve their own
PR. With a single maintainer that means **every change you make is blocked on a review that
cannot arrive**. This is not a misconfiguration; it is what requiring review means.

The escape hatch is the checkbox on the merge box — *"Merge without waiting for requirements to
be met (bypass rules)"* — which appears only for actors on the ruleset's bypass list.

**Why the rule is kept at one approval anyway.** The threat it defends against is unreviewed
code from someone else. An outside contributor's PR you *can* approve, because you are not its
author, so the rule works exactly as intended for the case that matters. Dropping required
approvals to zero would remove review for strangers' PRs too, and would quietly disable the Code
Owners rule on `LICENSE`, `NOTICE`, `docs/LICENSING.md` and `tests/test_licensing.py` — that rule
needs at least one required approval to mean anything.

So the cost is one checkbox on your own PRs, and the benefit is real review on everybody else's.

**Each bypass is a decision, not a step.** Before ticking it, it is worth half a second of "would
a reviewer have objected to this?". The box exists so that a red CI job or an unreviewable
emergency cannot trap you. If you find yourself ticking it without reading, that is the signal to
add a second maintainer — not to loosen the rule.

**The proper fix is a second pair of eyes.** `.github/CODEOWNERS` already points at
`@Wellmatix-Bio/maintainers`; once that team has another member you can review each other's
changes and the bypass stops being part of the routine. Until then, note in the PR that it was
self-merged, so the history says so.

Related: the bypass also lets you push to `main` directly. Same reasoning applies — it exists so
a broken `main` can be fixed, not as a shortcut past the process.

## Reviewing a change

Beyond the obvious, three things specific to this project:

- **Did a check that did not run get reported as a pass?** This is the rule the codebase bends
  hardest to keep, and it has been broken five times across two stages. Look for
  `score is not None and <threshold>` with an `else "pass"`.
- **Does a new test actually fail without the change?** Ask. Several tests here passed against
  their own sabotage until someone checked — including one that matched the wrong DOM field and
  one whose fake upstream encoded the same misunderstanding as the bug.
- **Is a number quoted or measured?** Counts in documents go stale within days. `scripts/
  check_briefs.py` and `check_baseline.py` exist because figures that were written down turned
  out to be wrong.

## Security alerts

Triage at **Security and quality → Dependabot / Code scanning / Secret scanning**.

- A **secret scanning** alert is urgent: rotate first, then decide about history. Treat anything
  ever committed as compromised regardless of what you do next.
- **CodeQL** findings arrive from the default setup. It is deliberately **not** a required status
  check — add it only once you have watched it behave across several PRs, or a noisy rule blocks
  every merge.
- `HISTORY_SCAN.md` records the pre-publication scan and the one finding it produced. Re-run its
  commands before any future history rewrite.

## Cutting a release

1. Move `CHANGELOG.md`'s `[Unreleased]` section under a version heading, following Keep a
   Changelog.
2. Tag `vX.Y.Z` and push the tag.
3. Generate release notes, then **edit them** — generated notes are a commit list, not a
   description of what changed for a user. **Breaking changes first.**
4. Say what is still not true of the project. `README.md`'s "Known issues" is the model.

## Things a new maintainer should read first

| | |
|---|---|
| [BASELINE.md](BASELINE.md) | Every known defect, what was fixed, and what was decided rather than fixed |
| [REVIEW_LOG.md](REVIEW_LOG.md) | Review queries and, more usefully, why each got past the tests |
| [LICENSING.md](LICENSING.md) | Why two dependencies are optional, and the outstanding legal review |
| [PATHWAY_THRESHOLD.md](PATHWAY_THRESHOLD.md) | The one open modelling decision, written for whoever owns the model |
| [REPO_CONFIGURATION.md](REPO_CONFIGURATION.md) | Every GitHub setting and why it is what it is |
