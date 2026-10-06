# Pre-publication history scan

Run 2026-10-06 against all 87 commits on every ref, following section 1 of
[GITHUB_REPOSITORY_SETUP.md](GITHUB_REPOSITORY_SETUP.md). The working tree is not the thing being
checked — git history is, because a force-push does not help once forks, clones, caches and the
GitHub Events API have a copy.

## Result

**One finding, and it is low severity. Everything else is clean.**

| Check | Result |
|---|---|
| Credential-shaped strings (`BEGIN PRIVATE KEY`, `BEGIN RSA PRIVATE KEY`, `"private_key"`, `ya29.`, `AKIA`, `ghp_`) | **none** |
| `.env`, `.pem`, `.key`, `.p12`, service-account JSON ever added | **none** — only `.env.example` and `deploy/.env.example`, both placeholders |
| Real values from the live `.env` (project, artifacts dir, service account, model store, seed file) | **0 commits each** |
| Real GCS bucket names | **none** — every `gs://` in history is `gs://bucket`, `gs://ci`, `gs://example-bucket` or `gs://test` |
| Real project id or project number | **none** — fixtures carry `projects/example-project` only |
| Blobs over 400 KB | **none** |
| Deleted files (`CLAUDE.md`, three `scripts/*.py`, `src/api/api.py`, a lock file) | checked individually; no credentials, no internal hosts |
| Author identities | 5, all `@wellmatix.co` or GitHub noreply addresses |

The fixture-capture script's redaction did its job: `web/scripts/capture_fixtures.py` rewrites the
project, location and bucket to placeholders, and the committed fixtures contain no real
infrastructure identifier.

## The finding: developer machine paths

`.claude/settings.json` was committed in the initial commit and appears in **71 commits**. It
carries a `PreToolUse` hook pointing at `C:/Users/<name>/.local/bin/graphify.EXE`.

It is **the only file in the entire history** containing a Windows or macOS user path. It has been
untracked and `.claude/` is ignored, so it is absent from the current tree — **but untracking does
not remove it from history.**

Two further paths exist in the current tree, both Linux:

| Path | Where | Why it is there |
|---|---|---|
| `/home/ajit/...` | `docs/DECISIONS.md`, `web/src/styles.css`, `web/src/test/design.test.ts`, `web/scripts/design_manifest.mjs` | References to the sibling project the design system was copied from. `design.test.ts` and the manifest script take `WMXCCS_STYLES` instead, so the path is a default, not a requirement |
| `/home/kounen/...` | `src/pipeline/s04_generation/routeC_moot.py` | Hardcoded WSL micromamba and p2rank locations in a module **nothing imports** |

## What it actually exposes

A username, and the fact that one contributor develops on Windows. No credential, no host, no
customer or patient data, nothing that can be used to reach anything.

The practical harm is smaller than the privacy one: **a project-level hook runs for every
contributor using the same tool and fails**, because the binary exists on one machine. That part is
fixed by the untracking; only the historical record remains.

## Recommendation

**Publishing without rewriting history is defensible**, and that is the recommendation. The
exposure is a username in a tool-configuration file, in a repository whose commits already carry
five authors' real names and email addresses — which is normal, expected, and not something anyone
rewrites history over.

**Rewrite only if the named person asks.** It is their name. If they do, the tool is
`git filter-repo --path .claude/settings.json --invert-paths`, and it rewrites **every commit SHA
in the repository** — every open PR must be rebased, every clone re-cloned, and any SHA quoted in
an issue, a document or a changelog stops resolving. Several documents here cite commit SHAs.

Do it **before** the repository is public, or not at all; afterwards it achieves nothing, because
the old objects are already distributed.

## Two things to settle before publishing

Neither is a history problem, and both block a public release for different reasons:

- **A GitHub Pages site is public even when the repository is private** on most plans. Enabling
  Pages publishes before you intend to.
- `src/pipeline/s04_generation/routeC_moot.py` is **104 KB of code nothing imports**, carrying
  another developer's machine paths. Either wire it up or delete it; shipping it invites a
  contributor to spend a day on a module that is not in the pipeline.

## Reproducing this scan

```bash
git log --all -S'BEGIN PRIVATE KEY' --oneline
git log --all --diff-filter=A --name-only --format='' | sort -u \
  | grep -iE '\.env$|credential|\.pem$|\.key$|secret|service.?account.*\.json'
git grep -I -lE 'C:[\\/]Users[\\/][A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+/' $(git rev-list --all)
git grep -I -ohE 'gs://[a-z0-9][a-z0-9._-]+' $(git rev-list --all) | sort -u
```

Re-run before publishing if any commit has landed since the date at the top of this file.
