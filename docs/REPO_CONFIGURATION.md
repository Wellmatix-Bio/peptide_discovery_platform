# What was configured, and why

A record of the GitHub settings applied to this repository when it went from private to public on
2026-10-07, with the reasoning for each non-obvious choice. **Written to be copied**: the next
repository making the same move can work down it and get the same posture in about an hour.

[GITHUB_REPOSITORY_SETUP.md](GITHUB_REPOSITORY_SETUP.md) is the generic guide. This is what one
real project actually chose, including the places it deviated and why.

---

## The ordering problem, first

Several settings **exist only on public repositories** on the Free plan: branch protection and
rulesets, secret scanning, push protection, code scanning, private vulnerability reporting, and
Pages. So a repository cannot be hardened while private and then opened.

**The protections you most want in place before publishing are the ones publishing switches on.**
No ordering avoids that window. What narrows it:

1. **Do the history scan properly first.** Secret scanning's first run covers all history anyway,
   so it finds what a manual scan would have — only later, and in public. From the moment you
   publish, its value is stopping the *next* accident. See [HISTORY_SCAN.md](HISTORY_SCAN.md).
2. **Apply the rest within minutes of flipping visibility, before the URL is shared.**

The order used here: everything available while private → publish → ruleset → security features →
Pages → discoverability.

---

## Settings → General

| Setting | Value | Why |
|---|---|---|
| Automatically delete head branches | **On** | Merged branches disappear. Without it they accumulate; this repo had eight stale ones before it was turned on |
| Always suggest updating pull request branches | **On** | Surfaces the "Update branch" button instead of letting PRs drift behind |
| Issues | On | |
| Discussions | On | Keeps questions out of Issues, so the issue count means something |
| Wiki, Projects | Off | An empty tab invites confusion |

## Settings → Rules → Rulesets

One ruleset, **`main protection`**, enforcement **Active**, targeting **the default branch** (not
a literal `main`, so it follows a rename).

| Rule | Value | Why |
|---|---|---|
| Bypass list | the maintainer, *Always allow* | **Deliberate.** Without a bypass actor, a red CI job locks you out of fixing it. This project hit exactly that situation with an unmergeable TypeScript upgrade |
| Restrict deletions | On | |
| Block force pushes | On | |
| Require a pull request | On, **1** approval | |
| ↳ Dismiss stale approvals on new commits | On | An approval is of a diff, not of a branch |
| ↳ Require review from Code Owners | On | `.github/CODEOWNERS` routes licence files and `test_licensing.py` specifically |
| ↳ Require conversation resolution | On | No merging over an unanswered review comment |
| ↳ **Allowed merge methods** | **Merge only** | See below |
| Require status checks | On: `python`, `web`, `images` | The three job names in `ci.yml` |
| ↳ Require branches to be up to date | On | Otherwise checks pass against a base that no longer exists |
| Require linear history | **Off** | It contradicts merge commits; enabling both blocks every merge |
| Require signed commits | Off | A real barrier to drive-by contributors. Revisit when there are any |
| Require code scanning / code quality results | **Off** | Requiring a check that nothing reports blocks every merge. Add CodeQL here only after watching it behave for a few PRs |

### Why merge commits and not squash or rebase

- **Squash** collapses a PR into one commit. Several PRs here carry multiple commits that each
  explain a distinct change, and the commit messages are this project's densest documentation.
- **Rebase** rewrites SHAs. This project cited a commit by hash in `docs/BASELINE.md`, and the
  practice of re-dating commits after pushing had already produced two rejected pushes and one
  duplicated commit on `main`. Rewriting more was not the direction to go.

## Settings → Actions → General

| Setting | Value | Why |
|---|---|---|
| Actions permissions | Allow all actions | Only GitHub's own are used. Tightening to "allow <org>, and select others" is better and costs nothing |
| **Fork pull request workflows** | **Require approval for all external contributors** | Without it, anyone's PR runs code on your runners |
| **Workflow permissions** | **Read repository contents and packages** | The highest-value Actions setting. Verified safe here first: `ci.yml` uses no `GITHUB_TOKEN`, no secrets, no pushes, no artifact upload |
| Allow Actions to create and approve PRs | Off | Follows from read-only |
| **Require actions pinned to a full commit SHA** | **Off, for now** | **This would break CI immediately** — the workflow uses `@v7` tags. A tag can be moved; a SHA cannot. The order is: pin the `uses:` lines, merge that, *then* enable. Enabling first blocks the very PR that would fix it |

## Settings → Advanced Security

| Feature | Value | Note |
|---|---|---|
| Private vulnerability reporting | **On** | Without it, people file security bugs as public issues. `SECURITY.md` points here |
| Dependency graph | On | Prerequisite for Dependabot |
| Dependabot alerts | On | |
| Malware alerts | On | |
| **Dependabot security updates** | **On** | Opens PRs for *vulnerable* packages only |
| Grouped security updates | On | One PR per ecosystem instead of per package |
| Secret scanning | **On** | |
| **Push protection** | **On** | Blocks a secret at push time rather than reporting it once public |
| CodeQL | **On, Default setup** | Default, not Advanced: default needs no workflow file in the repo, which matters when a ruleset requires PRs |
| Copilot Autofix | On | Suggests, does not apply |
| **Code quality** | **Off** | Bills Actions minutes and opens an AI-generated workflow PR |
| Prevent direct alert dismissals | Off | Sensible default for a small team; revisit as it grows |

### Dependabot version updates are paused, and that is not the same as security updates

`.github/dependabot.yml` sets `open-pull-requests-limit: 0` for every ecosystem, which disables
**version** updates. **Security updates are the settings page above and are unaffected** — a
vulnerable package still raises a PR.

Why: grouping cut the queue from 14 PRs to 8, and what remained were three major-version
migrations (React 18→19, TypeScript 5→7 with Vite 5→8, a Node base image) that no rebase could
turn green. Majors are migrations, not bumps. Leaving them at the top of a queue trains everyone
to ignore the queue.

## Settings → Pages

**Deploy from a branch**, `main`, `/docs`.

`docs/.nojekyll` is an empty file and is **not optional**: without it Pages runs Jekyll over the
folder, which ignores paths beginning with `_` and can fail the build outright.

**A Pages site is public even when the repository is private**, on every plan below Enterprise.
Enabling Pages publishes the whole `/docs` folder — not only what the landing page links to.

---

## What this does not cover

- **The legal review.** A checklist item in [LICENSING.md](LICENSING.md), not a GitHub setting,
  and the only item here that a tool cannot do for you.
- **A release.** `CHANGELOG.md` is still `[Unreleased]`; nothing is tagged.
- **Organisation-level settings.** Everything above is per-repository. An org owner can set
  defaults for all repositories, which is worth doing if this is the second or third time.
