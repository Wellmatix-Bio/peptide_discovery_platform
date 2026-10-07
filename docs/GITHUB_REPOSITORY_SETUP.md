# Configuring a GitHub repository for public release

A checklist for taking a repository public, written to be reused on other projects. Nothing in it
is specific to this one except the worked examples, which are marked.

Work through it in order. The irreversible things come first, because once a repository is public
you cannot un-publish what people have already cloned.

---

## 1. Before you flip the switch — the irreversible checks

These cannot be fixed afterwards. A force-push does not help: forks, clones, caches and the GitHub
Events API keep what was there.

**Scan the whole history, not the working tree.**

```bash
git log --all -S'<secret-or-identifier>' --oneline
```

```bash
git log --all --diff-filter=A --name-only --format='' | sort -u | grep -iE '\.env$|credential|\.pem$|\.key$|secret|service.?account.*\.json'
```

Check for: credentials and tokens, cloud project ids and account numbers, bucket and database
names, internal hostnames, customer or patient data, employee names and emails you have not
cleared, and licence keys.

**Also scan for developer machine paths**, which leak usernames and are easy to miss because they
are not secrets and no scanner flags them:

```bash
git grep -nIE 'C:[\\/]Users[\\/][A-Za-z0-9_.-]+|/home/[a-z][a-z0-9_-]+/|/Users/[A-Za-z0-9_.-]+/'
```

Editor and agent configuration is where these collect — a committed `.vscode/`, `.idea/` or
`.claude/` often carries an absolute path from whichever machine created it. Beyond the username,
a tool hook pointing at a binary that exists on one machine runs, and fails, for everyone else.

If you find something, **rotate the secret first**, then decide whether to rewrite history. Treat
anything that was ever committed as compromised regardless of what you do next.

**Settle the licence.** With no `LICENSE` file, published code is "all rights reserved" and nobody
may legally use it — including people who assume otherwise. Confirm your dependency licences allow
your choice; copyleft dependencies can constrain it.

**Check what you are entitled to publish**: third-party code, model weights, data, and anything
written under a contract that assigns rights elsewhere.

> **Worked example: the scan itself.** This project's run is written up in
> [HISTORY_SCAN.md](HISTORY_SCAN.md) — what was checked, what came back, and the one finding with
> a recommendation either way. Writing it down matters: the next person to ask "has anyone checked
> this?" gets an answer with a date on it instead of a shrug.

> **Worked example.** This project found two copyleft dependencies compiled in — `propy3`
> (GPL-2.0-only) and `s4pred` (GPL-3.0), which are also incompatible with each other. Both were
> made optional before publishing, which is recorded in `docs/LICENSING.md`. The history scan came
> back clean.

---

## 2. Files GitHub looks for

Place at the repository root (or in `.github/`, which GitHub also reads):

| File | Why |
|---|---|
| `README.md` | What this is, how to run it, what it will not do |
| `LICENSE` | Without it, nobody may use the code |
| `NOTICE` | Attribution, if your licence expects one (Apache-2.0 does) |
| `CONTRIBUTING.md` | Linked automatically from the new-issue and new-PR pages |
| `SECURITY.md` | Linked from the Security tab; says how to report privately |
| `CODE_OF_CONDUCT.md` | Expected of a project inviting contributions |
| `CHANGELOG.md` | What changed between releases |
| `.github/CODEOWNERS` | Auto-requests review from the right people |
| `.github/dependabot.yml` | Dependency update PRs |
| `.github/pull_request_template.md` | What a PR must say |
| `.github/ISSUE_TEMPLATE/*.yml` | Structured issue forms |

A README that says what the project **will not** do is worth more than one that only sells it. It
saves the maintainer the issues that are really misunderstandings.

---

## 3. Repository settings

**Settings → General**

- Default branch named and protected (below).
- **Features**: Issues on. Discussions on if you want questions kept out of Issues. Wiki and
  Projects off unless you will use them — an empty tab invites confusion.
- **Pull requests**: pick one merge strategy and disable the others, so history has one shape.
  Enable *Always suggest updating pull request branches* and *Automatically delete head branches*.
- **Archives**: *Include Git LFS objects* only if you use LFS.

**Settings → Moderation**

- Interaction limits are the lever for a brigading incident. Know they exist before you need them.
- Code review limits restrict review to collaborators if drive-by review becomes a problem.

---

## 4. Branch protection

**Settings → Rules → Rulesets** (rulesets supersede the older branch-protection UI).

On the default branch:

- **Require a pull request before merging** — at least 1 approval; dismiss stale approvals on new
  commits; require review from Code Owners.
- **Require status checks to pass** — name every CI job that must be green, and require branches
  to be up to date first.
- **Require conversation resolution** — no merging over an unanswered review comment.
- **Block force pushes** and **restrict deletions**.
- **Require signed commits** if your contributors can manage it; it is a real barrier for drive-by
  contributions, so weigh it.

Do not make the ruleset so strict that you cannot fix your own CI. Keep a documented bypass for
administrators and say when it may be used.

---

## 5. Actions and secrets

**Settings → Actions → General**

- *Allow actions*: prefer "Allow <org>, and select non-<org> actions" and pin third-party actions
  to a full commit SHA, not a tag. A tag can be moved; a SHA cannot.
- **Workflow permissions**: set to **read-only** by default and grant more per workflow. This is the
  single most valuable Actions setting.
- *Require approval for all outside collaborators* on workflows from forks. Without it, a pull
  request can run code with your runners and, depending on your workflows, your secrets.

**Secrets and variables**

- Repository secrets for CI; **environment** secrets with required reviewers for anything that
  deploys.
- A fork PR must never be able to read a deployment secret. Use `pull_request`, not
  `pull_request_target`, unless you understand exactly why you need the latter.

> **Worked example.** This project's CI builds images with a placeholder signing key and needs no
> real secret, which is the easy case. Keep it that way as long as you can.

> **Worked example: version updates are not free.** Grouping cut this project's queue from 14 PRs
> to 8, and what was left were three major-version migrations — React 18→19, TypeScript 5→7 with
> Vite 5→8, and a Node base-image jump — none of which a rebase can turn green. Version updates
> are now paused in `.github/dependabot.yml` (`open-pull-requests-limit: 0`), while **security
> updates and alerts stay on**, because those are repository settings rather than that file.
> Separating the two is the point: an unreviewed upgrade queue is a workload problem, an unseen
> vulnerability is not the same kind of problem at all.

---

## 6. Security features

> ### Some of this cannot be done before you publish
>
> On GitHub Free, several of the settings below exist only on **public** repositories, so a
> private repo cannot be hardened first and then opened. Check your own screens rather than
> trusting this table — plan features move — but expect the shape of it:
>
> | Setting | Private repo, Free plan |
> |---|---|
> | Dependency graph, Dependabot alerts and security updates | available |
> | Actions workflow permissions (§5) | available |
> | General settings: auto-delete branches, merge strategy (§3) | available |
> | **Branch protection / rulesets (§4)** | public only, or a paid plan |
> | **Secret scanning and push protection** | public only, or GitHub Advanced Security |
> | **Code scanning (CodeQL)** | public only, or GHAS |
> | **Private vulnerability reporting** | public only |
> | **Pages** | public only, or a paid plan |
>
> **The uncomfortable part: the protections you most want in place before publishing are the ones
> publishing switches on.** There is no order that avoids the window. Two things narrow it:
>
> 1. **Do §1's history scan first and finish it.** Secret scanning's first run scans all history,
>    so it finds what you would have found anyway — only later, and in public. Its value from the
>    moment you publish is stopping the NEXT accident, not the last one.
> 2. **Apply §4 and the rest of this section immediately after flipping visibility, before the URL
>    is shared anywhere.** Minutes, not days.
>
> So the practical order is: everything available while private → publish → protections → Pages →
> discoverability. Do not announce the repository until the ruleset is on, or the first external
> contributor meets an unprotected default branch.

**Settings → Code security**

| Feature | Notes |
|---|---|
| **Private vulnerability reporting** | **Turn this on.** Without it, your only honest advice is "email us", and people will open a public issue instead. |
| Dependency graph | Prerequisite for the rest |
| Dependabot alerts | Tells you about vulnerable dependencies |
| Dependabot security updates | Opens the PRs automatically |
| Secret scanning | Catches committed credentials |
| **Push protection** | Blocks the commit before it lands. The one that prevents the problem rather than reporting it |
| Code scanning (CodeQL) | Set up with the default configuration first |

Push protection is worth enabling even on a repository you believe is clean. It costs nothing until
the day it saves you.

---

## 7. Issues and Discussions

**Issue templates** as `.github/ISSUE_TEMPLATE/*.yml` forms rather than markdown — forms can make
fields required, which is how you stop receiving "it doesn't work".

Useful set: bug report, feature request, documentation. Add `config.yml` with
`blank_issues_enabled: false` and contact links pointing security reports to your policy and
questions to Discussions.

Ask for what you will actually need to reproduce: version or commit, how it was run, what happened,
what was expected, environment. Keep it short — every required field costs you reports.

**Labels.** Delete the defaults you will not use. A small set that earns its keep: `bug`,
`enhancement`, `documentation`, `good first issue`, `help wanted`, `needs-info`, `wontfix`.
`good first issue` and `help wanted` are surfaced by GitHub's own discovery pages, so they bring
contributors.

**Discussions** categories: Announcements (maintainers post), Q&A (answerable), Ideas, Show and
tell. Keeping questions out of Issues makes your issue count mean something.

---

## 8. Releases

- Tag with semantic versioning, `v`-prefixed: `v1.4.0`.
- Write the release notes; auto-generated notes are a commit list, not a description of what
  changed for a user. Generate them, then edit.
- State **breaking changes** first, and what to do about each.
- Attach build artifacts if users need them; say which checksums to verify.
- Pre-releases get `-rc.1` and the pre-release flag, so package managers skip them.

A `CHANGELOG.md` in Keep a Changelog format gives people the history without clicking through
releases, and gives you the text to paste into them.

---

## 9. What the repository looks like from outside

- **Description, topics and website.** Topics drive GitHub search; a repository with none is found
  by name only.
- **Social preview image** — what appears when the link is shared.
- **Pin** the repositories that matter on the org profile.
- **Badges** for CI status and licence. Keep them few and true: a badge for a check you do not run
  is worse than none.

---

## 10. After going public

- Watch the Security tab for the first week; secret scanning reports historical findings on first
  scan.
- Answer the first few issues quickly. Early responsiveness sets expectations for everyone who
  arrives later.
- Expect the first bug report to be an environment problem your README did not cover — and fix the
  README rather than only answering the issue.

---

## Quick checklist

```
Before publishing  (this is the irreversible half -- see section 1)
  [ ] git log --all -S scan for secrets, project ids, internal names
  [ ] Scan for developer machine paths, which leak usernames and no scanner flags
  [ ] Secrets rotated if anything was ever committed
  [ ] LICENSE present and compatible with every dependency
  [ ] Legal review, if the project needs one, is DONE rather than deferred
  [ ] Rights to publish all third-party code, data and weights confirmed
  [ ] README states what it does NOT do

While still private  (everything here that your plan allows)
  [ ] General: auto-delete head branches, one merge strategy, Features chosen
  [ ] Actions: workflow permissions read-only
  [ ] Dependency graph + Dependabot alerts and security updates

Immediately AFTER publishing, before sharing the URL
  [ ] Ruleset on the default branch, with a documented bypass actor
  [ ] Secret scanning + push protection
  [ ] Private vulnerability reporting
  [ ] Code scanning

Repository
  [ ] README, LICENSE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT
  [ ] Issue forms + config.yml, PR template, CODEOWNERS, dependabot.yml
  [ ] Description, topics, social preview

Protection
  [ ] Ruleset on the default branch: PR required, checks required, force-push blocked
  [ ] Actions workflow permissions read-only
  [ ] Third-party actions pinned to SHAs
  [ ] Fork PRs cannot reach deployment secrets

Security
  [ ] Private vulnerability reporting ON
  [ ] Secret scanning + PUSH PROTECTION ON
  [ ] Dependabot alerts and security updates ON
  [ ] Code scanning configured

Release
  [ ] CHANGELOG.md
  [ ] Semantic version tag, hand-edited notes, breaking changes first
```

---

## 11. A landing page with GitHub Pages

Pages serves **two different kinds of site**, and the distinction decides your URL:

| Kind | Repository | URL |
|---|---|---|
| **Project site** | any repository | `https://<org>.github.io/<repo>/` |
| **Organization/user site** | one repository named exactly `<org>.github.io` | `https://<org>.github.io/` |

**They are not alternatives — you can have both.** An org site is the landing page for the
organisation; each repository can still have its own project site. A repository-specific page does
not require an org site to exist first, which is the usual worry.

### Publishing a project site from `/docs`

**Settings → Pages → Build and deployment → Source: Deploy from a branch**, branch `main`, folder
`/docs`. The site is live within a minute or two.

Two files make that folder behave:

- `docs/index.html` — the page itself.
- `docs/.nojekyll` — **an empty file that matters.** Without it Pages runs Jekyll, which processes
  the folder as a site source: it ignores files and folders beginning with `_`, may rewrite
  Markdown, and will fail the build on syntax it does not like. With it, files are served exactly
  as committed.

Keep links in the page **relative** (`ARCHITECTURE.md`, `screenshots/x.jpg`). They then work both
on github.com, where the repository renders Markdown, and on the published site. An absolute path
beginning with `/` breaks on a project site, because the URL is already one level deep.

> **Worked example.** This project's page is `docs/index.html`, published from `main` → `/docs`,
> with every relative link checked against the folder it will be served from. Markdown documents
> beside it are served raw rather than rendered, so the page links to the ones a reader needs and
> the rest stay on github.com where they render properly.

### If the site needs a build step

Use **Source: GitHub Actions** instead, with a workflow that builds and uploads an artifact. Only
do this when something must be compiled; a static page from a branch has nothing to break, needs
no secrets, and cannot fail a deploy.

### Before you point anyone at it

- **A Pages site is public even when the repository is private** on most plans. Do not publish a
  page for a repository you have not finished scanning (section 1).
- Set the description and social preview image (section 9) — the page's own `<title>` and
  `<meta name="description">` are what a search engine shows.
- A custom domain goes in **Settings → Pages → Custom domain**, which writes a `CNAME` file into
  the publishing folder. Enable **Enforce HTTPS** once the certificate is issued.
