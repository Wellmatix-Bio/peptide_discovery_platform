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

---

## 6. Security features

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
Before publishing
  [ ] git log --all -S scan for secrets, project ids, internal names
  [ ] Secrets rotated if anything was ever committed
  [ ] LICENSE present and compatible with every dependency
  [ ] Rights to publish all third-party code, data and weights confirmed
  [ ] README states what it does NOT do

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
