# Changelog

Notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Web application: submit a run, watch it, read the ranked shortlist, run history, a models and
  health page, and account management.
- Accounts service and authenticating proxy — the only route to the job API, which has no
  authentication of its own.
- Docker Compose deployment, nginx configuration, and CI.
- Read-only provenance endpoints on the job API: `GET /healthz` and `GET /api/v1/models`.
- Apache-2.0 licence, `NOTICE`, and `docs/LICENSING.md`.
- Documentation: architecture, deployment, contributing, security policy, a walkthrough of every
  page with screenshots, and a reusable GitHub repository setup guide.

### Changed

- **`propy3` and `s4pred` are now optional.** Both are copyleft and mutually incompatible; neither
  is required, vendored or installed. The screens they enable report themselves as not run.
- **A screen that did not run is no longer reported as a pass.** Stage 8 records `not_screened` and
  the overall verdict becomes `flag`.
- The job API forwards an explicit list of settings to the worker, read from the process
  environment rather than from a parsed `.env` file.

### Fixed

- The entire `pipeline` package was unimportable when `s4pred` was absent, making 47 tests
  uncollectable. The backend baseline went from 29 passed / 4 failed / 2 collection errors to
  82 / 7 / 0.
- A peptide too short for the aggregation model's features was recorded as having **passed** a
  screen that never ran.
- A 401 from the upstream signed users out, treating another service's refusal as a dead session.
- A run belonging to another account rendered as a run in progress, with a cancel button.
- A second account signing in to the same tab could read the previous account's job id.

### Known issues

See the README. In short: the brief vocabulary contradicts itself, `status` never reflects the
Vertex job state, `s4pred` is an unresolvable submodule pointer, and stages s05–s09 have no unit
tests.
