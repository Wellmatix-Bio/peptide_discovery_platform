# Security policy

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability), which opens a disclosure visible only to maintainers.

Please include what you did, what happened, and what an attacker could achieve with it. We will
acknowledge receipt and tell you what we intend to do about it.

## What this project is, in security terms

Worth knowing before you assess it:

- **The job API has no authentication of its own, by design.** It is safe only because it
  publishes no host port and `services/accounts/` is the only route to it. If you can reach the
  job API directly, that is the finding — not the absence of auth on it.
- **The identity layer is demo-scoped.** Email/password accounts, stateless signed tokens, SQLite.
  It is meant to be replaced by a platform identity system, not grown. `services/accounts/README.md`
  states its limits plainly: the captcha is text in an SVG and a parser solves it, signing out
  revokes nothing server-side, and one instance is the only supported topology.
- **Registration is open and there is no per-user quota.** Anyone who can reach a deployment and
  register can start GPU jobs that bill to its cloud project. This was an explicit decision, and
  `docs/DEPLOYMENT.md` says to set a hard billing cap before exposing it. A report that this is
  exploitable in a specific deployment is useful; a report that it is possible is already known.
- **Model weights are not in this repository** and carry their own terms.

## Scope

In scope: the pipeline, the job API, the accounts service and its proxy, the web app, and the
deployment configuration in `deploy/`.

Out of scope: vulnerabilities in third-party dependencies with no exploitable path through this
code (report those upstream), and anything requiring an attacker to already control the host.
