"""Settings, read from the environment once, and the startup checks that fail loudly.

Two checks run before the service accepts a request:

- THE SIGNING KEY. Required from `ACCOUNTS_SECRET_KEY`. If absent, a RANDOM PER-PROCESS key is
  generated and a loud warning logged - never a constant. A hardcoded default lets anyone who has
  read this file mint a token for any account. The cost of the random key is visible on purpose:
  tokens die on restart and are not accepted by a sibling worker.
- THE DATA DIRECTORY IS WRITABLE. A bind-mounted directory is created root:root and this service
  runs as a non-root user, so the first real deploy otherwise fails later with an unhelpful
  "storage unavailable". The check names the uid, the directory's owner and the `chown` to run.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from .ratelimit import Policy

log = logging.getLogger("accounts")

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / ".data"
DEFAULT_UPSTREAMS = {
    "peptide": "http://127.0.0.1:8080",
}
KEY_ENV = "ACCOUNTS_SECRET_KEY"
# Shorter than this is a key somebody typed, not one somebody generated.
MIN_KEY_BYTES = 32


class StartupError(RuntimeError):
    """A configuration the service refuses to start with. The message says how to fix it."""


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    secret_key: bytes
    key_from_environment: bool
    upstreams: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UPSTREAMS))
    token_ttl_seconds: int = 12 * 60 * 60
    captcha_ttl_seconds: int = 10 * 60
    # NOT the pipeline's runtime. The job API answers quickly by design: create submits a
    # Vertex Custom Job and returns 202, and the run itself takes minutes to hours with the
    # client polling for it. What this covers is the API's own slowest call -- a create, which
    # writes config.json to GCS and submits to Vertex, and a results read, which pulls a run's
    # candidates_final.json back out of GCS.
    upstream_timeout_seconds: float = 120.0
    # Job results are kept so the run view can reopen a finished run. A full ranked shortlist
    # is the largest thing this service stores; above this a row is recorded without the body
    # and says so. The run's own artifacts in GCS remain the source of truth either way.
    max_kept_envelope_bytes: int = 4_000_000
    # The header carrying the real client address, trusted ONLY when set. Set it only when every
    # request arrives through a proxy that OVERWRITES it (deploy/nginx.conf does, for
    # X-Forwarded-For). Unset, the socket peer is the client.
    client_ip_header: str | None = None
    rate_policy: Policy = field(default_factory=Policy)

    @property
    def database(self) -> Path:
        return self.data_dir / "accounts.sqlite3"


def signing_key(environ: dict[str, str] | os._Environ = os.environ) -> tuple[bytes, bool]:
    """The key from the environment, or a random per-process one and a loud warning."""
    given = environ.get(KEY_ENV, "")
    if given:
        key = given.encode("utf-8")
        if len(key) < MIN_KEY_BYTES:
            raise StartupError(
                f"{KEY_ENV} is {len(key)} bytes; at least {MIN_KEY_BYTES} are required. Generate"
                " one with:  python -c 'import secrets; print(secrets.token_urlsafe(48))'"
            )
        return key, True
    log.warning(
        "\n" + "!" * 78 + "\n"
        f"  {KEY_ENV} IS NOT SET. Using a RANDOM key for this process only.\n"
        "  Every token dies when this process restarts, and a token issued by one worker\n"
        "  is REFUSED by another. Set it in the environment for any real deployment.\n"
        + "!" * 78
    )
    return secrets.token_bytes(48), False


def check_writable(directory: Path) -> None:
    """Create the directory if it can be, then prove a file can be written in it."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".write-probe-{os.getpid()}"
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as problem:
        uid = os.getuid() if hasattr(os, "getuid") else "unknown"
        gid = os.getgid() if hasattr(os, "getgid") else "unknown"
        try:
            stat = directory.stat()
            owner = f"{stat.st_uid}:{stat.st_gid}"
        except OSError:
            owner = "unknown (the directory could not be read)"
        raise StartupError(
            f"the accounts data directory {directory} is not writable by this process"
            f" (uid {uid}, gid {gid}); it is owned by {owner}. A bind-mounted directory is"
            " created root:root by Docker. Fix it on the host with:\n"
            f"    sudo chown -R {uid}:{gid} <the host path mounted at {directory}>\n"
            f"or use a named volume, which inherits ownership from the image. ({problem})"
        ) from problem


def from_environment(environ: dict[str, str] | os._Environ = os.environ) -> Settings:
    key, from_env = signing_key(environ)
    data_dir = Path(environ.get("ACCOUNTS_DATA_DIR", str(DEFAULT_DATA_DIR)))
    check_writable(data_dir)
    return Settings(
        data_dir=data_dir,
        secret_key=key,
        key_from_environment=from_env,
        upstreams={
            "peptide": environ.get(
                "PEPTIDE_UPSTREAM", DEFAULT_UPSTREAMS["peptide"]
            ).rstrip("/"),
        },
        client_ip_header=(environ.get("ACCOUNTS_CLIENT_IP_HEADER") or "").strip().lower() or None,
    )
