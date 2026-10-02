"""The deployment's load-bearing properties, read from deploy/ itself.

The job API has no authentication. It is safe ONLY because nothing outside the compose network
can reach it, so "no host port" is checked here rather than trusted to a comment.

DEPLOY/ IS BUILT IN PHASE 4. Until it exists every test here skips, with the reason below --
never silently. Phase 4 must see these turn green; a skip in its final run means the compose
file is missing, not that the checks passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
DEPLOY = REPO / "deploy"
COMPOSE_FILE = DEPLOY / "compose.yaml"

pytestmark = pytest.mark.skipif(
    not COMPOSE_FILE.exists(),
    reason=f"{COMPOSE_FILE} does not exist yet; it is built in phase 4 (see docs/DECISIONS.md)",
)

COMPOSE = (
    yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8")) if COMPOSE_FILE.exists() else {}
)
SERVICES = COMPOSE.get("services", {})

#: Containers that must never publish a host port. The job API is the whole reason this file
#: exists: it has no auth of its own, so a published port bypasses the accounts service entirely.
PRIVATE = ("api", "accounts")


def test_the_expected_services_exist():
    assert set(SERVICES) == {"api", "accounts", "web"}


def test_only_web_publishes_a_port():
    published = {name for name, spec in SERVICES.items() if spec.get("ports")}
    assert published == {"web"}, f"services publishing a host port: {published}"
    for name in PRIVATE:
        assert "network_mode" not in SERVICES[name], f"{name} could reach the host network"


def test_web_binds_loopback_unless_told_otherwise():
    (mapping,) = SERVICES["web"]["ports"]
    assert mapping.startswith("${WEB_BIND:-127.0.0.1}:")


def test_the_signing_key_is_required_not_defaulted():
    key = SERVICES["accounts"]["environment"]["ACCOUNTS_SECRET_KEY"]
    assert key.startswith("${ACCOUNTS_SECRET_KEY:?"), "compose must refuse to start without a key"


def test_accounts_runs_two_workers_on_shared_storage():
    env = SERVICES["accounts"]["environment"]
    assert env["WEB_CONCURRENCY"] == "2"
    (volume,) = SERVICES["accounts"]["volumes"]
    source, target = volume.split(":")
    assert target == "/data"
    # A NAMED volume (declared at the top level), not a bind mount into the host.
    assert "/" not in source and source in COMPOSE["volumes"]


def test_the_proxy_points_at_the_in_network_upstream():
    env = SERVICES["accounts"]["environment"]
    assert env["PEPTIDE_UPSTREAM"] == "http://api:8080"


def test_the_api_defaults_to_the_images_own_non_root_uid():
    """API_UID exists so an off-GCP deployment can read a bind-mounted key whose owner it cannot
    change. It must DEFAULT to the image's own uid, so a GCP deployment that sets nothing is
    unaffected -- and it must never default to 0."""
    user = str(SERVICES["api"]["user"])
    assert "API_UID" in user, "the api user should be configurable for off-GCP credentials"
    assert ":-10001}" in user, f"the api user must default to uid 10001, not {user!r}"
    assert ":-0}" not in user


def test_accounts_and_apis_run_as_non_root():
    for dockerfile in ("accounts.Dockerfile", "api.Dockerfile"):
        text = (DEPLOY / dockerfile).read_text(encoding="utf-8")
        assert "USER app" in text, f"{dockerfile} runs as root"


def test_the_api_container_is_given_the_settings_it_forwards_to_the_worker():
    """api.py's WORKER_ENV_KEYS is read from the process environment, so the API container must
    actually carry VERTEX_MODEL_STORE. If it does not, the worker falls back to
    model_sync.py's placeholder gs://TODO-bucket/model_store and fails during weight sync,
    minutes into a GPU run rather than at submit time. See docs/BASELINE.md."""
    env = SERVICES["api"]["environment"]
    assert "VERTEX_MODEL_STORE" in env
    for required in ("VERTEX_CLOUD_PROJECT", "VERTEX_ARTIFACTS_DIR", "WORKER_IMAGE_URI",
                     "WORKER_SERVICE_ACCOUNT", "MACHINE_TYPE", "ACCELERATOR_TYPE",
                     "ACCELERATOR_COUNT"):
        assert required in env, f"the API returns 503 without {required}"


def test_nginx_sends_api_and_auth_only_to_the_proxy():
    conf = (DEPLOY / "nginx.conf").read_text(encoding="utf-8")
    assert "proxy_pass http://accounts:8080;" in conf
    assert "api:8080" not in conf, "nginx must not reach the job API except through accounts"


def test_nginx_security_headers_are_not_shadowed_by_a_location_add_header():
    """An add_header inside a location drops every add_header above it, CSP included."""
    text = (DEPLOY / "nginx.conf").read_text(encoding="utf-8")
    conf = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    assert "Content-Security-Policy" in conf
    server_level, _, locations = conf.partition("location")
    assert "add_header" in server_level
    assert "add_header" not in locations


def test_secrets_are_not_committed():
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "deploy/.env" in ignored
    example = (DEPLOY / ".env.example").read_text(encoding="utf-8")
    assert "ACCOUNTS_SECRET_KEY=\n" in example, "the template must not carry a key"


def test_the_client_address_header_is_overwritten_by_nginx_and_trusted_only_behind_it():
    """The per-address rate limits read X-Forwarded-For. That is only sound when nginx OVERWRITES
    it; appending ($proxy_add_x_forwarded_for) would pass through whatever a client invented."""
    conf = "\n".join(line.split("#", 1)[0] for line in (DEPLOY / "nginx.conf").read_text(encoding="utf-8").splitlines())
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in conf
    assert "$proxy_add_x_forwarded_for" not in conf
    assert SERVICES["accounts"]["environment"]["ACCOUNTS_CLIENT_IP_HEADER"] == "x-forwarded-for"
    assert not SERVICES["accounts"].get("ports"), "the header is trusted because accounts is unreachable directly"
