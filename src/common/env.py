# Minimal .env loader — no external dependency for a single flag.
from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

DOTENV_KEYS: list[str] = []


def _load_dotenv(path: Path) -> list[str]:
    if not path.exists():
        return []
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        os.environ.setdefault(key, value.strip())
        keys.append(key)
    return keys


DOTENV_KEYS = _load_dotenv(_ENV_PATH)


def _env_bool(name: str, default: bool = False) -> bool:
    """Parses an environment value as a bool, since "False" and "0" are truthy strings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


DEV_MODE = _env_bool("DEV_MODE")
SEED_CANDIDATES_FILE = os.environ.get("SEED_CANDIDATES_FILE", "seed_candidates.fasta")
