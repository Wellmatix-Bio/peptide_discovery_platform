# Minimal .env loader — no external dependency for a single flag.
from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(_ENV_PATH)

DEV_MODE = os.environ.get("DEV_MODE", False)
