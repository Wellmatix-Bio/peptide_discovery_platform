"""Start the local backend with one command, from the platform folder:

    python scripts/dev_stack.py            # job API (DEV_MODE) on 8090 + accounts service on 8081
    python scripts/dev_stack.py --web      # ... and the React dev server on 5173

The job API runs with DEV_MODE=true, so studio jobs run as local processes on local weights; no Vertex, no bucket. The accounts
service gets a signing key generated once into services/accounts/.data/dev_secret.key (git-ignored), so sessions survive a restart.
Ctrl+C stops everything this script started.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS_DIR = ROOT / "services" / "accounts"
JOB_API_PORT, ACCOUNTS_PORT, WEB_PORT = 8090, 8081, 5173


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def dev_key() -> str:
    path = ACCOUNTS_DIR / ".data" / "dev_secret.key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--web", action="store_true", help="also start the React dev server (needs npm ci done in web/)")
    args = parser.parse_args()

    wanted = [("job API", JOB_API_PORT), ("accounts", ACCOUNTS_PORT)] + ([("web", WEB_PORT)] if args.web else [])
    busy = [f"{name} (port {port})" for name, port in wanted if port_in_use(port)]
    if busy:
        print("Already running, so not starting again: " + ", ".join(busy))
        print("Stop the old process first (Ctrl+C in its terminal), or it may be running old code or without DEV_MODE.")
        return 1

    env = {**os.environ, "DEV_MODE": "true"}
    jobs = [
        ("job API", [sys.executable, "-m", "uvicorn", "backend.api_e2e.api:app", "--port", str(JOB_API_PORT), "--app-dir", "src"], ROOT, env),
        (
            "accounts",
            [sys.executable, "-m", "uvicorn", "accounts.app:build", "--factory", "--port", str(ACCOUNTS_PORT)],
            ACCOUNTS_DIR,
            {**env, "ACCOUNTS_SECRET_KEY": dev_key(), "PEPTIDE_UPSTREAM": f"http://127.0.0.1:{JOB_API_PORT}"},
        ),
    ]
    if args.web:
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if not npm:
            print("npm was not found on PATH, so --web cannot start the UI.")
            return 1
        jobs.append(("web", [npm, "run", "dev"], ROOT / "web", env))

    processes = []
    try:
        for name, command, cwd, job_env in jobs:
            print(f"starting {name}: {' '.join(command[:4])} ...")
            processes.append((name, subprocess.Popen(command, cwd=str(cwd), env=job_env)))
        print(f"\nJob API   http://127.0.0.1:{JOB_API_PORT}\nAccounts  http://127.0.0.1:{ACCOUNTS_PORT}")
        if args.web:
            print(f"UI        http://localhost:{WEB_PORT}")
        print("Ctrl+C stops everything.\n")
        while True:
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n{name} exited with code {process.returncode}; stopping the rest.")
                    return process.returncode or 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping ...")
        return 0
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
