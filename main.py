"""Entry point: python main.py <config_path> runs the full pipeline for that run config."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runner import PipelineRunner, load_config  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>", file=sys.stderr)
        sys.exit(1)

    config = load_config(sys.argv[1])
    candidates = PipelineRunner(config).run()
    print(f"Run {config.run_id} finished with {len(candidates)} candidates.")


if __name__ == "__main__":
    main()
