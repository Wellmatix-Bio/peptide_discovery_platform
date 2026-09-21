# Dev-mode GPU memory release between stages.
#
# Every stage module lazy-loads its models into module-level globals and
# never frees them (see s05-s08 stage.py `_get_*_model()` helpers), so on an
# 8GB dev GPU, later stages OOM once earlier stages' models are still
# resident. Production runs on hardware with enough VRAM to keep everything
# loaded and skip the reload cost, so this only runs under DEV_MODE.
from __future__ import annotations

import inspect
from typing import Any

from common.env import DEV_MODE
from common.logging import get_logger
import torch

logger = get_logger(__name__)


def release_stage_models(stage_globals: dict[str, Any], *, stage_name: str) -> None:
    """Clear a stage module's cached `_*_model`/`_*_cache` globals and free
    GPU memory. Only clears plain data (model instances), never functions —
    the `_get_*_model()` loaders live in the same namespace and also match
    the name pattern.

    Call with `globals()` from the stage module after its models are no
    longer needed. No-op unless DEV_MODE is set.
    """
    if not DEV_MODE:
        return

    freed = [
        name
        for name, value in stage_globals.items()
        if name.startswith("_")
        and (name.endswith("_model") or name.endswith("_cache"))
        and value is not None
        and not inspect.isroutine(value)
    ]
    for name in freed:
        stage_globals[name] = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if freed:
        logger.info(
            "gpu.release_stage_models",
            extra={"stage": stage_name, "released": freed},
        )
