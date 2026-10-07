# Dev-mode GPU memory release between stages, since stages never free their lazy-loaded models; runs only under DEV_MODE.
from __future__ import annotations

import inspect
from typing import Any

from common.env import DEV_MODE
from common.logging import get_logger
import torch

logger = get_logger(__name__)


def release_stage_models(stage_globals: dict[str, Any], *, stage_name: str) -> None:
    """Clear a stage module's cached `_*_model`/`_*_cache` globals (data only, not loader functions) and free GPU memory; call with `globals()`, no-op unless DEV_MODE."""
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
