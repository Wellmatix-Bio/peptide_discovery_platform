from __future__ import annotations

import os
from pathlib import Path

from common import storage
from common.env import DEV_MODE

VERTEX_MODEL_STORE = os.environ.get(
    "VERTEX_MODEL_STORE", "gs://TODO-bucket/model_store"
)
MODEL_WEIGHTS_DIRNAME = "model_weights"

# In-process dedup only, so two predictors needing the same model don't download it twice.
_synced: set[str] = set()


def weights_dir_for(code_dir: Path) -> Path:
    """Local weights directory (model_store/model_weights/<name>/) for the model whose code lives at `code_dir`; pure path computation, safe at import time."""
    return code_dir.parent / MODEL_WEIGHTS_DIRNAME / code_dir.name


def sync_model_weights(code_dir: Path) -> Path:
    """Download the model's weights from GCS once per process and return the local weights directory; call lazily, a no-op under DEV_MODE or a local VERTEX_MODEL_STORE."""
    model_name = code_dir.name
    weights_dir = weights_dir_for(code_dir)

    if DEV_MODE or model_name in _synced or not storage.is_gcs_path(VERTEX_MODEL_STORE):
        _synced.add(model_name)
        return weights_dir

    gcs_prefix = storage.join(VERTEX_MODEL_STORE, model_name)
    storage.download_dir(gcs_prefix, weights_dir)
    _synced.add(model_name)
    return weights_dir
