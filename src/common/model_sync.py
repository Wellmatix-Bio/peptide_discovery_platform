from __future__ import annotations

import os
from pathlib import Path

from common import storage
from common.env import DEV_MODE

VERTEX_MODEL_STORE = os.environ.get(
    "VERTEX_MODEL_STORE", "gs://TODO-bucket/model_store"
)
MODEL_WEIGHTS_DIRNAME = "model_weights"

# In-process dedup only -- two predictors in the same job needing the same
# model shouldn't download it twice. Carries no meaning across processes.
_synced: set[str] = set()


def weights_dir_for(code_dir: Path) -> Path:
    """Local weights directory for the model whose code lives at `code_dir`
    (model_store/<name>/) -- model_store/model_weights/<name>/. Pure path
    computation, no I/O; safe at module import time, e.g.
    `MODEL_DIR = weights_dir_for(Path(__file__).resolve().parent)`."""
    return code_dir.parent / MODEL_WEIGHTS_DIRNAME / code_dir.name


def sync_model_weights(code_dir: Path) -> Path:
    """Download the weights for the model whose code lives at `code_dir`
    from GCS, if not already synced in this process. Returns the local
    weights directory (see weights_dir_for). Call this lazily, from a
    predictor's own _load()/__init__ body on first real use -- never at
    module scope (see module docstring).

    No-op download (still returns the weights dir) under DEV_MODE -- local
    dev keeps weights checked out under model_weights/ directly -- and when
    VERTEX_MODEL_STORE is a local path, not a gs:// URI.
    """
    model_name = code_dir.name
    weights_dir = weights_dir_for(code_dir)

    if DEV_MODE or model_name in _synced or not storage.is_gcs_path(VERTEX_MODEL_STORE):
        _synced.add(model_name)
        return weights_dir

    gcs_prefix = storage.join(VERTEX_MODEL_STORE, MODEL_WEIGHTS_DIRNAME, model_name)
    storage.download_dir(gcs_prefix, weights_dir)
    _synced.add(model_name)
    return weights_dir
