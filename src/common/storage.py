# Local-filesystem/GCS I/O branch shared by AuditWriter, BoundaryWriter, and
# FeatureExtractor.save/load. Local under DEV_MODE, gs://bucket/prefix otherwise.
from __future__ import annotations

from pathlib import Path

from common.env import DEV_MODE
from tqdm import tqdm

_GCS_PREFIX = "gs://"


def is_gcs_path(path: str | Path) -> bool:
    return str(path).startswith(_GCS_PREFIX)


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    without_prefix = uri[len(_GCS_PREFIX) :]
    bucket, _, blob_path = without_prefix.partition("/")
    return bucket, blob_path


def _gcs_client():
    from google.cloud import storage  # local import: optional dependency, GCS-mode only

    return storage.Client()


def ensure_dir(path: str | Path) -> None:
    """No-op on GCS -- blobs are created implicitly on write."""
    if is_gcs_path(path):
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def write_text(path: str | Path, data: str, *, encoding: str = "utf-8") -> None:
    if is_gcs_path(path):
        bucket_name, blob_path = _split_gcs_uri(str(path))
        blob = _gcs_client().bucket(bucket_name).blob(blob_path)
        blob.upload_from_string(data.encode(encoding))
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding=encoding)


def append_text(path: str | Path, data: str, *, encoding: str = "utf-8") -> None:
    """GCS has no native append; reads then rewrites the whole blob."""
    if is_gcs_path(path):
        bucket_name, blob_path = _split_gcs_uri(str(path))
        blob = _gcs_client().bucket(bucket_name).blob(blob_path)
        existing = blob.download_as_text(encoding=encoding) if blob.exists() else ""
        blob.upload_from_string((existing + data).encode(encoding))
        return
    with open(path, "a", encoding=encoding) as f:
        f.write(data)


def read_text(path: str | Path, *, encoding: str = "utf-8") -> str:
    if is_gcs_path(path):
        bucket_name, blob_path = _split_gcs_uri(str(path))
        blob = _gcs_client().bucket(bucket_name).blob(blob_path)
        return blob.download_as_text(encoding=encoding)
    return Path(path).read_text(encoding=encoding)


def exists(path: str | Path) -> bool:
    if is_gcs_path(path):
        bucket_name, blob_path = _split_gcs_uri(str(path))
        return _gcs_client().bucket(bucket_name).blob(blob_path).exists()
    return Path(path).exists()


def write_bytes(path: str | Path, data: bytes) -> None:
    if is_gcs_path(path):
        bucket_name, blob_path = _split_gcs_uri(str(path))
        blob = _gcs_client().bucket(bucket_name).blob(blob_path)
        blob.upload_from_string(data)
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def read_bytes(path: str | Path) -> bytes:
    if is_gcs_path(path):
        bucket_name, blob_path = _split_gcs_uri(str(path))
        blob = _gcs_client().bucket(bucket_name).blob(blob_path)
        return blob.download_as_bytes()
    return Path(path).read_bytes()


def join(base: str | Path, *parts: str) -> str:
    """Path join that works for both a local Path and a gs:// string."""
    if is_gcs_path(base):
        return "/".join([str(base).rstrip("/"), *parts])
    return str(Path(base, *parts))


def list_blobs(prefix: str) -> list[str]:
    """Every gs:// object under `prefix`, as full gs:// paths. Local-only
    callers have no use for this -- GCS-only."""
    bucket_name, blob_prefix = _split_gcs_uri(prefix)
    blobs = _gcs_client().bucket(bucket_name).list_blobs(prefix=blob_prefix)
    return [f"{_GCS_PREFIX}{bucket_name}/{blob.name}" for blob in blobs]


def download_dir(gcs_prefix: str, local_dir: str | Path) -> None:
    """Download every object under `gcs_prefix` into `local_dir`, preserving
    the path structure below the prefix (e.g. gs://.../weights/model.bin ->
    local_dir/weights/model.bin)."""
    local_dir = Path(local_dir)
    _, blob_prefix = _split_gcs_uri(gcs_prefix)
    for blob_path in tqdm(list_blobs(gcs_prefix), desc=f"downloading {gcs_prefix}"):
        _, full_blob_name = _split_gcs_uri(blob_path)
        relative = full_blob_name[len(blob_prefix) :].lstrip("/")
        if not relative:
            continue
        dest = local_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(read_bytes(blob_path))
