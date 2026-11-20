# Model name, version, checksum.
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_CARD_FILENAME = "model_card.json"
_DIR_NAME_RE = re.compile(r"^(?P<name>.+)_(?P<version>v\d+)$")
_CHECKSUM_KEYS = ("checksum", "sha256")


@dataclass(frozen=True)
class ModelRef:
    name: str
    version: str
    checksum: str | None = None


def _split_dir_name(dirname: str) -> tuple[str, str] | None:
    m = _DIR_NAME_RE.match(dirname)
    return (m["name"], m["version"]) if m else None


class ModelRegistry:
    """Scans `store_path/<predictor>/model_card.json` into ModelRefs; bad or missing cards are skipped."""

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)
        self._refs: dict[tuple[str, str], ModelRef] = {}
        self._scan()

    def _scan(self) -> None:
        try:
            dirs = sorted(p for p in self.store_path.iterdir() if p.is_dir())
        except OSError:
            logger.warning("Model store not readable: %s", self.store_path)
            return
        for d in dirs:
            card_path = d / MODEL_CARD_FILENAME
            if not card_path.is_file():
                continue
            ref = self._load_ref(d.name, card_path)
            if ref is not None:
                self._refs[(ref.name, ref.version)] = ref

    @staticmethod
    def _load_ref(dirname: str, card_path: Path) -> ModelRef | None:
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable model card %s: %s", card_path, exc)
            return None
        if not isinstance(card, dict):
            logger.warning("Skipping model card %s: not a JSON object", card_path)
            return None
        # Card's model_name, when it is a "<name>_v<N>" string, wins over the directory name.
        card_name = card.get("model_name")
        split = _split_dir_name(card_name) if isinstance(card_name, str) else None
        split = split or _split_dir_name(dirname)
        if split is None:
            logger.warning("Skipping model card %s: cannot derive name/version", card_path)
            return None
        checksum = next(
            (card[k] for k in _CHECKSUM_KEYS if isinstance(card.get(k), str) and card[k]), None
        )
        return ModelRef(name=split[0], version=split[1], checksum=checksum)

    def list(self) -> list[ModelRef]:
        return sorted(self._refs.values(), key=lambda r: (r.name, r.version))

    def versions(self, name: str) -> list[ModelRef]:
        return [r for r in self.list() if r.name == name]

    def get(self, name: str, version: str | None = None) -> ModelRef | None:
        """Exact (name, version) lookup, or the highest version of `name` when version is None."""
        if version is not None:
            return self._refs.get((name, version))
        refs = self.versions(name)
        return max(refs, key=lambda r: int(r.version[1:])) if refs else None

    def __contains__(self, ref: object) -> bool:
        return isinstance(ref, ModelRef) and self._refs.get((ref.name, ref.version)) is not None

    def __len__(self) -> int:
        return len(self._refs)
