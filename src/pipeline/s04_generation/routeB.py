# Route B: de novo generation via a tag-conditioned ProtGPT2 + LoRA adapter (generation only; training lives in Generation Stage/route_B.ipynb).

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from common.logging import get_logger
from pipeline.base import RunContext
from pipeline.s04_generation.routeA import (
    ConstraintConfig,
    physicochemical_filter,
)
from src.schemas.stage4_route import Stage4Route
from schemas.candidate import Candidate
from schemas.run_config import StageConfig

MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"
if str(MODEL_STORE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_STORE_DIR))

from model_store.routeb_protgpt2_lora_v1 import ProtGPT2Generator

logger = get_logger(__name__)

_generator_cache: ProtGPT2Generator | None = None


def _real_generator() -> ProtGPT2Generator:
    global _generator_cache
    if _generator_cache is None:
        _generator_cache = ProtGPT2Generator()
    return _generator_cache


def _make_variant_id(sequence: str, tags: list[str]) -> str:
    return hashlib.sha1(f"{','.join(tags)}:{sequence}".encode("utf-8")).hexdigest()[:16]


class RouteB(Stage4Route):
    """De novo candidate generation: tag-conditioned ProtGPT2 sampling (Stage 4, route B)."""

    def run(
        self,
    ) -> list[Candidate]:
        tags = self.config.get("tags", ["<AMP>"])
        n_peptides = self.config.get("n_peptides", 100)
        min_length = self.config.get("min_length", 6)
        max_length = self.config.get("max_length", 35)
        constraint_config = ConstraintConfig(min_length=min_length, max_length=max_length)

        logger.info(
            "routeb.generate.start",
            extra={"tags": tags, "n_peptides": n_peptides},
        )
        generator = self.config.get("generator") or _real_generator()
        sequences = generator.generate(
            n_peptides=n_peptides,
            tags=tags,
            min_length=min_length,
            max_length=max_length,
            max_new_tokens=self.config.get("max_new_tokens", 120),
            batch_size=self.config.get("batch_size", 16),
            max_attempts=self.config.get("max_attempts", 20),
        )
        if len(sequences) < n_peptides:
            logger.warning(
                "routeb.generate.underfilled",
                extra={"requested": n_peptides, "collected": len(sequences)},
            )

        candidates_out = []
        for sequence in sequences:
            filter_result = physicochemical_filter(sequence, constraint_config)
            candidates_out.append(
                Candidate(
                    id=_make_variant_id(sequence, tags),
                    sequence=sequence,
                    predictions={
                        "route": "B",
                        "generation_tags": tags,
                        "physicochemical_passed": filter_result.passed,
                        "physicochemical_fail_reasons": filter_result.fail_reasons,
                        **filter_result.computed_attributes,
                    },
                )
            )

        n_passed = sum(
            1 for c in candidates_out if c.predictions["physicochemical_passed"]
        )
        logger.info(
            "routeb.generate.done",
            extra={"n_generated": len(candidates_out), "n_passed_filter": n_passed},
        )
        return candidates_out
