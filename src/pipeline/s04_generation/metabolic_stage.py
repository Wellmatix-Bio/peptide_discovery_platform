# Stage 4 (metabolism / weight control): reference-guided generation scored on the brief's target receptors.
from __future__ import annotations

from pipeline.base import RunContext
from pipeline.s04_generation.routeA_metabolic import RouteAMetabolic
from pipeline.s04_generation.routeB import RouteB
from pipeline.s04_generation.stage import Stage4
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage4_route import Stage4Route


def _build_route_a_metabolic(
    candidates: list[Candidate], config: StageConfig, ctx: RunContext
) -> Stage4Route:
    """One GA run per incoming candidate; the brief's receptors set the GA fitness and seed screening."""
    receptors = getattr(ctx.brief, "target_receptors", None) or config.params.get(
        "target_receptors"
    )
    shared = {
        "min_length": ctx.brief.min_length,
        "max_length": ctx.brief.max_length,
        **({"target_receptors": list(receptors)} if receptors else {}),
        "ga_params": config.params.get("route_a_ga_params", {}),
    }
    return RouteAMetabolic(
        {
            "seed_candidates": [
                {"seed_sequence": c.sequence} for c in candidates if c.sequence
            ],
            "feature_extractor": ctx.feature_extractor,
            **shared,
        }
    )


def _build_route_b_metabolic(
    candidates: list[Candidate], config: StageConfig, ctx: RunContext
) -> Stage4Route:
    """Route B (ProtGPT2 + LoRA) with no conditioning tags: the metabolic target has no tag token."""
    return RouteB(
        {
            "tags": [],
            "n_peptides": ctx.brief.route_b_n_peptides,
            "min_length": ctx.brief.min_length,
            "max_length": ctx.brief.max_length,
            "max_new_tokens": config.params.get("max_new_tokens", 120),
            "batch_size": config.params.get("batch_size", 16),
            "max_attempts": config.params.get("max_attempts", 20),
            "generator": config.params.get("generator"),
        }
    )


class Stage4Metabolic(Stage4):
    """Stage 4 with the metabolic Route A and an untagged Route B in place of the wound-healing routes; opt-in."""

    name = "s04_metabolic_candidate_generation"
    routes = {
        "Route A (metabolic)": _build_route_a_metabolic,
        "Route B (untagged)": _build_route_b_metabolic,
    }
