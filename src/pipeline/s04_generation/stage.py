# Stage 4: Candidate Generation - reference-guided (A), de novo (B), target-interface (C), multifunctional (D).
from __future__ import annotations

from typing import Callable

from common.gpu import release_stage_models
from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext
from pipeline.s04_generation import routeB
from pipeline.s04_generation.routeA import RouteA
from pipeline.s04_generation.routeB import RouteB
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage4_route import Stage4Route

logger = get_logger(__name__)


def _build_route_a(
    candidates: list[Candidate], config: StageConfig, ctx: RunContext
) -> Stage4Route:
    """One GA run per incoming candidate, all sharing the same GA parameters
    (route_a_ga_params) and hard-reject constraints (route_a_constraint_config)
    — the seed sequence is the only thing that varies per candidate."""
    shared_params = {
        "min_length": config.params.get("min_length", 6),
        "max_length": config.params.get("max_length", 35),
        **config.params.get("route_a_ga_params", {}),
        **config.params.get("route_a_constraint_config", {}),
    }
    return RouteA(
        {
            "seed_candidates": [
                {"seed_sequence": candidate.sequence, **shared_params}
                for candidate in candidates
                if candidate.sequence
            ],
        }
    )


def _build_route_b(
    candidates: list[Candidate], config: StageConfig, ctx: RunContext
) -> Stage4Route:
    return RouteB(
        {
            "tags": config.params.get("tags", ["<AMP>"]),
            "n_peptides": config.params.get("n_peptides", 100),
            "min_length": config.params.get("min_length", 6),
            "max_length": config.params.get("max_length", 35),
            "max_new_tokens": config.params.get("max_new_tokens", 120),
            "batch_size": config.params.get("batch_size", 16),
            "max_attempts": config.params.get("max_attempts", 20),
            "generator": config.params.get("generator"),
        }
    )


class Stage4(CandidateStage):
    name = "s04_candidate_generation"
    produces = {"sequence"}

    #: route_key -> factory building a configured Stage4Route from this
    #: stage's (candidates, config, ctx). Populated as each route is implemented.
    routes: dict[
        str, Callable[[list[Candidate], StageConfig, RunContext], Stage4Route]
    ] = {
        "Route A": _build_route_a,
        "Route B": _build_route_b,
    }

    def run(
        self,
        candidates: list[Candidate],
        config: StageConfig,
        ctx: RunContext,
    ) -> list[Candidate]:
        generated: list[Candidate] = list(candidates)
        for route_key, build_route in self.routes.items():
            route = build_route(candidates, config, ctx)
            logger.info(
                f"Running Stage 4 route {route_key} with {len(candidates)} candidates..."
            )
            currently_generated = route.run()
            generated.extend(currently_generated)
            logger.info(
                f"Stage 4 {route_key} generated {len(currently_generated)} new candidates."
            )
        return generated

    def release_models(self) -> None:
        release_stage_models(vars(routeB), stage_name=self.name)
