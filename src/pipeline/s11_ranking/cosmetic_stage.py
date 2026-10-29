# Stage 11 (cosmetics): weighted final_score from brief-scaled component weights, plus Pareto fronts over the components.
from __future__ import annotations

import math
from typing import Any

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext, StageError
from pipeline.s05_physchem_screening.cosmetic_physchem import has_lipid
from pipeline.s11_ranking.pareto import pareto_fronts
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage11CosmeticParams

logger = get_logger(__name__)

COMPONENTS = ("primary_function", "secondary_functions", "safety", "delivery_fit", "stability", "developability")


def percentile_ranks(values: list[float]) -> list[float]:
    """Best = 1, worst = 0, ties share the mean rank; fewer than two values give 0.5."""
    n = len(values)
    if n < 2:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 / (n - 1)
        i = j + 1
    return ranks


def component_weights(params: Stage11CosmeticParams, n_secondary: int, depth: str, level: str, ph: float) -> dict[str, float]:
    """Base points scaled by the brief, normalized to sum to 1."""
    low, high = params.ph_comfort_range
    points = {
        "primary_function": params.points_primary,
        "secondary_functions": params.points_secondary_each * n_secondary,
        "safety": params.points_safety * (params.safety_strict_multiplier if level == "strict" else 1.0),
        "delivery_fit": params.points_delivery * params.delivery_depth_multiplier[depth],
        "stability": params.points_stability * params.stability_depth_multiplier[depth],
        "developability": params.points_developability
        * (params.developability_ph_multiplier if not low <= ph <= high else 1.0),
    }
    total = math.fsum(points.values())
    if total <= 0:
        raise StageError("s11_cosmetic_ranking", "every component has zero points; check the stage params.")
    return {name: value / total for name, value in points.items()}


class CosmeticStage11(CandidateStage):
    """Opt-in; replaces the wound-healing ranking for a cosmetic run. Scores and orders, never filters.

    `final_score` is 0-100. `pareto_rank` layers candidates into fronts over the weighted components' scores and is
    separate from the order, which is the `final_score` position.
    """

    name = "s11_cosmetic_ranking"
    requires = {"sequence"}
    produces = {"ranking"}

    def run(self, candidates: list[Candidate], config: StageConfig, ctx: RunContext) -> list[Candidate]:
        params = Stage11CosmeticParams.model_validate(config.params)
        brief = ctx.brief
        depth = params.target_depth or getattr(brief, "target_depth", None)
        level = params.safety_level or getattr(brief, "safety_level", None)
        ph = params.product_ph or getattr(getattr(brief, "product", None), "ph", None)
        missing = [n for n, v in (("target_depth", depth), ("safety_level", level), ("product_ph", ph)) if v is None]
        if missing:
            raise StageError(self.name, f"set {missing} in the stage params or the cosmetic brief.")
        if any(not c.sequence for c in candidates):
            raise StageError(self.name, "received a candidate with no sequence")
        if not candidates:
            return []

        verdict = next(
            (c.predictions["functional_verdict"] for c in candidates if "functional_verdict" in c.predictions), {}
        )
        brief_functions = getattr(brief, "functions", None)
        primary = verdict.get("primary_function") or getattr(brief_functions, "primary", None)
        secondary = verdict.get("secondary_functions")
        if secondary is None:
            secondary = [f for f in getattr(brief_functions, "secondary", None) or [] if f != primary]

        weights = component_weights(params, len(secondary), depth, level, ph)
        primary_ranks = self._primary_ranks(candidates, primary)
        records = [
            self._score(c, rank, secondary, weights, params, level) for c, rank in zip(candidates, primary_ranks)
        ]

        weighted = [n for n in COMPONENTS if weights[n] > 0]
        fronts = pareto_fronts([[r["components"][n]["score"] for n in weighted] for r in records])
        for record, front in zip(records, fronts):
            record["pareto_rank"] = front
        ranked = sorted(records, key=lambda r: (-r["final_score"], -r["coverage"], r["candidate_id"]))
        for i, record in enumerate(ranked):
            record["final_score_rank"] = i + 1

        configuration = {
            **params.model_dump(mode="json"),
            "target_depth": depth,
            "safety_level": level,
            "product_ph": ph,
            "primary_function": primary,
            "secondary_functions": secondary,
            "weights": weights,
        }
        by_id = {c.id: c for c in candidates}
        for record in ranked:
            by_id[record["candidate_id"]].predictions["ranking"] = {**record, "configuration": configuration}
        logger.info("s11_cosmetic.done", extra={"n": len(ranked)})
        return [by_id[r["candidate_id"]] for r in ranked]

    @staticmethod
    def _primary_ranks(candidates: list[Candidate], primary: str | None) -> list[float | None]:
        """Percentile rank of the primary score among the candidates that have one; None where it is missing."""
        scores = [(c.predictions.get("functional_scores") or {}).get(primary, {}).get("score") for c in candidates]
        have = [i for i, s in enumerate(scores) if s is not None]
        ranks: list[float | None] = [None] * len(candidates)
        for i, rank in zip(have, percentile_ranks([scores[i] for i in have])):
            ranks[i] = rank
        return ranks

    def _score(self, candidate, primary_rank, secondary, weights, params, level) -> dict[str, Any]:
        p = candidate.predictions
        components: dict[str, dict[str, Any]] = {}

        def put(name: str, score: float | None, detail: dict[str, Any]) -> None:
            screened = score is not None
            components[name] = {
                "score": score if screened else params.not_screened_score,
                "screened": screened,
                "weight": weights[name],
                "detail": detail,
            }

        put("primary_function", primary_rank, {"basis": "percentile_rank_among_survivors"})

        scored = p.get("functional_scores") or {}
        if secondary and all(f in scored for f in secondary):
            passed = sum(1 for f in secondary if scored[f]["pass"])
            put("secondary_functions", passed / len(secondary), {"passed": passed, "of": len(secondary)})
        else:
            put("secondary_functions", None, {"of": len(secondary)})

        irritation = p.get("irritation") or {}
        cutoffs = irritation.get("cutoffs")
        if irritation.get("status") == "ok" and cutoffs:
            limit = cutoffs["medium_min"] if level == "strict" else cutoffs["high_min"]
            score = irritation["score"]
            margin = (0.0 if score > 0 else 1.0) if limit <= 0 else 1 - min(max(score / limit, 0.0), 1.0)
            lipid = has_lipid(p.get("modifications"))
            put(
                "safety",
                margin * (params.lipid_safety_factor if lipid else 1.0),
                {"irritation_score": score, "t_reject": limit, "lipid_tail": lipid},
            )
        else:
            put("safety", None, {"reason": "irritation_not_screened"})
        # A predicted sensitizer only reaches this stage when Stage 8 was told not to reject it: penalize instead.
        if (p.get("skin_sensitization") or {}).get("sensitizer"):
            safety = components["safety"]
            safety["score"] *= params.skin_sensitizer_safety_factor
            safety["detail"]["skin_sensitizer_factor"] = params.skin_sensitizer_safety_factor

        mw = p.get("molecular_weight")
        if mw is None:
            put("delivery_fit", None, {})
        else:
            fraction = min(max((mw - params.delivery_mw_full) / params.delivery_mw_span, 0.0), 1.0)
            put("delivery_fit", 1 - params.delivery_max_penalty * fraction, {"molecular_weight": mw})

        stability = (p.get("cleavage_stability") or {}).get("score")
        put("stability", stability, {"reason": None if stability is not None else "cleavage_not_screened"})

        synthesis_flags = (p.get("synthesis_feasibility") or {}).get("flags") or {}
        total_flags = p.get("physchem_flag_count", 0) + len(synthesis_flags)
        put(
            "developability",
            1 - params.developability_flag_penalty * min(total_flags, params.developability_flag_cap),
            {"physchem_flags": p.get("physchem_flag_count", 0), "synthesis_flags": sorted(synthesis_flags)},
        )

        for component in components.values():
            component["contribution"] = component["weight"] * component["score"]
        return {
            "candidate_id": candidate.id,
            "sequence": candidate.sequence,
            "final_score": 100 * math.fsum(c["contribution"] for c in components.values()),
            "final_score_rank": None,
            "pareto_rank": None,
            "coverage": math.fsum(c["weight"] for c in components.values() if c["screened"]),
            "not_screened": [n for n, c in components.items() if not c["screened"] and c["weight"] > 0],
            "components": components,
        }
