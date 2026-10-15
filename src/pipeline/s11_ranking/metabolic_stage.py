# Stage 11 (metabolism / weight control): ranks candidates on receptor agonism, stability, safety and developability.
from __future__ import annotations

import math
from typing import Any

from common.logging import get_logger
from pipeline.base import CandidateStage, RunContext
from pipeline.s06_functional_models.metabolic_stage import geometric_mean, potency_score
from pipeline.s11_ranking.stage import ApplyFlagDeductions, Bound, FlagRule, _read_path
from schemas.candidate import Candidate
from schemas.run_config import StageConfig
from schemas.stage_configs import Stage11MetabolicParams

logger = get_logger(__name__)

MODULES = (
    "receptor_agonism",
    "stability",
    "anti_inflammatory",
    "safety",
    "developability",
    "mechanistic_confidence",
)

# Flag lines come from the metabolic Stage 5 / Stage 8 thresholds; deductions are capped per flag.
MAX_FLAG_DEDUCTION = 0.4
FLAG_RULES: dict[str, dict[str, FlagRule]] = {
    "stability": {
        "oxidation_risk": FlagRule(
            source="oxidation_risk.risk_category", levels={"medium": 0.2, "high": 0.4}
        ),
        "deamidation_risk": FlagRule(
            source="deamidation_risk.risk_category", levels={"medium": 0.2, "high": 0.4}
        ),
        "instability_index": FlagRule(
            source="instability_index", bounds=[Bound(line=40.0, limit=100.0)]
        ),
    },
    "developability": {
        "aggregation_tendency": FlagRule(
            source="aggregation_tendency.score", bounds=[Bound(line=0.470, limit=0.7)]
        ),
        "solubility": FlagRule(
            source="solubility.score", bounds=[Bound(line=0.173, limit=0.0)]
        ),
        "disulfide_complexity": FlagRule(
            source="disulfide_complexity.category", levels={"flag": 0.2, "high": 0.4}
        ),
    },
    "mechanistic_confidence": {
        "secondary_structure_mechanism": FlagRule(
            source="secondary_structure_consistency.mechanism_consistent",
            levels={"False": 0.2},
        ),
        "secondary_structure_confidence": FlagRule(
            source="secondary_structure_consistency.mean_confidence",
            bounds=[Bound(line=0.7, limit=0.5)],
        ),
    },
}


def _mean(values: list[float | None]) -> float | None:
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else None


def _usable(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def half_life_score(minutes: float | None, target_minutes: float) -> float | None:
    """0.5 at the client's target, 1.0 at 10x it and 0.0 at a tenth of it (log scale); longer is better."""
    if not _usable(minutes) or minutes <= 0:
        return None
    return max(0.0, min(1.0, 0.5 + 0.5 * math.log10(minutes / target_minutes)))


def hemolysis_score(phc50: float | None, cutoff: float | None) -> float | None:
    """1 at pHC50 0, 0.5 at the cutoff and 0 at twice it; None when hemolysis is off or not run."""
    if cutoff is None or not _usable(phc50):
        return None
    return max(0.0, min(1.0, 1.0 - phc50 / (2.0 * cutoff)))


class Stage11Metabolic(CandidateStage):
    """Weighted mean of the module scores that are available: F = sum(w_i S_i) / sum(w_i) over scored modules.

    Opt-in: it runs only when named in the run config, normally in place of `s11_ranking`. A candidate whose
    EC50 is unusable for any selected receptor is `insufficient_evidence` and is not ranked; any other
    missing module is left out and the remaining weights renormalize. Scores and orders, never filters.
    """

    name = "s11_metabolic_ranking"
    requires = {"sequence"}
    produces = {"ranking"}

    def run(
        self, candidates: list[Candidate], config: StageConfig, ctx: RunContext
    ) -> list[Candidate]:
        params = Stage11MetabolicParams.model_validate(config.params).model_dump()
        brief_receptors = getattr(ctx.brief, "target_receptors", None) if ctx.brief else None
        receptors = [
            r for r in ("GLP1R", "GIPR", "GCGR") if r in (brief_receptors or params["target_receptors"])
        ]
        if not receptors:
            raise ValueError("Stage 11 (metabolic) needs at least one target receptor")
        # A metabolic brief is the single configuration: it switches hemolysis and sets the half-life target.
        brief = ctx.brief
        if getattr(brief, "hemolysis_screening", True) is False:
            params["hemolysis_phc50_reject_max"] = None
        if getattr(brief, "min_plasma_half_life_minutes", None) is not None:
            params["half_life_target_minutes"] = brief.min_plasma_half_life_minutes
        weights = params["weights"]
        if set(weights) != set(MODULES) or any(w < 0 for w in weights.values()):
            raise ValueError(f"weights must give a non-negative value for exactly {MODULES}")
        deductions = ApplyFlagDeductions(MAX_FLAG_DEDUCTION)

        results = []
        for candidate in candidates:
            if not candidate.sequence:
                raise ValueError(f"Stage 11 candidate {candidate.id} has no sequence")
            results.append(self._score(candidate, receptors, params, deductions))

        ranked = sorted(
            (r for r in results if r["status"] == "ranked"),
            key=lambda r: (
                -r["final_score"],
                -r["evidence_coverage"],
                -(r["modules"]["stability"]["score"] or -1.0),
                r["candidate_id"],
            ),
        )
        for i, result in enumerate(ranked):
            result["rank"] = i + 1
        insufficient = [r for r in results if r["status"] != "ranked"]

        by_id = {c.id: c for c in candidates}
        configuration = {**params, "target_receptors": receptors, "flag_rules": {
            m: {n: rule.model_dump(mode="json") for n, rule in rules.items()}
            for m, rules in FLAG_RULES.items()
        }}
        for result in ranked + insufficient:
            by_id[result["candidate_id"]].predictions["ranking"] = {
                **result,
                "configuration": configuration,
            }
        logger.info(
            "s11m.done",
            extra={"total": len(results), "insufficient_evidence": len(insufficient)},
        )
        return [by_id[r["candidate_id"]] for r in ranked + insufficient]

    def _score(
        self,
        candidate: Candidate,
        receptors: list[str],
        params: dict,
        deductions: ApplyFlagDeductions,
    ) -> dict:
        p = candidate.predictions
        raw: dict[str, float | None] = {}
        detail: dict[str, dict | None] = {m: None for m in MODULES}

        # Receptor agonism: geometric mean of the selected receptors' EC50 scores; every one must be usable.
        ec50 = p.get("ec50_nm") or {}
        floor = params["ec50_floor_nm"]
        ec50_missing = [r for r in receptors if not (_usable(ec50.get(r)) and ec50[r] > 0)]
        if ec50_missing:
            raw["receptor_agonism"] = None
            detail["receptor_agonism"] = {"missing_ec50": ec50_missing}
        else:
            per_receptor = {
                r: potency_score(ec50[r], params["ec50_max_nm"][r], floor) for r in receptors
            }
            raw["receptor_agonism"] = geometric_mean(list(per_receptor.values()))
            detail["receptor_agonism"] = {"receptor_scores": per_receptor}

        # Stability: average of proteolytic stability and plasma half-life, whichever are available.
        cleavage = _read_path(p, "cleavage_stability.score")
        half_life = half_life_score(
            _read_path(p, "plasma_half_life.minutes"), params["half_life_target_minutes"]
        )
        raw["stability"] = _mean([cleavage if _usable(cleavage) else None, half_life])
        detail["stability"] = {"proteolytic": cleavage, "half_life": half_life}

        anti_inflammatory = p.get("anti_inflammatory_probability")
        raw["anti_inflammatory"] = anti_inflammatory if _usable(anti_inflammatory) else None

        # Safety: mean of (1 - cytotoxicity) and the optional hemolysis score.
        cytotoxicity = _read_path(p, "cytotoxicity.score")
        cytotoxicity_score = 1.0 - cytotoxicity if _usable(cytotoxicity) else None
        hemolysis = hemolysis_score(
            _read_path(p, "hemolysis.phc50"), params["hemolysis_phc50_reject_max"]
        )
        raw["safety"] = _mean([cytotoxicity_score, hemolysis])
        detail["safety"] = {"cytotoxicity": cytotoxicity_score, "hemolysis": hemolysis}

        # Developability: synthesis feasibility, with solubility and aggregation applied as deductions.
        synthesis = _read_path(p, "synthesis_feasibility.ml_feasibility_prior.score")
        raw["developability"] = synthesis if _usable(synthesis) else None

        # Mechanistic confidence: receptor binding affinity (pKd) and structural confidence.
        lo, hi = params["pkd_range"]
        pkd = p.get("binding_pkd") or {}
        pkd_scores = [
            max(0.0, min(1.0, (pkd[r] - lo) / (hi - lo))) for r in receptors if _usable(pkd.get(r))
        ]
        structural = _read_path(p, "mechanism.structural_confidence")
        raw["mechanistic_confidence"] = _mean(
            [_mean(pkd_scores), structural if _usable(structural) else None]
        )
        detail["mechanistic_confidence"] = {
            "pkd_score": _mean(pkd_scores),
            "structural_confidence": structural,
        }

        modules: dict[str, dict] = {}
        for name in MODULES:
            value = raw[name]
            rules = FLAG_RULES.get(name, {})
            fired, total = (
                deductions.evaluate(rules, {n: _read_path(p, r.source) for n, r in rules.items()})
                if value is not None
                else ([], 0.0)
            )
            modules[name] = {
                "raw_measurement": value,
                "measurement_detail": detail[name],
                "flags": [f.model_dump(mode="json") for f in fired],
                "total_deduction": total,
                "score": None if value is None else max(0.0, value - total),
                "nominal_weight": params["weights"][name],
            }

        scored = [m for m in MODULES if modules[m]["score"] is not None and params["weights"][m] > 0]
        weight_sum = math.fsum(params["weights"][m] for m in scored)
        for name in MODULES:
            weight = params["weights"][name] / weight_sum if name in scored and weight_sum else 0.0
            modules[name]["weight"] = weight
            modules[name]["contribution"] = (
                weight * modules[name]["score"] if name in scored else None
            )

        weighted = [m for m in MODULES if params["weights"][m] > 0]
        missing = [m for m in weighted if modules[m]["score"] is None]
        status = "ranked" if scored and raw["receptor_agonism"] is not None else "insufficient_evidence"
        return {
            "candidate_id": candidate.id,
            "sequence": candidate.sequence,
            "status": status,
            "final_score": (
                min(1.0, math.fsum(modules[m]["contribution"] for m in scored))
                if status == "ranked"
                else None
            ),
            "rank": None,
            "evidence_coverage": len([m for m in weighted if m not in missing]) / len(weighted)
            if weighted
            else 0.0,
            "incomplete_evidence": bool(missing),
            "modules": modules,
            "missing_modules": missing,
        }
