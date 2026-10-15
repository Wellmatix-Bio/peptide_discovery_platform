# Route A (metabolic): screen seeds on receptor binding affinity, build a per-seed variant pool, then run one global GA with elitism and family-capped survival.

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, TypedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.logging import get_logger
from pipeline.s04_generation.routeA import (
    ConstraintConfig,
    REJECTED_FITNESS_FLOOR,
    Stage4Route,
    _is_valid_sequence,
    _make_individual,
    _make_variant_id,
    _setup_deap,
    hard_constrained_mutant,
    hard_constrained_offspring,
    physicochemical_filter_batch,
    tournament_selection,
    validate_sequence,
)
from schemas.candidate import Candidate

logger = get_logger(__name__)

BatchPredictor = Callable[[list[str]], list[float]]

# Floor applied before the log in the geometric mean, so one zero score does not make every log undefined.
SCORE_EPSILON = 1e-6

# Attempts per variant to find a sequence not already in the seed's pool before accepting a duplicate.
MAX_UNIQUE_ATTEMPTS = 5


@dataclass(frozen=True)
class MetabolicGAConfig:
    """GA and seed-screening parameters; receptor_names must each have a predictor."""

    receptor_names: tuple[str, ...]
    n_variants_per_seed: int = 5
    # Thresholds are binding pKd = 9 - log10(Kd [nM]): 7.0 is Kd 100 nM, 8.0 is Kd 10 nM.
    screen_threshold: float = (
        7.0  # seed passes outright at or above this binding pKd geometric mean
    )
    near_match_threshold: float = (
        8.0  # ...or when pKd to any single receptor reaches this
    )
    min_seeds_kept: int = 50  # top-ranked seeds are backfilled up to this count
    n_generations: int = 20
    mutation_rate: float = 0.1
    tournsize: int = 3
    elite_fraction: float = 0.05
    max_family_share: float = (
        0.25  # cap on one seed family's share of the surviving population
    )
    random_seed: int = 42

    def validate(self) -> None:
        if not self.receptor_names:
            raise ValueError("receptor_names must be non-empty.")
        if self.n_variants_per_seed < 1:
            raise ValueError("n_variants_per_seed must be >= 1.")
        if self.n_generations < 0:
            raise ValueError("n_generations must be >= 0.")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if self.tournsize < 2:
            raise ValueError("tournsize must be >= 2.")
        if not 0 <= self.elite_fraction < 1:
            raise ValueError("elite_fraction must be in [0, 1).")
        if not 0 < self.max_family_share <= 1:
            raise ValueError("max_family_share must be in (0, 1].")


@dataclass(frozen=True)
class Evaluation:
    fitness_score: float
    receptor_scores: Dict[str, float]
    client_fitness: float | None = None


class MetabolicGAResult(TypedDict):
    best: dict
    history: pd.DataFrame
    population: List
    seeds_kept: List[str]


def geometric_mean(scores: list[float]) -> float:
    """Geometric mean of positive scores (raw pKd / pEC50 or any positive scale), each clipped to SCORE_EPSILON from below."""
    return math.exp(sum(math.log(max(s, SCORE_EPSILON)) for s in scores) / len(scores))


def _score_raw(
    sequences: list[str],
    receptors: Dict[str, BatchPredictor],
    config: MetabolicGAConfig,
    client_fitness_batch: BatchPredictor | None,
) -> list[Evaluation]:
    """Receptor (and optional client) scores for each sequence in one batch per predictor, with no physicochemical filtering."""
    if not sequences:
        return []
    per_receptor: Dict[str, list[float]] = {}
    for name in config.receptor_names:
        values = receptors[name](sequences)
        if len(values) != len(sequences):
            raise ValueError(
                f"Predictor {name!r} returned {len(values)} values for {len(sequences)} sequences."
            )
        per_receptor[name] = [float(v) for v in values]
    client = (
        [float(v) for v in client_fitness_batch(sequences)]
        if client_fitness_batch is not None
        else None
    )

    evaluations = []
    for i, sequence in enumerate(sequences):
        scores = {name: per_receptor[name][i] for name in config.receptor_names}
        client_score = client[i] if client is not None else None
        factors = list(scores.values()) + (
            [client_score] if client_score is not None else []
        )
        if not all(math.isfinite(s) for s in factors):
            logger.warning(
                "routeA_metabolic.score_not_finite", extra={"sequence": sequence}
            )
            evaluations.append(Evaluation(REJECTED_FITNESS_FLOOR, scores, client_score))
        else:
            evaluations.append(
                Evaluation(geometric_mean(factors), scores, client_score)
            )
    return evaluations


def _evaluate_population(
    sequences: list[str],
    cache: Dict[str, Evaluation],
    receptors: Dict[str, BatchPredictor],
    config: MetabolicGAConfig,
    client_fitness_batch: BatchPredictor | None,
    constraint_config: ConstraintConfig,
) -> None:
    """Evaluate every not-yet-cached sequence: physchem filter first, predictors only on passers."""
    to_score = [seq for seq in dict.fromkeys(sequences) if seq not in cache]
    if not to_score:
        return
    filter_results = physicochemical_filter_batch(to_score, constraint_config)
    passed = [seq for seq, r in zip(to_score, filter_results) if r.passed]
    scored = dict(
        zip(passed, _score_raw(passed, receptors, config, client_fitness_batch))
    )
    rejected = Evaluation(
        REJECTED_FITNESS_FLOOR, {name: 0.0 for name in config.receptor_names}, None
    )
    for seq in to_score:
        cache[seq] = scored.get(seq, rejected)


def screen_seeds(
    seeds: list[str],
    binding_predictors: Dict[str, BatchPredictor],
    config: MetabolicGAConfig,
) -> list[str]:
    """Keep seeds whose binding affinity matches the profile or comes close on any one receptor, backfilling the top-ranked up to min_seeds_kept."""
    unique = list(dict.fromkeys(seeds))
    evaluations = _score_raw(unique, binding_predictors, config, None)

    def best_receptor(ev: Evaluation) -> float:
        return max(ev.receptor_scores.values())

    kept = [
        seq
        for seq, ev in zip(unique, evaluations)
        if ev.fitness_score >= config.screen_threshold
        or best_receptor(ev) >= config.near_match_threshold
    ]
    if len(kept) < config.min_seeds_kept:
        ranked = sorted(
            zip(unique, evaluations),
            key=lambda pair: (pair[1].fitness_score, best_receptor(pair[1])),
            reverse=True,
        )
        for seq, _ in ranked:
            if len(kept) >= config.min_seeds_kept:
                break
            if seq not in kept:
                kept.append(seq)
    logger.info(
        "routeA_metabolic.seeds_screened",
        extra={"n_seeds": len(unique), "n_kept": len(kept)},
    )
    return kept


def _select_survivors(
    parents: List, offspring: List, pop_size: int, n_elite: int, family_cap: int
) -> List:
    """Elites first, then fittest-first fill with a per-family cap and no duplicate sequences; leftover slots are backfilled ignoring the cap."""
    merged = sorted(
        parents + offspring, key=lambda ind: ind.fitness.values[0], reverse=True
    )
    survivors: List = []
    seen: set[str] = set()
    family_counts: Dict[str, int] = {}

    def take(individual) -> None:
        survivors.append(individual)
        seen.add("".join(individual))
        family_counts[individual.family_id] = (
            family_counts.get(individual.family_id, 0) + 1
        )

    for individual in merged:
        if len(survivors) >= n_elite:
            break
        if "".join(individual) not in seen:
            take(individual)
    for individual in merged:
        if len(survivors) >= pop_size:
            break
        if "".join(individual) in seen:
            continue
        if family_counts.get(individual.family_id, 0) < family_cap:
            take(individual)
    for individual in merged:
        if len(survivors) >= pop_size:
            break
        if "".join(individual) not in seen:
            take(individual)
    return survivors


def run_metabolic_ga(
    seed_sequences: list[str],
    min_length: int,
    max_length: int,
    receptors: Dict[str, BatchPredictor],
    binding_predictors: Dict[str, BatchPredictor],
    config: MetabolicGAConfig,
    client_fitness_batch: BatchPredictor | None = None,
    constraint_config: ConstraintConfig | None = None,
) -> MetabolicGAResult:
    config.validate()
    for label, predictors in (
        ("potency", receptors),
        ("binding", binding_predictors),
    ):
        missing = [n for n in config.receptor_names if n not in predictors]
        if missing:
            raise ValueError(f"No {label} predictor supplied for receptors: {missing}")
    if not seed_sequences:
        raise ValueError("seed_sequences must be non-empty.")
    for seq in seed_sequences:
        validate_sequence(seq)
    constraint_config = constraint_config or ConstraintConfig(
        min_length=min_length, max_length=max_length
    )

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    rng = random.Random(config.random_seed)
    _setup_deap()
    eval_cache: Dict[str, Evaluation] = {}
    history_rows: List[Dict[str, object]] = []
    ordinal = 0

    def record_batch(individuals: List, phase: str) -> None:
        sequences = ["".join(ind) for ind in individuals]
        _evaluate_population(
            sequences,
            eval_cache,
            receptors,
            config,
            client_fitness_batch,
            constraint_config,
        )
        for individual, sequence in zip(individuals, sequences):
            ev = eval_cache[sequence]
            individual.fitness.values = (ev.fitness_score,)
            history_rows.append(
                {
                    "variant_id": individual.variant_id,
                    "parent_id": individual.parent_id,
                    "family_id": individual.family_id,
                    "sequence": sequence,
                    "generation": individual.generation,
                    "phase": phase,
                    "fitness_score": ev.fitness_score,
                    **{f"{n}_score": s for n, s in ev.receptor_scores.items()},
                    "client_fitness": ev.client_fitness,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

    def new_individual(sequence: str, parent_id: str, family_id: str, generation: int):
        nonlocal ordinal
        variant_id = _make_variant_id(sequence, generation, ordinal)
        ordinal += 1
        individual = _make_individual(sequence, variant_id, parent_id, generation)
        individual.family_id = family_id
        return individual

    # Phase 1: screen seeds on binding affinity, then a fixed number of independent variants per kept seed.
    kept_seeds = screen_seeds(seed_sequences, binding_predictors, config)
    population: List = []
    for seed_sequence in tqdm(kept_seeds, desc="RouteA-metabolic seeds"):
        seed_id = _make_variant_id(seed_sequence, 0, ordinal)
        ordinal += 1
        pool = {seed_sequence}
        for _ in range(config.n_variants_per_seed):
            sequence = seed_sequence
            for _ in range(MAX_UNIQUE_ATTEMPTS):
                sequence = hard_constrained_mutant(
                    seed_sequence,
                    min_length,
                    max_length,
                    config.mutation_rate,
                    rng,
                    constraint_config,
                )
                if sequence not in pool:
                    break
            pool.add(sequence)
            population.append(new_individual(sequence, seed_id, seed_id, 0))

    pop_size = len(population)
    n_elite = max(1, math.ceil(config.elite_fraction * pop_size))
    n_families = len(kept_seeds)
    family_cap = max(
        math.ceil(config.max_family_share * pop_size), math.ceil(pop_size / n_families)
    )
    record_batch(population, "init")

    # Phase 2: global tournament selection, crossover + mutation, elitist family-capped survival.
    for generation in range(1, config.n_generations + 1):
        selected = tournament_selection(population, pop_size, config.tournsize, rng)
        offspring: List = []
        for i in range(0, pop_size, 2):
            parent_a = selected[i]
            parent_b = selected[(i + 1) % pop_size]
            res = hard_constrained_offspring(
                "".join(parent_a),
                "".join(parent_b),
                min_length,
                max_length,
                config.mutation_rate,
                rng,
                constraint_config,
            )
            if res is None:
                continue
            fitter = max(parent_a, parent_b, key=lambda p: p.fitness.values[0])
            parent_ids = ",".join(sorted([parent_a.variant_id, parent_b.variant_id]))
            for child_seq in res:
                if len(offspring) < pop_size:
                    offspring.append(
                        new_individual(
                            child_seq, parent_ids, fitter.family_id, generation
                        )
                    )
        if not offspring:
            logger.warning(
                "routeA_metabolic.generation_collapsed",
                extra={"generation": generation},
            )
            break
        record_batch(offspring, "evolve")
        population = _select_survivors(
            population, offspring, pop_size, n_elite, family_cap
        )
        fitnesses = [ind.fitness.values[0] for ind in population]
        logger.info(
            "routeA_metabolic.generation",
            extra={
                "generation": generation,
                "best_fitness": round(max(fitnesses), 4),
                "mean_fitness": round(float(np.mean(fitnesses)), 4),
                "n_families": len({ind.family_id for ind in population}),
            },
        )

    history = pd.DataFrame(history_rows)
    best_row = history.loc[history["fitness_score"].idxmax()].to_dict()
    return {
        "best": best_row,
        "history": history,
        "population": population,
        "seeds_kept": kept_seeds,
    }


# Potency receptors the model cannot rank (leave-one-cluster-out R2 < 0), kept out of the default fitness.
POTENCY_UNRELIABLE_RECEPTORS = ("GIPR",)
DEFAULT_TARGET_RECEPTORS = ("GLP1R", "GCGR")


def build_model_predictors(
    feature_extractor,
    receptor_names: tuple[str, ...],
    binding_split: str = "cluster",
) -> tuple[Dict[str, BatchPredictor], Dict[str, BatchPredictor]]:
    """Wire binding_affinity_predictor_v1 (seed screening, pKd) and functional_potency_predictor_v1 (GA fitness, pEC50) as raw-scale batch predictors per receptor."""
    from model_store.binding_affinity_predictor_v1 import BindingAffinityPredictor
    from model_store.binding_affinity_predictor_v1.predictor import (
        RECEPTORS as BINDING_RECEPTORS,
    )
    from model_store.functional_potency_predictor_v1 import FunctionalPotencyPredictor
    from model_store.functional_potency_predictor_v1.predictor import (
        RECEPTORS as POTENCY_RECEPTORS,
    )

    unsupported = [
        n
        for n in receptor_names
        if n not in BINDING_RECEPTORS or n not in POTENCY_RECEPTORS
    ]
    if unsupported:
        raise ValueError(
            f"No model for receptors {unsupported}; supported: {sorted(set(BINDING_RECEPTORS) & set(POTENCY_RECEPTORS))}."
        )

    binding_model = BindingAffinityPredictor(split=binding_split)
    potency_model = FunctionalPotencyPredictor()
    potency_cache: Dict[str, Dict[str, float]] = {}

    def potency(sequences: list[str], receptor: str) -> list[float]:
        # The potency model scores all receptors from one ESM2 pass, so fill the cache once per new sequence.
        missing = [s for s in dict.fromkeys(sequences) if s not in potency_cache]
        if missing:
            all_scores = potency_model.predict_pec50_all_batch(
                missing, feature_extractor
            )
            for i, s in enumerate(missing):
                potency_cache[s] = {r: v[i] for r, v in all_scores.items()}
        return [potency_cache[s][receptor] for s in sequences]

    def binding(sequences: list[str], receptor: str) -> list[float]:
        return binding_model.predict_pkd_batch(sequences, feature_extractor, receptor)

    return (
        {n: (lambda seqs, n=n: potency(seqs, n)) for n in receptor_names},
        {n: (lambda seqs, n=n: binding(seqs, n)) for n in receptor_names},
    )


class RouteAMetabolic(Stage4Route):
    """Reference-guided generation for metabolic targets: seed screening on receptor binding affinity, then global GA optimization on functional potency (Stage 4, route A).

    Scores stay on the models' original scales: seed screening uses binding pKd and the GA fitness is the
    geometric mean of functional pEC50 across the target receptors, so `{receptor}_score` and `fitness_score`
    in a candidate's predictions are pEC50 values, not [0, 1] scores. A client fitness, if given, joins that
    geometric mean on whatever scale it has.

    Config: `feature_extractor` (required unless both predictor dicts are passed), `target_receptors`
    (default GLP1R and GCGR), `binding_split`, and optional `receptor_predictors` / `binding_predictors`
    overrides (name -> batch predictor returning positive raw scores).
    """

    def __init__(self, config: dict[str, Any] = {}):
        super().__init__(config)

    def run(self) -> list[Candidate]:
        cfg = self.config
        min_length = cfg.get("min_length", 6)
        max_length = cfg.get("max_length", 35)
        receptors: Dict[str, BatchPredictor] = cfg.get("receptor_predictors", {})
        binding: Dict[str, BatchPredictor] = cfg.get("binding_predictors", {})
        if not receptors and not binding:
            feature_extractor = cfg.get("feature_extractor")
            if feature_extractor is None:
                raise ValueError(
                    "RouteAMetabolic needs config['feature_extractor'] to run the bundled models, "
                    "or config['receptor_predictors'] and config['binding_predictors'] overrides."
                )
            target_receptors = tuple(
                cfg.get("target_receptors", DEFAULT_TARGET_RECEPTORS)
            )
            receptors, binding = build_model_predictors(
                feature_extractor,
                target_receptors,
                binding_split=cfg.get("binding_split", "cluster"),
            )
        elif not receptors or not binding:
            raise ValueError(
                "Pass both config['receptor_predictors'] (functional potency, used for GA fitness) "
                "and config['binding_predictors'] (binding affinity, used for seed screening), or neither."
            )
        ga_params = cfg.get("ga_params", {})
        ga_config = MetabolicGAConfig(
            receptor_names=tuple(cfg.get("target_receptors", receptors)), **ga_params
        )

        seeds = []
        for candidate in cfg.get("seed_candidates", []):
            sequence = candidate.get("seed_sequence")
            if _is_valid_sequence(sequence, min_length, max_length):
                seeds.append(sequence.strip())
            else:
                logger.info(
                    "routeA_metabolic.invalid_seed_candidate",
                    extra={"candidate": sequence},
                )

        result = run_metabolic_ga(
            seeds,
            min_length,
            max_length,
            receptors,
            binding,
            ga_config,
            client_fitness_batch=cfg.get("client_fitness_batch"),
            constraint_config=ConstraintConfig.from_params(
                {
                    "min_length": min_length,
                    "max_length": max_length,
                    **cfg.get("constraint_config", {}),
                }
            ),
        )
        history_by_variant = result["history"].set_index("variant_id")
        candidates = []
        for individual in result["population"]:
            row = history_by_variant.loc[individual.variant_id]
            candidates.append(
                Candidate(
                    id=individual.variant_id,
                    sequence="".join(individual),
                    predictions={
                        **{
                            f"{n}_score": float(row[f"{n}_score"])
                            for n in ga_config.receptor_names
                        },
                        "fitness_score": float(individual.fitness.values[0]),
                        "generation": int(individual.generation),
                        "family_id": individual.family_id,
                        "route": "A_metabolic",
                    },
                )
            )
        return candidates
