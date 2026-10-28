# Cosmetic Route A: seed-based GA ranked by primary-function activity (1/0), then physicochemical penalty, then diversity.
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rapidfuzz.distance import Levenshtein

from common.logging import get_logger
from pipeline.s04_generation.cosmetic_validity import validate_sequence_and_modifications
from schemas.candidate import Candidate
from schemas.stage4_route import Stage4Route

logger = get_logger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PROPERTIES = ("charge", "gravy", "molecular_weight")
MUTATIONS = ("substitution", "insertion", "deletion")
MAX_ATTEMPTS_PER_SLOT = 25  # bounded retry so a tiny length window cannot hang the GA

#: Batch decision: sequences -> predicted active (True) or not.
PrimaryPredictor = Callable[[list[str]], list[bool]]


def compute_properties(sequence: str, ph: float) -> dict[str, float]:
    analysis = ProteinAnalysis(sequence)
    return {
        "charge": float(analysis.charge_at_pH(ph)),
        "gravy": float(analysis.gravy()),
        "molecular_weight": float(analysis.molecular_weight()),
    }


def build_reference(sequences: list[str], ph: float) -> dict[str, dict[str, Any]]:
    """Per-property 10th/90th percentile band and the scale that normalises the penalty (None when it cannot be)."""
    props = [compute_properties(s, ph) for s in sequences]
    reference: dict[str, dict[str, Any]] = {}
    for name in PROPERTIES:
        values = np.array([p[name] for p in props])
        q10, q25, q75, q90 = np.percentile(values, [10, 25, 75, 90])
        iqr = float(q75 - q25)
        full_range = float(values.max() - values.min())
        # A zero IQR falls back to the full range; if that is zero too, the property cannot be normalised.
        scale = iqr if iqr > 0 else (full_range if full_range > 0 else None)
        reference[name] = {
            "q10": float(q10),
            "q90": float(q90),
            "iqr": iqr,
            "range": full_range,
            "scale": scale,
        }
    return reference


def physicochemical_penalty(
    properties: dict[str, float], reference: dict[str, dict[str, Any]]
) -> tuple[float, dict[str, float]]:
    """Distance outside each property's percentile band in scale units, summed; properties without a scale are skipped."""
    penalties: dict[str, float] = {}
    for name in PROPERTIES:
        ref = reference[name]
        if ref["scale"] is None:
            continue
        x = properties[name]
        if x < ref["q10"]:
            penalties[name] = (ref["q10"] - x) / ref["scale"]
        elif x > ref["q90"]:
            penalties[name] = (x - ref["q90"]) / ref["scale"]
        else:
            penalties[name] = 0.0
    return float(sum(penalties.values())), penalties


def nearest_distance(sequence: str, others: list[str]) -> float:
    """Normalised Levenshtein distance to the nearest other sequence; 0 when there is none."""
    distances = [
        Levenshtein.distance(sequence, o) / max(len(sequence), len(o))
        for o in others
        if o != sequence
    ]
    return min(distances) if distances else 0.0


@dataclass
class Individual:
    sequence: str
    generation: int
    parents: tuple[str, ...] = ()
    history: list[str] = field(default_factory=list)
    active: bool = False
    properties: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    p_phys: float = 0.0
    diversity: float = 0.0

    def key(self) -> tuple[int, float, float]:
        """Higher is better: active first, then lower penalty, then higher diversity."""
        return (int(self.active), -self.p_phys, self.diversity)


def mutate_once(sequence: str, rng: random.Random) -> tuple[str, str]:
    """Exactly one substitution, insertion or deletion, each with probability 1/3."""
    kind = rng.choice(MUTATIONS)
    if kind == "substitution":
        pos = rng.randrange(len(sequence))
        new = rng.choice([a for a in AMINO_ACIDS if a != sequence[pos]])
        return sequence[:pos] + new + sequence[pos + 1 :], f"substitution@{pos}:{sequence[pos]}>{new}"
    if kind == "insertion":
        pos = rng.randrange(len(sequence) + 1)
        new = rng.choice(AMINO_ACIDS)
        return sequence[:pos] + new + sequence[pos:], f"insertion@{pos}:{new}"
    pos = rng.randrange(len(sequence))
    return sequence[:pos] + sequence[pos + 1 :], f"deletion@{pos}:{sequence[pos]}"


def crossover(a: str, b: str, rng: random.Random) -> str:
    """Single-point crossover: head of `a` joined to the tail of `b`."""
    if len(a) < 2 or len(b) < 2:
        return a
    return a[: rng.randint(1, len(a) - 1)] + b[rng.randint(1, len(b) - 1) :]


def tournament(scored: list[Individual], size: int, rng: random.Random) -> Individual:
    return max((rng.choice(scored) for _ in range(size)), key=Individual.key)


def run_cosmetic_ga(
    seed_sequence: str,
    modifications: dict[str, Any],
    reference: dict[str, dict[str, Any]],
    predict_active: PrimaryPredictor,
    min_length: int,
    max_length: int,
    ph: float,
    rng: random.Random,
    pop_size: int = 7,
    n_generations: int = 25,
    tournament_size: int = 3,
    crossover_prob: float = 0.5,
) -> list[Individual]:
    """Evolve one seed and return the final population; modifications are inherited unchanged, never mutated."""
    decisions: dict[str, bool] = {}

    def valid(sequence: str) -> bool:
        return not validate_sequence_and_modifications(
            sequence, modifications, min_length, max_length
        )

    def score(individuals: list[Individual]) -> None:
        new = [s for s in dict.fromkeys(i.sequence for i in individuals) if s not in decisions]
        if new:
            decisions.update(zip(new, (bool(d) for d in predict_active(new))))
        sequences = [i.sequence for i in individuals]
        for ind in individuals:
            ind.active = decisions[ind.sequence]
            ind.properties = compute_properties(ind.sequence, ph)
            ind.p_phys, ind.penalties = physicochemical_penalty(ind.properties, reference)
            ind.diversity = nearest_distance(ind.sequence, sequences)

    population = [Individual(seed_sequence, 0, history=["seed"])]
    seen = {seed_sequence}
    for _ in range((pop_size - 1) * MAX_ATTEMPTS_PER_SLOT):
        if len(population) >= pop_size:
            break
        child, step = mutate_once(seed_sequence, rng)
        if child in seen or not valid(child):
            continue
        seen.add(child)
        population.append(Individual(child, 0, (seed_sequence,), ["seed", f"g0:{step}"]))
    score(population)

    for generation in range(1, n_generations + 1):
        offspring: list[Individual] = []
        for _ in range(pop_size * MAX_ATTEMPTS_PER_SLOT):
            if len(offspring) >= pop_size:
                break
            parent_a = tournament(population, tournament_size, rng)
            parent_b = tournament(population, tournament_size, rng)
            history = list(parent_a.history)
            sequence = parent_a.sequence
            if rng.random() < crossover_prob:
                sequence = crossover(parent_a.sequence, parent_b.sequence, rng)
                history.append(f"g{generation}:crossover({parent_b.sequence})")
            sequence, step = mutate_once(sequence, rng)
            history.append(f"g{generation}:{step}")
            if sequence in seen or not valid(sequence):
                continue
            seen.add(sequence)
            offspring.append(
                Individual(sequence, generation, (parent_a.sequence, parent_b.sequence), history)
            )
        pool = population + offspring
        score(pool)
        pool.sort(key=Individual.key, reverse=True)
        population = pool[:pop_size]
        score(population)  # diversity is recalculated against the surviving population
    return sorted(population, key=Individual.key, reverse=True)


def _make_id(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]


class RouteACosmetic(Stage4Route):
    """Seed-based GA for cosmetic peptides (Stage 4, cosmetic route A); the primary-function decision is injected."""

    def run(self) -> list[Candidate]:
        cfg = self.config
        min_length, max_length, ph = cfg["min_length"], cfg["max_length"], cfg["product_ph"]
        seeds = [s for s in cfg["seed_candidates"] if s.get("sequence")]
        if not seeds:
            return []
        reference = build_reference([s["sequence"] for s in seeds], ph)
        unnormalizable = [n for n, r in reference.items() if r["scale"] is None]
        if unnormalizable:
            logger.warning(
                "routeA_cosmetic.property_not_normalizable",
                extra={"properties": unnormalizable},
            )
        seed_sequences = {s["sequence"] for s in seeds}
        # Stage 5 builds its seed-relative percentile bounds from this set.
        seed_set = [
            {"sequence": s["sequence"], "modifications": s.get("modifications") or {}}
            for s in seeds
        ]

        best: dict[str, tuple[int, Candidate]] = {}
        for seed in seeds:
            modifications = seed.get("modifications") or {}
            rng = random.Random(f"{cfg['random_seed']}:{seed['id']}")
            population = run_cosmetic_ga(
                seed["sequence"],
                modifications,
                reference,
                cfg["predict_active"],
                min_length,
                max_length,
                ph,
                rng,
                pop_size=cfg["pop_size"],
                n_generations=cfg["n_generations"],
                tournament_size=cfg["tournament_size"],
                crossover_prob=cfg["crossover_prob"],
            )
            # Seeds are not new candidates, and an inactive variant is infeasible.
            feasible = [
                i for i in population if i.active and i.sequence not in seed_sequences
            ][: cfg["top_k"]]
            for rank, ind in enumerate(feasible, start=1):
                candidate = Candidate(
                    id=_make_id(ind.sequence),
                    sequence=ind.sequence,
                    predictions={
                        "route": "A_cosmetic",
                        "parent_seed_id": seed["id"],
                        "generation": ind.generation,
                        "mutation_history": ind.history,
                        "primary_function": cfg["primary_function"],
                        "primary_active": int(ind.active),
                        "physicochemical_properties": ind.properties,
                        "physicochemical_penalties": ind.penalties,
                        "p_phys": ind.p_phys,
                        "diversity": ind.diversity,
                        "rank": rank,
                        "modifications": modifications,
                        "seed_set": seed_set,
                        "reference_unnormalizable_properties": unnormalizable,
                    },
                )
                kept = best.get(ind.sequence)
                if kept is None or rank < kept[0]:
                    best[ind.sequence] = (rank, candidate)
        return [candidate for _, candidate in best.values()]
