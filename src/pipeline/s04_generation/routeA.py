# # Route A: Genetic-Algorithm Peptide Optimizer v1
#
# Minimal DEAP-based loop: seed peptide -> population -> score -> select -> crossover/mutate -> repeat. The production predictor adapters are explicit placeholders; the mock predictors are for demo/tests only.

from __future__ import annotations

import hashlib
import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple, TypedDict
from common.logging import get_logger
from tqdm import tqdm
from src.schemas.stage4_route import Stage4Route
from schemas.candidate import Candidate
from dataclasses import dataclass, field
from model_store.amp_classifier_v1 import AMPClassifier
from model_store.hemolysis_predictor_v1 import ReplicatedHemoPI2Predictor
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

try:
    from deap import base, creator
except ImportError as exc:
    raise ImportError(
        "DEAP is required. Install it with: pip install deap pandas numpy"
    ) from exc

logger = get_logger(__name__)

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale, used for the aggregation-score sliding window
# (modlAMP's own scales are normalized differently and built for the moment/global
# hydrophobicity calculations below, not for a patch-detection heuristic).
KYTE_DOOLITTLE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
AGGREGATION_WINDOW = (
    5  # residues — Zyggregator-style patch detection operates on short windows
)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # Keep
DATA_DIR = Path("./src/data")  # Keep
REQUIRED_HISTORY_COLUMNS = [
    "variant_id",
    "parent_id",
    "sequence",
    "generation",
    "fitness_score",
    "amp_prob",
    "hemolysis_phc50",
    "timestamp",
]


MAX_CONSTRAINT_RETRIES = 25  # bounded retry before falling back, so a pathological config.constraint_config can't hang the GA loop forever

# Fitness floor assigned to a physicochemically-rejected candidate — well below any
# achievable fitness (range (0, 1]) so it never wins selection, without breaking the
# fixed-size population/history bookkeeping the GA loop assumes.
REJECTED_FITNESS_FLOOR = -10.0

# Rejected candidates are floored on fitness but still need a stand-in pHC50 for the
# history record; a high value reads as "maximally hemolytic" alongside the floor.
REJECTED_HEMOLYSIS_PHC50 = 10.0


MODEL_STORE_DIR = Path(__file__).resolve().parents[3] / "model_store"
if str(MODEL_STORE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_STORE_DIR))

_amp_model_cache = None
_hemolysis_model_cache = None


@dataclass(frozen=True)
class ConstraintConfig:
    """
    Hard-reject thresholds.
    TODO: Defaults are broad placeholders — tune per target class the way the rest of this pipeline's thresholds are tuned.
    """

    min_length: int = 8
    max_length: int = 40
    charge_min: float = -3.0
    charge_max: float = 8.0
    hydro_min: float = -1.0
    hydro_max: float = 1.0
    aggregation_threshold: float = 3.0
    ph: float = 7.4

    @classmethod
    def from_params(cls, params: dict) -> "ConstraintConfig":
        """Build from config.params, falling back to this dataclass's own
        defaults for any field the caller didn't set."""
        defaults = cls()
        return cls(
            **{
                field_name: params.get(field_name, getattr(defaults, field_name))
                for field_name in defaults.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class FilterResult:
    sequence: str
    passed: bool
    fail_reasons: list[str] = field(default_factory=list)
    computed_attributes: dict = field(default_factory=dict)


def compute_attributes(sequence: str, ph: float = 7.4) -> dict:
    gd = GlobalDescriptor([sequence])
    gd.calculate_charge(ph=ph, amide=False)
    net_charge = float(gd.descriptor[0][0])

    gd = GlobalDescriptor([sequence])
    gd.isoelectric_point()
    isoelectric_point = float(gd.descriptor[0][0])

    gd = GlobalDescriptor([sequence])
    gd.calculate_MW(amide=False)
    molecular_weight = float(gd.descriptor[0][0])

    pd_global = PeptideDescriptor([sequence], "eisenberg")
    pd_global.calculate_global()
    hydrophobicity = float(pd_global.descriptor[0][0])

    pd_moment = PeptideDescriptor([sequence], "eisenberg")
    pd_moment.calculate_moment()
    hydrophobic_moment = float(pd_moment.descriptor[0][0])

    aggregation_score = _aggregation_score(sequence)
    solubility_flag = _solubility_flag(net_charge, hydrophobicity)

    return {
        "length": len(sequence),
        "molecular_weight": molecular_weight,
        "net_charge": net_charge,
        "isoelectric_point": isoelectric_point,
        "hydrophobicity": hydrophobicity,
        "hydrophobic_moment": hydrophobic_moment,
        "aggregation_score": aggregation_score,
        "solubility_flag": solubility_flag,
    }


def _aggregation_score(sequence: str) -> float:
    """Simple hydrophobic-patch heuristic: the highest mean Kyte-Doolittle
    hydrophobicity over any contiguous AGGREGATION_WINDOW-residue stretch.
    Not a real Zyggregator run (that needs its own model/webserver) — a cheap
    proxy for the same idea, that a hydrophobic patch drives aggregation risk."""
    if len(sequence) < AGGREGATION_WINDOW:
        window = (
            AGGREGATION_WINDOW if len(sequence) >= AGGREGATION_WINDOW else len(sequence)
        )
    else:
        window = AGGREGATION_WINDOW
    if window == 0:
        return 0.0

    scores = [KYTE_DOOLITTLE[aa] for aa in sequence]
    windows = [
        sum(scores[i : i + window]) / window for i in range(len(scores) - window + 1)
    ]
    return max(windows) if windows else 0.0


def _solubility_flag(net_charge: float, hydrophobicity: float) -> str:
    """Heuristic from charge + hydrophobicity: a peptide near-neutral charge and
    strongly hydrophobic is the classic low-solubility combination."""
    if abs(net_charge) < 1.0 and hydrophobicity > 0.5:
        return "very_low"
    if abs(net_charge) < 2.0 and hydrophobicity > 0.2:
        return "low"
    if abs(net_charge) >= 3.0 or hydrophobicity <= 0.0:
        return "high"
    return "moderate"


def physicochemical_filter(
    sequence: str, config: ConstraintConfig = ConstraintConfig()
) -> FilterResult:
    sequence = sequence.strip().upper()
    if not sequence or any(aa not in AMINO_ACID_SET for aa in sequence):
        return FilterResult(
            sequence=sequence,
            passed=False,
            fail_reasons=["invalid_sequence"],
            computed_attributes={},
        )

    attrs = compute_attributes(sequence, ph=config.ph)

    fail_reasons = []
    if attrs["length"] < config.min_length or attrs["length"] > config.max_length:
        fail_reasons.append("length_out_of_range")
    if not (config.charge_min <= attrs["net_charge"] <= config.charge_max):
        fail_reasons.append("charge_out_of_range")
    if not (config.hydro_min <= attrs["hydrophobicity"] <= config.hydro_max):
        fail_reasons.append("hydrophobicity_out_of_range")
    if attrs["aggregation_score"] > config.aggregation_threshold:
        fail_reasons.append("aggregation_score_too_high")
    if attrs["solubility_flag"] == "very_low":
        fail_reasons.append("solubility_very_low")

    return FilterResult(
        sequence=sequence,
        passed=not fail_reasons,
        fail_reasons=fail_reasons,
        computed_attributes=attrs,
    )


def validate_inputs(
    seed_sequence: str,
    min_length: int,
    max_length: int,
    pop_size: int,
    n_generations: int,
    mutation_rate: float,
) -> None:
    if not isinstance(seed_sequence, str) or not seed_sequence:
        raise ValueError("seed_sequence must be a non-empty string.")
    if any(aa not in AMINO_ACID_SET for aa in seed_sequence):
        raise ValueError(
            f"seed_sequence contains non-standard amino acids: {sorted(set(seed_sequence) - AMINO_ACID_SET)}"
        )
    if min_length <= 5:
        raise ValueError("min_length must be > 5.")
    if max_length < min_length:
        raise ValueError("max_length must be >= min_length.")
    if not (min_length <= len(seed_sequence) <= max_length):
        raise ValueError(
            "seed_sequence length must be between min_length and max_length."
        )
    if max_length > 50:
        raise ValueError("max_length must be <= 50.")

    if pop_size < 2:
        raise ValueError("pop_size must be >= 2.")
    if n_generations < 1:
        raise ValueError("n_generations must be >= 1.")
    if not (0 <= mutation_rate <= 1):
        raise ValueError("mutation_rate must be between 0 and 1.")


def load_data_peptides(
    data_dir: Path = DATA_DIR, min_length: int = 8, max_length: int = 40
) -> pd.DataFrame:
    """Load valid standard-amino-acid peptide sequences from src/data/*.xlsx."""
    rows = []
    for path in sorted(data_dir.glob("*.xlsx")):
        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet_name)
            sequence_columns = [
                col
                for col in df.columns
                if str(col).strip().lower() in {"sequence", "peptide_sequence", "seq"}
            ]
            for sequence_column in sequence_columns:
                for raw_sequence in df[sequence_column].dropna():
                    sequence = str(raw_sequence).strip().upper()
                    if (
                        min_length <= len(sequence) <= max_length
                        and set(sequence) <= AMINO_ACID_SET
                    ):
                        rows.append(
                            {
                                "source_file": path.name,
                                "sheet": sheet_name,
                                "sequence": sequence,
                                "length": len(sequence),
                            }
                        )
    peptides = (
        pd.DataFrame(rows).drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    )
    if peptides.empty:
        raise ValueError(f"No valid standard amino-acid peptides found in {data_dir}.")
    return peptides


def validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )


def enforce_length(
    sequence: str, min_length: int, max_length: int, rng: random.Random
) -> str:
    """Trim invalid residues and pad short peptides back to the allowed length."""
    cleaned = "".join(aa for aa in sequence if aa in AMINO_ACID_SET)
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    while len(cleaned) < min_length:
        cleaned += rng.choice(AMINO_ACIDS)
    return cleaned


def substitution_mutation(
    sequence: str, mutation_rate: float, rng: random.Random
) -> str:
    """Randomly replace residues with other standard amino acids at a target rate."""
    validate_sequence(sequence)
    out = []
    for aa in sequence:
        if rng.random() < mutation_rate:
            out.append(
                rng.choice([candidate for candidate in AMINO_ACIDS if candidate != aa])
            )
        else:
            out.append(aa)
    return "".join(out)


def truncation_mutation(sequence: str, min_length: int, rng: random.Random) -> str:
    """Remove a random segment from either end while preserving the minimum length."""
    validate_sequence(sequence)
    if len(sequence) <= min_length:
        return sequence
    n_remove = rng.randint(1, len(sequence) - min_length)
    return (
        sequence[n_remove:] if rng.choice(["N", "C"]) == "N" else sequence[:-n_remove]
    )


def mutate_sequence(
    sequence: str,
    min_length: int,
    max_length: int,
    mutation_rate: float,
    rng: random.Random,
) -> str:
    """Apply substitution and optional truncation, then restore valid length bounds."""
    sequence = substitution_mutation(sequence, mutation_rate, rng)
    if rng.random() < mutation_rate:
        sequence = truncation_mutation(sequence, min_length, rng)
    return enforce_length(sequence, min_length, max_length, rng)


def single_point_crossover(
    parent_a: str, parent_b: str, min_length: int, max_length: int, rng: random.Random
) -> Tuple[str, str]:
    """Swap tails from two parents at a random cut point to create two children."""
    validate_sequence(parent_a)
    validate_sequence(parent_b)
    if len(parent_a) < 2 or len(parent_b) < 2:
        return enforce_length(parent_a, min_length, max_length, rng), enforce_length(
            parent_b, min_length, max_length, rng
        )
    cut_a = rng.randint(1, len(parent_a) - 1)
    cut_b = rng.randint(1, len(parent_b) - 1)
    child_a = parent_a[:cut_a] + parent_b[cut_b:]
    child_b = parent_b[:cut_b] + parent_a[cut_a:]
    return enforce_length(child_a, min_length, max_length, rng), enforce_length(
        child_b, min_length, max_length, rng
    )


def hard_constrained_mutant(
    seed_sequence: str,
    min_length: int,
    max_length: int,
    mutation_rate: float,
    rng: random.Random,
    constraint_config: ConstraintConfig,
) -> str:
    """GA hard-constraint: keep mutating until the result clears physicochemical_filter
    (Stage 5's config.constraint_config), rather than adding a candidate that was never
    going to be viable. Falls back to the last attempt if MAX_CONSTRAINT_RETRIES is
    exhausted — record() will still fitness-floor it via the pre-predictor gate."""
    candidate = mutate_sequence(
        seed_sequence, min_length, max_length, mutation_rate, rng
    )
    for _ in range(MAX_CONSTRAINT_RETRIES - 1):
        if physicochemical_filter(candidate, constraint_config).passed:
            break
        candidate = mutate_sequence(
            seed_sequence, min_length, max_length, mutation_rate, rng
        )
    return candidate


def hard_constrained_offspring(
    parent_a: str,
    parent_b: str,
    min_length: int,
    max_length: int,
    mutation_rate: float,
    rng: random.Random,
    constraint_config: ConstraintConfig,
) -> Tuple[str, str] | None:
    """Same hard-constraint retry, applied to a crossover + mutation pair."""

    def _make_pair() -> Tuple[str, str]:
        """One crossover + mutation attempt, no constraint retry."""
        child_a, child_b = single_point_crossover(
            parent_a, parent_b, min_length, max_length, rng
        )
        child_a = mutate_sequence(child_a, min_length, max_length, mutation_rate, rng)
        child_b = mutate_sequence(child_b, min_length, max_length, mutation_rate, rng)
        return child_a, child_b

    child_a, child_b = _make_pair()
    a_ok = physicochemical_filter(child_a, constraint_config).passed
    b_ok = physicochemical_filter(child_b, constraint_config).passed
    for _ in range(MAX_CONSTRAINT_RETRIES - 1):
        if a_ok and b_ok:
            break
        child_a, child_b = _make_pair()
        a_ok = physicochemical_filter(child_a, constraint_config).passed
        b_ok = physicochemical_filter(child_b, constraint_config).passed
    if not (a_ok and b_ok):
        return None
    return child_a, child_b


@dataclass(frozen=True)
class Evaluation:
    fitness_score: float
    amp_prob: float
    hemolysis_phc50: float


class SimpleHallOfFame:
    """Small elitism helper equivalent to a size-1 HallOfFame for this v1."""

    def __init__(self) -> None:
        self.best = None

    def update(self, population: List) -> None:
        candidate = max(population, key=lambda ind: ind.fitness.values[0])
        if (
            self.best is None
            or candidate.fitness.values[0] > self.best.fitness.values[0]
        ):
            self.best = candidate


def tournament_selection(
    population: List, k: int, tournsize: int, rng: random.Random
) -> List:
    selected = []
    for _ in range(k):
        aspirants = [rng.choice(population) for _ in range(tournsize)]
        selected.append(max(aspirants, key=lambda ind: ind.fitness.values[0]))
    return selected


def _hemolysis_safety_factor(hemolysis_phc50: float) -> float:
    """10^(6-pHC50) / (10^(6-pHC50) + 100): -> 1 as pHC50 falls (safer), -> 0 as
    pHC50 rises (more hemolytic at lower concentration)."""
    numerator = 10 ** (6 - hemolysis_phc50)
    return numerator / (numerator + 100)


def _evaluate_with_cache(
    sequence: str,
    cache: Dict[str, Evaluation],
    amp_predictor: Callable[[str], float],
    hemolysis_predictor: Callable[[str], float],
    constraint_config: ConstraintConfig,
) -> Evaluation:
    if sequence not in cache:
        filter_result = physicochemical_filter(sequence, constraint_config)
        if not filter_result.passed:
            # Rejected before spending a predictor call — this is the "cheap gate" role
            # physicochemical_filter plays pre-Model-1/2, same as Routes B/C.
            cache[sequence] = Evaluation(
                REJECTED_FITNESS_FLOOR, 0.0, REJECTED_HEMOLYSIS_PHC50
            )
        else:
            amp_prob = amp_predictor(sequence)
            hemolysis_phc50 = hemolysis_predictor(sequence)
            if not 0 <= amp_prob <= 1:
                raise ValueError(f"AMP predictor returned {amp_prob}; expected [0, 1].")
            fitness_score = amp_prob * _hemolysis_safety_factor(hemolysis_phc50)
            cache[sequence] = Evaluation(fitness_score, amp_prob, hemolysis_phc50)
    return cache[sequence]


def _setup_deap() -> base.Toolbox:
    if not hasattr(creator, "PeptideFitnessMax"):
        creator.create("PeptideFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "PeptideIndividual"):
        creator.create(
            "PeptideIndividual", list, fitness=getattr(creator, "PeptideFitnessMax")
        )
    return base.Toolbox()


def _make_variant_id(sequence: str, generation: int, ordinal: int) -> str:
    return hashlib.sha1(
        f"{generation}:{ordinal}:{sequence}".encode("utf-8")
    ).hexdigest()[:16]


def _make_individual(sequence: str, variant_id: str, parent_id: str, generation: int):
    individual = getattr(creator, "PeptideIndividual")(sequence)
    individual.variant_id = variant_id
    individual.parent_id = parent_id
    individual.generation = generation
    return individual


def _summarize_generation(generation: int, population: List) -> None:
    fitnesses = [ind.fitness.values[0] for ind in population]
    logger.info(
        "ga.generation",
        extra={
            "generation": generation,
            "best_fitness": round(max(fitnesses), 4),
            "mean_fitness": round(float(np.mean(fitnesses)), 4),
        },
    )


def _real_amp_predictor() -> AMPClassifier:
    global _amp_model_cache
    if _amp_model_cache is None:
        _amp_model_cache = AMPClassifier()
    return _amp_model_cache


def _real_hemolysis_predictor() -> ReplicatedHemoPI2Predictor:
    global _hemolysis_model_cache
    if _hemolysis_model_cache is None:
        _hemolysis_model_cache = ReplicatedHemoPI2Predictor()
    return _hemolysis_model_cache


def predict_amp(sequence: str) -> float:
    """Real AMP probability from amp_classifier_v1's saved ensemble."""
    return _real_amp_predictor().predict_proba(sequence)


def predict_hemolysis_phc50(sequence: str) -> float:
    """Real hemolysis model output from hemolysis_predictor_v1: pHC50, not a probability."""
    return _real_hemolysis_predictor().predict_phc50(sequence)


class BestIndividual(TypedDict):
    sequence: str
    fitness_score: float
    amp_prob: float
    hemolysis_phc50: float
    generation: int


class GAResult(TypedDict):
    best: BestIndividual
    history: pd.DataFrame
    population: List


def run_ga(
    seed_sequence: str,
    min_length: int,
    max_length: int,
    pop_size: int = 10,
    n_generations: int = 20,
    mutation_rate: float = 0.1,
    random_seed: int = 42,
    amp_predictor: Callable[[str], float] | None = None,
    hemolysis_predictor: Callable[[str], float] | None = None,
    tournsize: int = 3,
    constraint_config: ConstraintConfig | None = None,
) -> GAResult:
    validate_inputs(
        seed_sequence, min_length, max_length, pop_size, n_generations, mutation_rate
    )
    if tournsize < 2:
        raise ValueError("tournsize must be >= 2.")
    amp_predictor = predict_amp if amp_predictor is None else amp_predictor
    hemolysis_predictor = (
        predict_hemolysis_phc50 if hemolysis_predictor is None else hemolysis_predictor
    )
    constraint_config = constraint_config if constraint_config is not None else ConstraintConfig()
    random.seed(random_seed)
    np.random.seed(random_seed)
    rng = random.Random(random_seed)
    toolbox = _setup_deap()
    toolbox.register("select", tournament_selection, tournsize=tournsize, rng=rng)
    eval_cache: Dict[str, Evaluation] = {}
    history_rows: List[Dict[str, object]] = []
    hall_of_fame = SimpleHallOfFame()  # seems unnecessary
    ordinal = 0

    def record(individual) -> None:
        evaluation = _evaluate_with_cache(
            "".join(individual), eval_cache, amp_predictor, hemolysis_predictor, constraint_config
        )
        individual.fitness.values = (evaluation.fitness_score,)
        history_rows.append(
            {
                "variant_id": individual.variant_id,
                "parent_id": individual.parent_id,
                "sequence": "".join(individual),
                "generation": individual.generation,
                "fitness_score": evaluation.fitness_score,
                "amp_prob": evaluation.amp_prob,
                "hemolysis_phc50": evaluation.hemolysis_phc50,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    population = []
    seed_id = _make_variant_id(seed_sequence, 0, ordinal)
    ordinal += 1
    population.append(_make_individual(seed_sequence, seed_id, "", 0))
    while len(population) < pop_size:
        sequence = hard_constrained_mutant(
            seed_sequence, min_length, max_length, mutation_rate, rng, constraint_config
        )
        variant_id = _make_variant_id(sequence, 0, ordinal)
        ordinal += 1
        population.append(_make_individual(sequence, variant_id, seed_id, 0))

    for individual in population:
        record(individual)
    hall_of_fame.update(population)
    _summarize_generation(0, population)

    for generation in range(1, n_generations + 1):
        clone = getattr(toolbox, "clone")
        select = getattr(toolbox, "select")
        selected = list(map(clone, select(population, pop_size)))
        offsprings = []
        for i in range(0, pop_size, 2):
            if len(offsprings) >= pop_size:
                break
            parent_a = selected[i]
            parent_b = selected[(i + 1) % pop_size]
            res = hard_constrained_offspring(
                "".join(parent_a),
                "".join(parent_b),
                min_length,
                max_length,
                mutation_rate,
                rng,
                constraint_config,
            )
            if res is None:
                continue
            child_seq_a, child_seq_b = res
            parent_ids = ",".join(sorted([parent_a.variant_id, parent_b.variant_id]))
            for child_seq in (child_seq_a, child_seq_b):
                if len(offsprings) >= pop_size:
                    break
                variant_id = _make_variant_id(child_seq, generation, ordinal)
                ordinal += 1
                offsprings.append(
                    _make_individual(child_seq, variant_id, parent_ids, generation)
                )
        for individual in offsprings:
            record(individual)
        population = offsprings
        if not population:
            # Every pair this generation exhausted MAX_CONSTRAINT_RETRIES with no
            # passing offspring -- nothing to select parents from next generation,
            # so stop here rather than crashing on hall_of_fame.update's max([]).
            logger.warning(
                "ga.generation_collapsed",
                extra={"generation": generation},
            )
            break
        hall_of_fame.update(population)
        _summarize_generation(generation, population)

    history = pd.DataFrame(history_rows, columns=REQUIRED_HISTORY_COLUMNS)
    best_row = history.loc[history["fitness_score"].idxmax()].to_dict()
    best = {
        "sequence": best_row["sequence"],
        "fitness_score": float(best_row["fitness_score"]),
        "amp_prob": float(best_row["amp_prob"]),
        "hemolysis_phc50": float(best_row["hemolysis_phc50"]),
        "generation": int(best_row["generation"]),
    }
    logger.info(
        "ga.best",
        extra={
            "sequence": best["sequence"],
            "amp_prob": round(best["amp_prob"], 4),
            "hemolysis_phc50": round(best["hemolysis_phc50"], 4),
            "fitness_score": round(best["fitness_score"], 4),
            "generation": best["generation"],
        },
    )
    return {"best": best, "history": history, "population": population}


class RouteA(Stage4Route):
    """Reference-guided candidate generation: GA optimization of a seed peptide
    (Stage 4, route A). See CLAUDE.md's multi-route generation convention."""

    def __init__(self, config: dict[str, Any] = {}):
        super().__init__(config)

    def run(self) -> list[Candidate]:
        seed_candidates = self.config.get("seed_candidates", [])
        candidates: list[Candidate] = []
        for seed_params in tqdm(seed_candidates, desc="RouteA"):
            seed_sequence = seed_params.get("seed_sequence")
            if not seed_sequence:
                raise ValueError(
                    "RouteA requires config.params['seed_sequence'] (reference-guided "
                    "generation needs a seed peptide to optimize)."
                )
            constraint_config = ConstraintConfig.from_params(seed_params)
            try:
                result = run_ga(
                    seed_sequence=seed_sequence,
                    min_length=seed_params.get("min_length", constraint_config.min_length),
                    max_length=seed_params.get("max_length", constraint_config.max_length),
                    pop_size=seed_params.get("pop_size", 10),
                    n_generations=seed_params.get("n_generations", 20),
                    mutation_rate=seed_params.get("mutation_rate", 0.1),
                    random_seed=seed_params.get("random_seed", 42),
                    constraint_config=constraint_config,
                )
            except Exception as e:
                logger.error(
                    "routeA.ga_failed",
                    extra={
                        "seed_sequence": seed_sequence,
                        "error": str(e),
                    },
                )
                continue
            # `population` holds DEAP individuals (variant_id/parent_id/generation
            # as attributes, fitness via .fitness.values[0]); amp_prob and
            # hemolysis_phc50 only live in `history`, keyed by variant_id.
            history_by_variant = result["history"].set_index("variant_id")
            for individual in result["population"]:
                history_row = history_by_variant.loc[individual.variant_id]
                candidates.append(
                    Candidate(
                        id=individual.variant_id,
                        sequence="".join(individual),
                        predictions={
                            "amp_prob": float(history_row["amp_prob"]),
                            "hemolysis_phc50": float(history_row["hemolysis_phc50"]),
                            "fitness_score": float(individual.fitness.values[0]),
                            "generation": int(individual.generation),
                            "route": "A",
                        },
                    )
                )

        return candidates
