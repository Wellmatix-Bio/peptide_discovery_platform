# Unit tests for s04_generation.
#
# run_ga() is tested directly with mock amp/hemolysis predictors so no real
# model (GPU, ESM2, XGBoost) is loaded. RouteA.run() currently cannot be
# exercised the same way: it calls run_ga() without predictor overrides, so
# it always hits the real models — its own tests are slow for that reason.

from __future__ import annotations

import math

import pandas as pd
import pytest

from pipeline.s04_generation.routeA import (
    REJECTED_FITNESS_FLOOR,
    REQUIRED_HISTORY_COLUMNS,
    RouteA,
    _hemolysis_safety_factor,
    run_ga,
)
from pipeline.s04_generation.routeB import RouteB

# A short seed known to pass physicochemical_filter under DEFAULT_CONSTRAINT_CONFIG
# (net charge, hydrophobicity, aggregation, length all within default bounds).
SEED_SEQUENCE = "GLFDIVKKVVGALGSL"  # 16 aa, magainin-family-like AMP


def mock_amp_predictor(sequence: str) -> float:
    """Deterministic, sequence-independent stand-in AMP probability."""
    return 0.8


def mock_hemolysis_predictor(sequence: str) -> float:
    """Deterministic, sequence-independent stand-in pHC50 (low -> safe)."""
    return 2.0


def failing_amp_predictor(sequence: str) -> float:
    return -0.5  # out of [0, 1], used to test the predictor-contract check


class TestRunGa:
    def test_returns_expected_result_shape(self) -> None:
        result = run_ga(
            seed_sequence=SEED_SEQUENCE,
            min_length=6,
            max_length=40,
            pop_size=6,
            n_generations=2,
            mutation_rate=0.1,
            random_seed=42,
            amp_predictor=mock_amp_predictor,
            hemolysis_predictor=mock_hemolysis_predictor,
        )
        assert set(result.keys()) == {"best", "history", "population"}
        assert set(REQUIRED_HISTORY_COLUMNS) <= set(result["history"].columns)
        assert set(result["best"].keys()) == {
            "sequence", "fitness_score", "amp_prob", "hemolysis_phc50", "generation",
        }

    def test_first_generation_seeds_from_seed_sequence(self) -> None:
        result = run_ga(
            seed_sequence=SEED_SEQUENCE,
            min_length=6,
            max_length=40,
            pop_size=6,
            n_generations=1,
            mutation_rate=0.1,
            random_seed=42,
            amp_predictor=mock_amp_predictor,
            hemolysis_predictor=mock_hemolysis_predictor,
        )
        gen0 = result["history"][result["history"]["generation"] == 0]
        assert SEED_SEQUENCE in set(gen0["sequence"])

    def test_deterministic_with_fixed_seed(self) -> None:
        kwargs = dict(
            seed_sequence=SEED_SEQUENCE,
            min_length=6,
            max_length=40,
            pop_size=8,
            n_generations=3,
            mutation_rate=0.15,
            random_seed=123,
            amp_predictor=mock_amp_predictor,
            hemolysis_predictor=mock_hemolysis_predictor,
        )
        result_a = run_ga(**kwargs)
        result_b = run_ga(**kwargs)
        cols = ["variant_id", "parent_id", "sequence", "generation", "fitness_score", "amp_prob", "hemolysis_phc50"]
        pd.testing.assert_frame_equal(result_a["history"][cols], result_b["history"][cols])
        assert result_a["best"]["sequence"] == result_b["best"]["sequence"]

    def test_fitness_score_matches_amp_times_safety_factor(self) -> None:
        result = run_ga(
            seed_sequence=SEED_SEQUENCE,
            min_length=6,
            max_length=40,
            pop_size=6,
            n_generations=1,
            mutation_rate=0.1,
            random_seed=42,
            amp_predictor=mock_amp_predictor,
            hemolysis_predictor=mock_hemolysis_predictor,
        )
        history = result["history"]
        accepted = history[history["fitness_score"] != REJECTED_FITNESS_FLOOR]
        expected = accepted["amp_prob"] * accepted["hemolysis_phc50"].apply(_hemolysis_safety_factor)
        assert (accepted["fitness_score"] - expected).abs().max() < 1e-9

    def test_rejected_candidates_are_fitness_floored(self) -> None:
        # A high mutation rate makes rejects likely without guaranteeing them
        # (hard_constrained_mutant/offspring retry against the filter).
        result = run_ga(
            seed_sequence=SEED_SEQUENCE,
            min_length=6,
            max_length=40,
            pop_size=10,
            n_generations=2,
            mutation_rate=0.9,
            random_seed=7,
            amp_predictor=mock_amp_predictor,
            hemolysis_predictor=mock_hemolysis_predictor,
        )
        history = result["history"]
        rejected = history[history["fitness_score"] == REJECTED_FITNESS_FLOOR]
        # Not asserting rejects necessarily occur (stochastic), only that if they
        # do, they carry the documented sentinel values.
        if not rejected.empty:
            assert (rejected["amp_prob"] == 0.0).all()

    def test_best_has_max_fitness_in_history(self) -> None:
        result = run_ga(
            seed_sequence=SEED_SEQUENCE,
            min_length=6,
            max_length=40,
            pop_size=6,
            n_generations=2,
            mutation_rate=0.1,
            random_seed=42,
            amp_predictor=mock_amp_predictor,
            hemolysis_predictor=mock_hemolysis_predictor,
        )
        assert result["best"]["fitness_score"] == pytest.approx(
            result["history"]["fitness_score"].max()
        )

    def test_predictor_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="AMP predictor returned"):
            run_ga(
                seed_sequence=SEED_SEQUENCE,
                min_length=6,
                max_length=40,
                pop_size=2,
                n_generations=1,
                mutation_rate=0.1,
                random_seed=42,
                amp_predictor=failing_amp_predictor,
                hemolysis_predictor=mock_hemolysis_predictor,
            )

    def test_invalid_seed_sequence_raises(self) -> None:
        with pytest.raises(ValueError):
            run_ga(
                seed_sequence="ACDEFGHIKLXNPQRST",  # X is not a standard amino acid
                min_length=6,
                max_length=40,
                pop_size=2,
                n_generations=1,
                mutation_rate=0.1,
                random_seed=42,
                amp_predictor=mock_amp_predictor,
                hemolysis_predictor=mock_hemolysis_predictor,
            )

    def test_pop_size_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="pop_size"):
            run_ga(
                seed_sequence=SEED_SEQUENCE,
                min_length=6,
                max_length=40,
                pop_size=1,
                n_generations=1,
                mutation_rate=0.1,
                random_seed=42,
                amp_predictor=mock_amp_predictor,
                hemolysis_predictor=mock_hemolysis_predictor,
            )

    def test_tournsize_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="tournsize"):
            run_ga(
                seed_sequence=SEED_SEQUENCE,
                min_length=6,
                max_length=40,
                pop_size=6,
                n_generations=1,
                mutation_rate=0.1,
                random_seed=42,
                amp_predictor=mock_amp_predictor,
                hemolysis_predictor=mock_hemolysis_predictor,
                tournsize=1,
            )

    def test_population_can_shrink_below_pop_size(self) -> None:
        # High mutation rate makes hard_constrained_offspring more likely to
        # exhaust its retries and return None, which run_ga drops instead of
        # backfilling -- population/offsprings are allowed to end up smaller
        # than pop_size (pop_size is an attempt budget, not a guaranteed
        # survivor count).
        result = run_ga(
            seed_sequence=SEED_SEQUENCE,
            min_length=6,
            max_length=40,
            pop_size=10,
            n_generations=3,
            mutation_rate=0.95,
            random_seed=99,
            amp_predictor=mock_amp_predictor,
            hemolysis_predictor=mock_hemolysis_predictor,
        )
        # No strict assertion on size (stochastic); just confirm it never
        # exceeds pop_size and the run completes without backfill logic.
        assert len(result["population"]) <= 10


class TestRouteA:
    def test_missing_seed_sequence_raises(self) -> None:
        route = RouteA({"seed_candidates": [{}]})
        with pytest.raises(ValueError, match="seed_sequence"):
            route.run()

    def test_run_returns_candidate_per_population_member(self) -> None:
        """RouteA.run() converts every DEAP individual in the final population
        to a Candidate (not just one row) — variant_id/sequence/generation from
        the individual's own attributes, amp_prob/hemolysis_phc50 looked up
        from `history` by variant_id.

        Note: RouteA.run() calls run_ga() without predictor overrides, so this
        hits the real (GPU-loading) AMP/hemolysis models -- slow, and requires
        the real model_store weights and dependencies (torch, transformers,
        xgboost, modlamp) installed.
        """
        pop_size = 2
        route = RouteA({
            "seed_candidates": [
                {
                    "seed_sequence": SEED_SEQUENCE,
                    "min_length": 6,
                    "max_length": 40,
                    "pop_size": pop_size,
                    "n_generations": 1,
                }
            ]
        })
        candidates = route.run()

        assert len(candidates) == pop_size
        for candidate in candidates:
            assert candidate.sequence
            assert candidate.predictions["route"] == "A"
            assert 0.0 <= candidate.predictions["amp_prob"] <= 1.0
            assert math.isfinite(candidate.predictions["hemolysis_phc50"])
            assert candidate.predictions["generation"] >= 0


class FakeProtGPT2Generator:
    """Stand-in for ProtGPT2Generator matching its generate() signature, so a
    call-signature mismatch against the real class would surface as a
    TypeError here rather than being silently swallowed by a Mock()."""

    def __init__(self, sequences: list[str]) -> None:
        self.sequences = sequences
        self.calls: list[dict] = []

    def generate(
        self,
        n_peptides: int,
        tags: list[str],
        min_length: int,
        max_length: int,
        max_new_tokens: int,
        batch_size: int,
        max_attempts: int,
    ) -> list[str]:
        self.calls.append({
            "n_peptides": n_peptides,
            "tags": tags,
            "min_length": min_length,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "max_attempts": max_attempts,
        })
        return self.sequences[:n_peptides]


class TestRouteB:
    def test_returns_one_candidate_per_generated_sequence(self) -> None:
        generator = FakeProtGPT2Generator([SEED_SEQUENCE, "ACDEFGHIKLMNPQRS"])
        route = RouteB({"generator": generator, "n_peptides": 2})

        candidates = route.run()

        assert [c.sequence for c in candidates] == [SEED_SEQUENCE, "ACDEFGHIKLMNPQRS"]
        for candidate in candidates:
            assert set(candidate.predictions.keys()) >= {
                "route", "generation_tags", "physicochemical_passed", "physicochemical_fail_reasons",
            }
            assert candidate.predictions["route"] == "B"

    def test_config_is_forwarded_to_generator(self) -> None:
        generator = FakeProtGPT2Generator([SEED_SEQUENCE])
        route = RouteB({
            "generator": generator,
            "tags": ["<AMP>", "<ANTIBIOFILM>"],
            "n_peptides": 5,
            "min_length": 10,
            "max_length": 30,
            "max_new_tokens": 64,
            "batch_size": 8,
            "max_attempts": 3,
        })

        route.run()

        assert generator.calls == [{
            "n_peptides": 5,
            "tags": ["<AMP>", "<ANTIBIOFILM>"],
            "min_length": 10,
            "max_length": 30,
            "max_new_tokens": 64,
            "batch_size": 8,
            "max_attempts": 3,
        }]

    def test_defaults_used_when_config_omitted(self) -> None:
        generator = FakeProtGPT2Generator([SEED_SEQUENCE])
        route = RouteB({"generator": generator})

        route.run()

        call = generator.calls[0]
        assert call["tags"] == ["<AMP>"]
        assert call["n_peptides"] == 100
        assert call["max_new_tokens"] == 120
        assert call["batch_size"] == 16
        assert call["max_attempts"] == 20

    def test_underfilled_generation_does_not_raise(self) -> None:
        # Fake generator returns fewer sequences than requested, mirroring the
        # real generator's behavior when it exhausts max_attempts early.
        generator = FakeProtGPT2Generator([SEED_SEQUENCE])
        route = RouteB({"generator": generator, "n_peptides": 10})

        candidates = route.run()

        assert len(candidates) == 1

    def test_variant_id_is_deterministic(self) -> None:
        generator = FakeProtGPT2Generator([SEED_SEQUENCE])
        route = RouteB({"generator": generator, "tags": ["<AMP>"], "n_peptides": 1})

        first_run = route.run()
        second_run = route.run()

        assert first_run[0].id == second_run[0].id

    def test_empty_generation_returns_empty_list(self) -> None:
        generator = FakeProtGPT2Generator([])
        route = RouteB({"generator": generator, "n_peptides": 5})

        assert route.run() == []
