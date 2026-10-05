# Unit tests for s11_ranking brief-driven weights (N_k accumulation).
from __future__ import annotations

import math

import pytest

from pipeline.s11_ranking.stage import (
    BUILTIN_RANKING_POLICY,
    MODULE_NAMES,
    WeightAllocator,
)
from schemas.brief import Brief

OBJECTIVES = (
    "wound_closure", "antimicrobial", "anti_inflammatory", "immunomodulation", "angiogenesis", "collagen_ecm",
)
ALWAYS_ON = ("safety", "stability", "synthesis_feasibility", "mechanistic_confidence")


def allocate(functions=(), context=(), pathogens=()):
    return WeightAllocator(BUILTIN_RANKING_POLICY).allocate(
        Brief(
            min_length=5, max_length=30,
            desired_functions=list(functions), wound_context=list(context), pathogens=list(pathogens)
        )
    )


def n_k(allocation):
    return {m: allocation[m].n_k for m in OBJECTIVES}


def test_chronic_diabetic_example():
    a = allocate(["anti_inflammatory"], ["chronic", "diabetic"])
    assert n_k(a) == {
        "wound_closure": 1.0,  # chronic + diabetic
        "antimicrobial": 0.0,
        "anti_inflammatory": 2.0,  # 1.0 reference + chronic + diabetic
        "immunomodulation": 1.0,  # chronic + diabetic (not referenced)
        "angiogenesis": 0.5,  # diabetic
        "collagen_ecm": 0.0,
    }


def test_always_on_n_k_is_the_average_of_all_six_objective_n_k():
    a = allocate(["anti_inflammatory"], ["chronic", "diabetic"])
    average = (1.0 + 0.0 + 2.0 + 1.0 + 0.5 + 0.0) / 6  # zeros and the placeholder count
    for module in ALWAYS_ON:
        assert a[module].n_k == pytest.approx(average)
        assert a[module].activated is True
        assert a[module].references == [] and a[module].implications == []
    total = 4.5 + 4 * average
    assert a["anti_inflammatory"].nominal_weight == pytest.approx(2.0 / total)
    assert a["safety"].nominal_weight == pytest.approx(average / total)


def test_always_on_modules_all_get_the_same_weight():
    a = allocate(["angiogenesis"], ["ischemic"])
    assert len({round(a[m].nominal_weight, 12) for m in ALWAYS_ON}) == 1


def test_anti_inflammatory_reference_is_one_point_and_immunomodulation_half():
    a = allocate(["anti_inflammatory"])
    assert a["anti_inflammatory"].n_k == 1.0
    assert a["immunomodulation"].n_k == 0.0
    b = allocate(["immunomodulation"])
    assert b["immunomodulation"].n_k == 0.5
    assert b["anti_inflammatory"].n_k == 0.0
    assert b["immunomodulation"].requested is True


def test_both_functions_reference_their_own_modules():
    a = allocate(["anti_inflammatory", "immunomodulation"])
    assert a["anti_inflammatory"].n_k == 1.0
    assert a["immunomodulation"].n_k == 0.5


def test_no_wound_context_leaves_wound_closure_out():
    a = allocate(["anti_inflammatory"])
    assert a["wound_closure"].n_k == 0.0
    assert a["wound_closure"].nominal_weight == 0.0
    assert a["safety"].n_k == pytest.approx(1 / 6)


def test_empty_brief_gives_every_module_equal_weight():
    a = allocate()
    assert all(a[m].n_k == 1.0 for m in MODULE_NAMES)
    assert {round(a[m].nominal_weight, 9) for m in MODULE_NAMES} == {round(1 / 10, 9)}
    assert all(a[m].activated for m in MODULE_NAMES)


def test_brief_with_no_mapped_entries_is_treated_as_empty():
    a = allocate(context=["high_exudate"])
    assert {round(a[m].nominal_weight, 9) for m in MODULE_NAMES} == {round(1 / 10, 9)}


def test_weights_always_sum_to_one():
    briefs = [
        dict(),
        dict(functions=["angiogenesis"], context=["ischemic"]),
        dict(functions=["anti_inflammatory"], context=["chronic", "diabetic"]),
        dict(functions=["antimicrobial"], pathogens=["Escherichia_coli"], context=["infected"]),
    ]
    for brief in briefs:
        assert math.fsum(x.nominal_weight for x in allocate(**brief).values()) == pytest.approx(1.0)


def test_pathogens_reference_antimicrobial_once():
    one = allocate(["anti_inflammatory"], pathogens=["Escherichia_coli"])
    several = allocate(["anti_inflammatory"], pathogens=["Escherichia_coli", "Staphylococcus_aureus"])
    assert one["antimicrobial"].n_k == several["antimicrobial"].n_k == 1.0
    assert several["antimicrobial"].requested is True


def test_antimicrobial_function_and_pathogens_are_one_reference():
    assert allocate(["antimicrobial"], pathogens=["Escherichia_coli"])["antimicrobial"].n_k == 1.0


def test_several_functions_for_one_module_are_one_reference():
    migration = allocate(["fibroblast_migration", "keratinocyte_migration"])
    assert migration["wound_closure"].n_k == 1.0


@pytest.mark.parametrize(
    "function, module, points",
    [
        ("angiogenesis", "angiogenesis", 1.0),
        ("anti_inflammatory", "anti_inflammatory", 1.0),
        ("immunomodulation", "immunomodulation", 0.5),
        ("antimicrobial", "antimicrobial", 1.0),
        ("cell_proliferation/migration", "wound_closure", 1.0),
        ("fibroblast_migration", "wound_closure", 1.0),
        ("keratinocyte_migration", "wound_closure", 1.0),
        ("collagen_remodeling", "collagen_ecm", 1.0),
        ("collagen_synthesis", "collagen_ecm", 1.0),
    ],
)
def test_desired_function_maps_to_module(function, module, points):
    a = allocate([function])
    assert a[module].n_k == points
    assert a[module].requested is True
    assert a[module].references == [function]


@pytest.mark.parametrize(
    "context, expected",
    [
        ("infected", {"antimicrobial", "immunomodulation"}),
        ("biofilm_positive", {"antimicrobial", "immunomodulation"}),
        ("necrotic", {"antimicrobial", "wound_closure", "immunomodulation"}),
        ("chronic", {"anti_inflammatory", "immunomodulation", "wound_closure"}),
        ("diabetic", {"anti_inflammatory", "immunomodulation", "angiogenesis", "wound_closure"}),
        ("high_glucose", {"anti_inflammatory"}),
        ("ischemic", {"angiogenesis"}),
        ("low_perfusion", {"angiogenesis"}),
        ("acute", {"wound_closure"}),
        ("surgical", {"wound_closure"}),
        ("traumatic", {"wound_closure"}),
        ("clean", {"wound_closure"}),
        ("burn", {"anti_inflammatory"}),
        ("radiation_induced", {"anti_inflammatory"}),
    ],
)
def test_wound_context_implies_modules_at_half_a_point(context, expected):
    # Reference a module the context does not touch so the brief is not "empty".
    a = allocate(["collagen_synthesis"], [context])
    for module in OBJECTIVES:
        if module == "collagen_ecm":
            continue
        assert a[module].n_k == (0.5 if module in expected else 0.0)
        assert a[module].implications == ([context] if module in expected else [])


def test_infected_implies_immunomodulation_but_not_anti_inflammatory():
    a = allocate(["collagen_synthesis"], ["infected"])
    assert a["immunomodulation"].n_k == 0.5
    assert a["anti_inflammatory"].n_k == 0.0


def test_high_glucose_implies_anti_inflammatory_only():
    a = allocate(["collagen_synthesis"], ["high_glucose"])
    assert a["anti_inflammatory"].n_k == 0.5
    assert a["immunomodulation"].n_k == 0.0


def test_context_tags_stack():
    a = allocate(["collagen_synthesis"], ["ischemic", "low_perfusion"])
    assert a["angiogenesis"].n_k == 1.0
    assert a["angiogenesis"].implications == ["ischemic", "low_perfusion"]


def test_reference_beats_implication_in_weight():
    implied = allocate(["anti_inflammatory"], ["ischemic"])
    referenced = allocate(["anti_inflammatory", "angiogenesis"])
    assert referenced["angiogenesis"].nominal_weight > implied["angiogenesis"].nominal_weight


def test_unrequested_module_has_weight_zero_and_is_not_activated():
    a = allocate(["anti_inflammatory"], ["chronic"])
    assert a["angiogenesis"].nominal_weight == 0.0
    assert a["collagen_ecm"].nominal_weight == 0.0
    assert not a["angiogenesis"].activated


def test_adding_objectives_raises_the_always_on_n_k():
    one = allocate(["anti_inflammatory"])
    many = allocate(["anti_inflammatory", "angiogenesis", "antimicrobial"])
    assert many["safety"].n_k > one["safety"].n_k


def test_every_module_is_allocated():
    assert set(allocate()) == set(MODULE_NAMES)
