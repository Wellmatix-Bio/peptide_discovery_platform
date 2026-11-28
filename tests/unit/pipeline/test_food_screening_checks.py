"""Food screening and ranking rules, ported from Phase 1 `7. AI Screening and Ranking/test_wpe.py`: model slots, Pareto pick, coverage, applicability."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from food.c_discovery import settings  # noqa: E402
from food.c_discovery.ranking import ranking  # noqa: E402
from food.c_discovery.screening import checks, model_registry  # noqa: E402

CONFIG = {name: {"objective": "safety", "class": "", "weights": ""} for name in ("allergenicity", "cytotoxicity", "solubility")}


def spec(cls: str, weights: str, tmp: Path) -> dict:
    return {"allergenicity": {"objective": "safety", "class": cls, "weights": weights, "sys_path": str(tmp)}}


def test_a_slot_with_no_class_is_a_stub_that_runs_nothing():
    models = model_registry.load_models(CONFIG)
    assert {m.name for m in models} == set(CONFIG)
    assert all(m.status == "not_configured" and m.predict(["LKPT"]) == {} for m in models)


def test_a_plugin_must_keep_the_contract(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"x")
    (tmp_path / "dummy_food_model.py").write_text(
        "class Good:\n  def __init__(self, weights): pass\n"
        "  def predict(self, seqs): return {s: {'score': 0.25, 'label': 'x', 'applicability': 'limited', 'detail': ''} for s in seqs}\n"
        "class Bad:\n  def __init__(self, weights): pass\n"
        "  def predict(self, seqs): return {s: {'score': 1.5, 'label': 'x', 'applicability': 'limited', 'detail': ''} for s in seqs}\n",
        encoding="utf-8",
    )
    (good,) = model_registry.load_models(spec("dummy_food_model:Good", str(tmp_path / "w.bin"), tmp_path))
    assert good.status == "ready" and good.predict(["LKPT"])["LKPT"]["score"] == 0.25
    (missing,) = model_registry.load_models(spec("dummy_food_model:Good", str(tmp_path / "nope.bin"), tmp_path))
    assert missing.status == "error" and "weights not found" in missing.detail  # a broken plug-in does not stop the pipeline
    (bad,) = model_registry.load_models(spec("dummy_food_model:Bad", str(tmp_path / "w.bin"), tmp_path))
    try:
        bad.predict(["LKPT"])
        raise AssertionError("contract violation was not caught")
    except ValueError:
        pass


def toy_pool(n=60):
    rng = np.random.default_rng(1)
    return [
        {"sequence": f"PEP{i:03d}", "evidence": float(rng.random()), "coverage": 0.6, "digestion_plausibility": float(rng.random()),
         "developability_rule": float(rng.random()), "model_allergenicity_score": float(rng.random()),
         "model_cytotoxicity_score": float(rng.random()), "evidence_gate_passed": bool(i % 2)}
        for i in range(n)
    ]


def test_pareto_pick_is_deterministic_and_respects_caps_and_gate():
    pool = toy_pool()
    objs = ["digestion_plausibility", "coverage", "developability_rule", "model_allergenicity_score", "model_cytotoxicity_score", "evidence"]
    clusters, parents = np.arange(len(pool)) // 3, [f"p{i // 5}" for i in range(len(pool))]
    a = ranking.pareto_pick_selection(pool, objs, clusters, parents, 10, 5)
    rp = list(reversed(pool))
    b = ranking.pareto_pick_selection(rp, objs, clusters[::-1], parents[::-1], 10, 5)
    assert [pool[i]["sequence"] for i in a[0]] == [rp[i]["sequence"] for i in b[0]]  # same list whatever the input order
    assert len(a[0]) == 10 and len(set(a[0])) == 10
    cl, pa = [clusters[i] for i in a[0]], [parents[i] for i in a[0]]
    assert max(cl.count(x) for x in set(cl)) <= settings.MAX_PER_CLUSTER and max(pa.count(x) for x in set(pa)) <= settings.MAX_PER_PARENT_CLUSTER
    assert a[1]["exact_evidence_in_final"] >= 5 or a[1]["shortfall"] > 0
    dropped = set(a[1]["dropped_for_diversity"] + a[1]["dropped_for_evidence_gate"])
    assert not dropped & {pool[i]["sequence"] for i in a[0]}  # dropped peptides never come back


def test_coverage_adds_and_removes():
    assert checks.coverage({"a": "in_domain", "b": "in_domain"}) == 1.0
    assert checks.coverage({"a": "in_domain", "b": "limited"}) == 0.75
    assert checks.coverage({"a": "in_domain", "b": "out_of_domain", "c": "not_run", "d": "limited"}) == 0.375  # (1 + 0 + 0 + .5) / 4
    assert checks.coverage({"a": "limited", "b": "limited"}) < checks.coverage({"a": "in_domain", "b": "limited"})


def test_an_out_of_domain_model_does_not_veto():
    v, basis, flagged = checks.combine_objective([("a", 0.8, "in_domain"), ("b", 0.0, "out_of_domain")])
    assert v == 0.8 and basis == "in_domain" and flagged == "b:out_of_domain"
    v, basis, _ = checks.combine_objective([("a", 0.8, "in_domain"), ("b", 0.3, "limited")])
    assert v == 0.3 and basis == "limited"  # limited models still count
    v, basis, _ = checks.combine_objective([("b", 0.1, "out_of_domain")])
    assert v == 0.1 and basis == "out_of_domain_only"  # nothing qualified: all scores used, and it says so
    v, basis, _ = checks.combine_objective([("b", 0.1, "out_of_domain")], base=0.5)
    assert v == 0.1 and basis == "out_of_domain_only"
    v, _, _ = checks.combine_objective([("a", 0.9, "in_domain")], base=0.5)
    assert v == 0.5
    assert checks.combine_objective([])[1] == "none"


def test_applicability_is_worst_case_and_never_hidden():
    assert checks.worst(["in_domain", "limited", "in_domain"]) == "limited"
    assert checks.worst(["in_domain", "out_of_domain"]) == "out_of_domain"
    assert checks.worst(["not_run"]) == "not_run"
    assert checks.evidence_applicability({"evidence_match": "none", "evidence_similarity": 0.3}) == "out_of_domain"
    assert checks.evidence_applicability({"evidence_match": "near", "evidence_similarity": 0.8}) == "in_domain"
    assert checks.stability_applicability({"length": 8, "uncut_bond_fraction": 0.5}) == "limited"
    assert checks.stability_applicability({"length": 8, "uncut_bond_fraction": float("nan")}) == "not_run"


def test_selection_gate_and_caps_and_the_shortfall_is_reported():
    n = 60
    Y = np.random.default_rng(0).random((n, 3))
    front = ranking.pareto_fronts(Y)
    crowd = ranking.crowding_distance(Y, front)
    direct = np.array([i % 3 == 0 for i in range(n)])  # a third have direct evidence
    clusters, parents = np.arange(n) // 2, [str(i % 12) for i in range(n)]
    chosen, gate = ranking.select_final(front, crowd, clusters, parents, direct, 25, 10)
    assert len(chosen) == len(set(chosen)) == 25
    assert sum(direct[i] for i in chosen) >= 10 and gate["shortfall"] == 0
    assert max(np.bincount(clusters[chosen])) <= settings.MAX_PER_CLUSTER
    assert max(Counter(parents[i] for i in chosen).values()) <= settings.MAX_PER_PARENT_CLUSTER
    _, gate2 = ranking.select_final(front, crowd, clusters, parents, np.zeros(n, bool), 25, 10)
    assert gate2["shortfall"] == 10


def test_a_model_that_fails_at_predict_time_is_reported_not_run_but_a_contract_break_is_not_swallowed():
    class Missing:
        def predict(self, seqs):
            raise FileNotFoundError("weights not found")

    class Broken:
        def predict(self, seqs):
            return {s: {"score": 2.0, "label": "x", "applicability": "limited", "detail": ""} for s in seqs}

    a, b = model_registry.ScreeningModel("a", {"objective": "safety"}), model_registry.ScreeningModel("b", {"objective": "safety"})
    a.load_with(lambda: Missing())
    b.load_with(lambda: Broken())
    assert model_registry.run_models([a], ["LKPT"], lambda m: None) == {}
    assert a.status == "error" and "FileNotFoundError" in a.detail
    try:
        model_registry.run_models([b], ["LKPT"], lambda m: None)
        raise AssertionError("contract break was swallowed")
    except model_registry.ContractError:
        pass
