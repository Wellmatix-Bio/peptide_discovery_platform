"""Plug-in slots for the screening models that are not wired yet (WP E).

Each model in `models_config.json` is either a stub (no class given: it does not run and the run reports it as
not run) or a plug-in (class and weights given: it runs and its scores feed the objective it belongs to).

Contract for a plug-in class:
    Class(weights=Path)                       # weights path from the config
    .predict(seqs: list[str]) -> {sequence: {"score": float in [0, 1],      # higher = safer / more developable
                                             "label": str,
                                             "applicability": "in_domain" | "limited" | "out_of_domain",
                                             "detail": str}}
Objectives: models with objective "safety" combine as the worst (minimum) score; "developability" models combine
with the rule-based score the same way. Nothing is averaged, so one bad score is not hidden by good ones.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


APPLICABILITY_ORDER = {"in_domain": 0, "limited": 1, "out_of_domain": 2}


class ContractError(ValueError):
    """A plug-in returned something outside the contract (score outside 0-1, unknown applicability); this is a bug, not a missing model."""


class ScreeningModel:
    def __init__(self, name: str, spec: dict):
        self.name, self.objective = name, spec["objective"]
        self.description, self.source, self.notes = spec.get("description", ""), spec.get("source", ""), spec.get("notes", "")
        self.status = "not_configured"
        self.detail = "stub: no class and weights given in the model config"
        self.impl = None

    def load(self, spec: dict) -> None:
        cls, weights = (spec.get("class") or "").strip(), (spec.get("weights") or "").strip()
        if not cls:
            return
        try:
            if spec.get("sys_path"):
                sys.path.insert(0, spec["sys_path"])
            mod, _, attr = cls.partition(":")
            if not attr:
                raise ValueError("class must look like 'package.module:ClassName'")
            if not weights or not Path(weights).exists():
                raise FileNotFoundError(f"weights not found: {weights or '(empty)'}")
            impl = getattr(importlib.import_module(mod), attr)(weights=Path(weights))
            if not callable(getattr(impl, "predict", None)):
                raise TypeError("class has no predict(seqs)")
            self.impl, self.status, self.detail = impl, "ready", f"{cls} with {weights}"
        except Exception as exc:                          # a broken plug-in must not stop the pipeline
            self.status, self.detail = "error", f"{type(exc).__name__}: {exc}"

    def load_with(self, factory) -> None:
        """Make the slot ready from a zero-argument factory; a factory that fails (missing weights or dependency) leaves it "error" with the reason."""
        try:
            impl = factory()
            if not callable(getattr(impl, "predict", None)):
                raise TypeError("class has no predict(seqs)")
            self.impl, self.status, self.detail = impl, "ready", type(impl).__name__
        except Exception as exc:  # a broken plug-in must not stop the pipeline
            self.status, self.detail = "error", f"{type(exc).__name__}: {exc}"

    def predict(self, seqs: list[str]) -> dict[str, dict]:
        if self.status != "ready":
            return {}
        out = self.impl.predict(seqs)
        bad = [s for s, r in out.items() if not (0.0 <= float(r["score"]) <= 1.0) or r.get("applicability") not in APPLICABILITY_ORDER]
        if bad:
            raise ContractError(f"model '{self.name}' broke the contract for {len(bad)} peptides (score outside 0-1 or bad applicability)")
        return out


def load_models(config: dict) -> list[ScreeningModel]:
    """`config` is the "models" mapping: name -> spec (objective, class, weights, ...)."""
    models = []
    for name, spec in config.items():
        m = ScreeningModel(name, spec)
        m.load(spec)
        models.append(m)
    return models


def run_models(models: list[ScreeningModel], seqs: list[str], log=print) -> dict[str, dict[str, dict]]:
    """{model name: {sequence: result}} for the models that are ready."""
    out = {}
    for m in models:
        if m.status == "ready":
            log(f"  model '{m.name}' ({m.objective}) on {len(seqs)} peptides")
            try:
                out[m.name] = m.predict(seqs)
            except ContractError:
                raise
            except Exception as exc:  # e.g. weights missing at predict time: not run, never a pass
                m.status, m.detail = "error", f"{type(exc).__name__}: {exc}"
                log(f"  model '{m.name}' failed and is reported as not run: {m.detail}")
        elif m.status == "error":
            log(f"  model '{m.name}' skipped: {m.detail}")
    return out


def status_table(models: list[ScreeningModel]) -> dict[str, dict]:
    return {m.name: {"objective": m.objective, "status": m.status, "detail": m.detail} for m in models}
