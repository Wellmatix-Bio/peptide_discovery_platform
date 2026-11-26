"""Digestion settings, and validation (stages, pathways).

Model: a *pathway* is an ordered list of *stages* (e.g. gastric -> intestinal), each
applying one or more enzymes. See digest.py for how fragments are enumerated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .rules import Enzyme, build_registry



ORIGIN_TYPE = "DIGESTION_DERIVED"
EVIDENCE_CLASS = "THEORETICAL"


@dataclass(frozen=True)
class Stage:
    name: str
    enzymes: tuple[str, ...]
    ph: float | None = None


@dataclass(frozen=True)
class Pathway:
    id: str
    stages: tuple[Stage, ...]


@dataclass(frozen=True)
class Settings:
    name: str
    pathways: tuple[Pathway, ...]
    max_missed_cleavages: int = 2
    min_length: int = 2
    max_length: int = 40
    custom_enzymes: tuple[tuple[str, str], ...] = ()   # canonical JSON of custom rules

    @staticmethod
    def from_dict(d: dict) -> "Settings":
        try:
            pathways = tuple(
                Pathway(p["id"], tuple(Stage(s["name"], tuple(s["enzymes"]), s.get("ph"))
                                       for s in p["stages"]))
                for p in d["pathways"])
        except KeyError as exc:
            raise ValueError(f"settings are missing required key {exc}; see configs/digestion_default.json")
        s = Settings(
            name=d.get("name", "unnamed"),
            pathways=pathways,
            max_missed_cleavages=int(d.get("max_missed_cleavages", 2)),
            min_length=int(d.get("min_length", 2)),
            max_length=int(d.get("max_length", 40)),
            custom_enzymes=tuple(sorted((k, json.dumps(v, sort_keys=True))
                                        for k, v in (d.get("custom_enzymes") or {}).items())),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if not self.pathways:
            raise ValueError("settings define no pathways")
        if self.max_missed_cleavages < 0:
            raise ValueError("max_missed_cleavages must be >= 0")
        if not (1 <= self.min_length <= self.max_length):
            raise ValueError(f"need 1 <= min_length <= max_length (got {self.min_length}, {self.max_length})")
        ids = [p.id for p in self.pathways]
        if len(set(ids)) != len(ids):
            raise ValueError("pathway ids must be unique")
        reg = self.registry()
        for p in self.pathways:
            if not p.stages:
                raise ValueError(f"pathway '{p.id}' has no stages")
            for st in p.stages:
                if not st.enzymes:
                    raise ValueError(f"pathway '{p.id}' stage '{st.name}' has no enzymes")
                for e in st.enzymes:
                    if e not in reg:
                        raise ValueError(f"pathway '{p.id}': unknown enzyme '{e}'. "
                                         f"Available: {sorted(reg)}")

    def registry(self) -> dict[str, Enzyme]:
        return build_registry({k: json.loads(v) for k, v in self.custom_enzymes})

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "max_missed_cleavages": self.max_missed_cleavages,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pathways": [{"id": p.id, "stages": [
                {"name": s.name, "enzymes": list(s.enzymes), "ph": s.ph} for s in p.stages]}
                for p in self.pathways],
            "custom_enzymes": {k: json.loads(v) for k, v in self.custom_enzymes},
        }

    def ph_warnings(self) -> list[str]:
        reg = self.registry()
        out = []
        for p in self.pathways:
            for st in p.stages:
                if st.ph is None:
                    continue
                for e in st.enzymes:
                    lo, hi = reg[e].ph_range
                    if not (lo <= st.ph <= hi):
                        out.append(f"pathway '{p.id}' stage '{st.name}': pH {st.ph} is outside the "
                                   f"{lo}-{hi} range of rule '{e}'; the rule may not apply")
        return out



def _pw(pid, *stages):
    return Pathway(pid, tuple(Stage(n, tuple(e), ph) for n, e, ph in stages))


# The fixed settings for this project's digestion run.
DEFAULT_SETTINGS = Settings(
    name="default_gi_v1",
    max_missed_cleavages=2,
    min_length=2,
    max_length=40,
    pathways=(
        _pw("pepsin_only", ("gastric", ["pepsin_ph1.3"], 1.3)),
        _pw("trypsin_only", ("intestinal", ["trypsin"], 7.5)),
        _pw("chymotrypsin_only", ("intestinal", ["chymotrypsin_high"], 7.5)),
        _pw("trypsin_chymotrypsin", ("intestinal", ["trypsin", "chymotrypsin_high"], 7.5)),
        _pw("gastric_to_intestinal",
            ("gastric", ["pepsin_ph1.3"], 1.3),
            ("intestinal", ["trypsin", "chymotrypsin_high"], 7.5)),
    ),
)
DEFAULT_SETTINGS.validate()
