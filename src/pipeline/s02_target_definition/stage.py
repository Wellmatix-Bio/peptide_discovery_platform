# Stage 2: Wound Biology and Targets - defines molecular targets per indication.
from __future__ import annotations

from typing import Any

import yaml

from pipeline.base import RunContext, SetupStage, StageError
from schemas.objectives import ObjectiveVector
from schemas.run_config import StageConfig

EVIDENCE_WEIGHT = {"strong": 0.9, "moderate": 0.6, "weak": 0.3}


class Stage2(SetupStage):
    name = "s02_wound_biology_and_targets"

    def load_deficit_rules(self, config: dict) -> dict:
        """Load deficit rules from the config file."""
        rules_path = config.get("deficit_rules_path")
        if not rules_path:
            raise StageError(
                self.name, "deficit_rules_path is not specified in the config."
            )

        try:
            with open(rules_path, "r") as f:
                rules = yaml.safe_load(f)
                return rules
        except Exception as e:
            raise StageError(
                self.name, f"Failed to load deficit rules from {rules_path}: {e}"
            ) from e

    def condition_met(self, contribution: dict, context: dict) -> bool:
        """Check if the condition for a contribution is met based on the wound context."""
        op = contribution["op"]
        field_values = context.get(contribution.get("field"), [])

        if op == "contains":
            return contribution["value"] in field_values
        if op == "contains_any":
            return any(v in field_values for v in contribution["value"])
        if op == "nonempty":
            return len(field_values) > 0
        raise StageError(self.name, f"Unknown op: {op}")

    def describe(self, contribution):
        """Described the main drivers of a deficit, for human-readable output."""
        op = contribution["op"]
        field = contribution["field"]
        if op == "nonempty":
            cond = f"{field} is non-empty"
        elif op == "contains_any":
            cond = f"{field} contains any of {contribution['value']}"
        else:
            cond = f"{field} contains '{contribution['value']}'"
        return f"{cond} (+{contribution['delta']:.2f}, {contribution['evidence']})"

    def run(self, config: StageConfig, ctx: RunContext) -> ObjectiveVector:
        """Run the stage to evaluate deficits based on the provided rules and context."""
        # Load deficit rules

        if not ctx.brief:
            raise StageError(
                self.name, "Brief context is missing. Ensure Stage 1 has been executed."
            )
        deficit_rules = self.load_deficit_rules(config.params)

        results = {}
        brief_context = ctx.brief.model_dump()

        for deficit_name, spec in deficit_rules["deficits"].items():
            severity = spec["base"]
            fired = []

            for contribution in spec.get("contributions", []):
                if self.condition_met(contribution, brief_context):
                    severity += contribution["delta"]
                    fired.append(contribution)

            severity = min(severity, spec["cap"])

            results[deficit_name] = {
                "severity": round(severity, 2),
                "drivers": [self.describe(c) for c in fired],
            }

        return ObjectiveVector(deficits=results)
