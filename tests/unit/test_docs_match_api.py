"""The documentation follows the API: every food route, and every pipeline's image setting, is written in API_SPECIFICATION.md; the README names the food pipeline's image setting."""

from __future__ import annotations

import re
from pathlib import Path

from backend.api_e2e import api

ROOT = Path(__file__).resolve().parents[2]
SPEC = (ROOT / "API_SPECIFICATION.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
FOOD_PREFIXES = ("/api/v1/jobs/food", "/api/v1/food")


def documented(path: str) -> str:
    return re.sub(r"\{(\w+):path\}", r"{\1}", path)  # the spec writes {job_id}, FastAPI {job_id:path}


def test_every_food_route_is_in_the_api_specification_with_its_method():
    missing = []
    for route in api.app.routes:
        path = getattr(route, "path", "")
        if path.startswith(FOOD_PREFIXES):
            for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                if f"`{method} {documented(path)}" not in SPEC:
                    missing.append(f"{method} {path}")
    assert not missing, f"add to API_SPECIFICATION.md: {missing}"


def test_the_food_routes_found_are_the_ones_the_pipeline_has():
    found = {documented(r.path) for r in api.app.routes if getattr(r, "path", "").startswith(FOOD_PREFIXES)}
    assert {"/api/v1/jobs/food/create", "/api/v1/jobs/food/{job_id}/cancel", "/api/v1/jobs/food/{job_id}/results", "/api/v1/food/digestion/run", "/api/v1/food/library/proteins"} <= found and len(found) >= 10


def test_every_pipelines_image_setting_is_in_the_specification():
    for settings in api.PIPELINE_SETTINGS.values():
        for name in settings:
            assert f"`{name}`" in SPEC, name


def test_the_readme_names_the_food_pipeline_and_its_image_setting():
    assert "WORKER_FOOD_IMAGE_URI" in README and "/api/v1/jobs/food/create" in README and "FOOD_DEVELOPMENT_PLAN" in README
