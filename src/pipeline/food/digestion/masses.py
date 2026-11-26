"""Alias of `common.food_digestion.masses`: the engine lives in `common` so the API can use it without the pipeline package."""
import sys

from common.food_digestion import masses as _module

sys.modules[__name__] = _module
