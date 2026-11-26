"""Alias of `common.food_digestion.rules`: the engine lives in `common` so the API can use it without the pipeline package."""
import sys

from common.food_digestion import rules as _module

sys.modules[__name__] = _module
