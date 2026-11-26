"""Alias of `common.food_digestion.engine`: the engine lives in `common` so the API can use it without the pipeline package."""
import sys

from common.food_digestion import engine as _module

sys.modules[__name__] = _module
