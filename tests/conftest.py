"""Shared test setup."""
import os

# common/env.py loads the developer's .env with setdefault, so an empty value here wins over it: the food tests use the library bundled with the code, never the gs:// folder of a real deployment.
os.environ["FOOD_LIBRARY_DIR"] = ""
