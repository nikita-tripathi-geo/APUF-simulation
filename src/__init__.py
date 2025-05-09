# src/apuf/__init__.py

"""
apuf - a library for simulating Arbiter PUFs and other delay-based PUFs.
"""

# Package version (keep in sync with pyproject.toml or bump manually)
__version__ = "0.1.0"

# Core classes / functions exposed at the top level
from .apuf import APUF, XORPUF
from .challenges import generate_k_challenges, generate_n_k_challenges

# Define what’s exported when someone does: from apuf import *
__all__ = [
    "APUF",
    "XORPUF",
    "generate_k_challenges",
    "generate_n_k_challenges",
    "__version__",
]
