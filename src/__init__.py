# src/apuf/__init__.py

"""
apuf - a library for simulating Arbiter PUFs and other delay-based PUFs.
"""

# Package version (keep in sync with pyproject.toml or bump manually)
__version__ = "0.2.0"

# Core classes / functions exposed at the top level
from .apuf import Response, APUF, XORPUF
from .challenges import generate_k_challenges, generate_n_k_challenges

# What’s exported with: `from apuf import *`
__all__ = [
    "Response",
    "APUF",
    "XORPUF",
    "generate_k_challenges",
    "generate_n_k_challenges",
    "__version__",
]
