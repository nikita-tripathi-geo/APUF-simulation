# src/apuf/__init__.py

"""
apuf - a library for simulating Arbiter PUFs and estimating response entropy.
"""

# Package version (keep in sync with pyproject.toml or bump manually)
__version__ = "0.1.0"

# Core classes / functions exposed at the top level
from .apuf import APUF
from .challenges import generate_k_challenges, generate_n_k_challenges, generate_challenges_mp
# from .multiproc import generate_challenges_mp
# from .utils import parity

# Define what’s exported when someone does:
#   from apuf import *
__all__ = [
    "APUF",
    "generate_k_challenges",
    "generate_n_k_challenges",
    # "generate_challenges_mp",
    # "parity",
    "__version__",
]
