"""Utility functions for generating APUF challenges,
with optional parallelization via multiprocessing.

This module provides:
  - generate_k_challenges(k, d, seed)
  - generate_n_k_challenges(n, k, d, seed)
  - generate_challenges_mp(n, k, d, seed, processes)
"""

from typing import Optional
from multiprocessing import Pool, cpu_count
import numpy as np


def generate_k_challenges(
    k: int = 1, d: int = 128, seed: Optional[int] = None
) -> np.ndarray:
    """Generate `k` random challenges and map them to LADM phase vectors.

    Each challenge is a binary vector of length `d`. This function
    maps random challenges to delay-based phase vectors.

    Args:
        k (int): Number of challenges.
        d (int): Length of each challenge (number of bits).
        seed (int, optional): PRNG seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: A phase matrix phi of shape (d, k), where each
            column corresponds to the transformed challenge.
    """
    # Input validation
    assert isinstance(k, int) and k > 0, "k must be a positive integer"
    assert isinstance(d, int) and d > 0, "d must be a positive integer"
    if seed:
        assert isinstance(seed, int), "seed must be an integer or None"

    # Adjust the number of layers
    d = d + 1

    # Generate binary challenges
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
        chals = rng.integers(0, 2, (k, d))
    else:
        chals = np.random.randint(0, 2, (k, d))

    # Map {0,1} -> {+1,-1}
    chals_prime = 1 - 2 * chals

    # Convert to phi
    phi = np.ones((k, d))

    for chal in range(k):
        for i in range(d):
            for j in range(i, d):
                phi[chal][i] *= chals_prime[chal][j]

    phi = np.transpose(phi)

    # set last bit to 1 (last bit of psi has to be 1)
    phi[-1] = np.ones(k)

    return phi


def generate_n_k_challenges(
    n: int = 1, k: int = 1, d: int = 128, seed: Optional[int] = None
) -> np.ndarray:
    """Generate `n` sequences of `k` random challenges.

    Args:
        n (int): Number of challenge sequences.
        k (int): Number of challenges per sequence.
        d (int): Length of each challenge.
        seed (int, optional): PRNG seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: Array of shape `(n, d+1, k)` containing
            `n` sequences of `k` challenges.
    """
    # Input validation
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    # reuse generate_k_challenges's validation for k, d, seed
    assert isinstance(k, int) and k > 0, "k must be a positive integer"
    assert isinstance(d, int) and d > 0, "d must be a positive integer"
    if seed:
        assert isinstance(seed, int), "seed must be an integer or None"

    # Adjust the number of layers
    d = d + 1

    # Generate binary challenges
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
        chals = rng.integers(0, 2, (n, k, d))
    else:
        chals = np.random.randint(0, 2, (n, k, d))

    chals_prime = 1 - 2 * chals

    phi = np.ones((n, k, d))

    for locker in range(n):
        for chal in range(k):
            for i in range(d):
                for j in range(i, d):
                    phi[locker][chal][i] *= chals_prime[locker][chal][j]

    result = np.transpose(phi, (0, 2, 1))
    for i in range(n):
        result[i][-1] = np.ones(k)

    return result


def unit_generation(params: tuple[int, int, int]) -> np.ndarray:
    """Helper for multiprocessing: generate one batch of challenges.

    Unpacks `(k, d, seed)` and calls `generate_k_challenges`.

    Args:
        params (tuple[int, int, int]):
            A triple `(k, d, seed)` for challenge generation.

    Returns:
        np.ndarray:
            Phase matrix of shape `(d+1, k)` for this unit.
    """
    k, d, seed = params
    return generate_k_challenges(k, d, seed)


def generate_challenges_mp(
    n: int = 1,
    k: int = 1,
    d: int = 128,
    seed: Optional[int] = None,
    proc: int = cpu_count(),
) -> list[np.ndarray]:
    """Generate `n` sets of `k` random challenges in parallel.

    This function spawns a pool of worker processes to generate `n`
    independent sequences of `k` challenges each, using different seeds.

    Args:
        n (int):
            Number of challenge sequences to generate.
        k (int):
            Number of challenges per sequence.
        d (int):
            Length of each challenge (number of arbiter stages).
        proc (int):
            Number of worker processes to use in the Pool. Defaults to maximum
            available cores.
        seed (int):
            Base seed; each batch uses seed + batch_index for reproducibility.

    Returns:
        np.ndarray:
            Array of shape `(n, d+1, k)`, where element [i] is the
            `i`-th sequence of `k` random challenges.
    """
    # Input validation
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    assert isinstance(k, int) and k > 0, "k must be a positive integer"
    assert isinstance(d, int) and d > 0, "d must be a positive integer"
    if seed:
        assert isinstance(seed, int), "seed must be an integer or None"
    assert isinstance(proc, int) and proc > 0, "proc must be a positive integer"

    # base is 0 if seed is None
    base = seed or 0
    params = [(k, d, s + base) for s in range(n)]

    with Pool(processes=proc) as pool:
        chals = pool.map(unit_generation, params)

    return chals
