# APUF

[![PyPI version](https://img.shields.io/pypi/v/apuf.svg)](https://pypi.org/project/apuf) 
[![CI coverage](https://github.com/nikita-tripathi-geo/APUF-simulation/actions/workflows/package-ci.yml/badge.svg)](https://github.com/nikita-tripathi-geo/APUF-simulation/actions/workflows/package-ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


A Python package for simulating delay-based Physical Unclonable Functions (PUFs) using Lim's Linear Additive Delay Model (LADM).
This module supports Arbiter PUFs (APUFs) and XORPUFs.
Provides challenge generation, response simulation (with noise), and bit-packing utilities.

## Features

- **$d$-layer Arbiter PUF** (`APUF`): simulate $d$ layers, compute $k$-bit responses given $k$ challenges ($d$-bit binary vectors).
- **XORPUF** (`XORPUF`): simulate multiple APUF instances, compute responses and XOR them for increased complexity.
- **Challenge generators**:  
  - `generate_k_challenges(k, d, seed)`   
  - `generate_n_k_challenges(n, k, d, seed)`  
  - `generate_challenges_mp(n, k, d, seed, processes)` for parallel batch generation.  
- **Response data-type** (`Response`): Both `APUF` and `XORPUF` return a $k$-bit response given $k$ challenges. `Response` provides a conventient interface to study PUF challenge-response behaviours.


## Installation

```bash
pip install apuf
```

> *Requires Python 3.9+ and **`numpy`**.*

## Quick Start

Import the module and call the functions directly:

```python
from apuf import APUF, XORPUF, generate_k_challenges

# 1) Single APUF
apuf = APUF(d=64, mean=0.0, std=0.05)

# 2) Generate 10 random challenges (phase vectors)
chals = generate_k_challenges(k=10, d=64, seed=42)  # shape (65, 10)

# 3) Measure responses twice with Gaussian noise (mean=0, std=0.005)
resp1 = apuf.get_responses(chals, nmean=0.0, nstd=0.005)
resp2 = apuf.get_responses(chals, nmean=0.0, nstd=0.005)

# 4) Due to measurement noise, resp1 and resp2 may be different
diff = resp1 - resp2
#   4.1) We can compute Hamming distance
hamming_distance = resp1.dist(resp2)
#   4.2) And fractional/normalized HD
fHD = resp1.dist(resp2)/len(resp1)

# 5) Pack responses into bytes for convenience
resp_bytes = APUF.compact_responses(resp)
print(resp, resp_bytes)

```

## API Reference
Below you will find a short reference for the core classes and functions.

### Classes

`Response` - Represents a PUF response vector.

+ Bitwise operations: `^`, `&`, `|`, `~`
+ `.dist(other)` returns Hamming distance.
+ `.hw` returns Hamming weight (number of 1s).

`APUF(d: int = 128, mean: float = 0.0, std: float = 0.05)` - Simulates a single APUF.

+ `get_responses(chals: np.ndarray, nmean: float = 0.0, nstd: float = 0.005) -> Response`
+ `compact_responses(resp: Response) -> bytes`

`XORPUF(children: list[APUF])` - Simulates an XORPUF by XORing responses of child APUFs.

`get_responses(chals: np.ndarray) -> Response`


### Functions
+ `generate_k_challenges(k: int, d: int, seed: int = None) -> np.ndarray`
+ `generate_n_k_challenges(n: int, k: int, d: int, seed: int = None) -> np.ndarray`
+ `generate_challenges_mp(n: int, k: int, d: int, seed: int = None, processes: int = cpu_count()) -> list[np.ndarray]`

Full docstrings and type hints are available in code.


## Testing

Run the provided unit tests with `unittest`:

```bash
git clone https://github.com/nikita-tripathi-geo/APUF-simulation.git
cd APUF-simulation
pip install -r requirements.txt
python -m unittest tests/*.py
```

## Contributing

Contributions, issues and feature requests are welcome!
Please see [Issues](https://github.com/nikita-tripathi-geo/APUF-simulation/issues).

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

MIT License (c) Nikita Tripathi
