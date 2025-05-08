"""Simulate an Arbiter PUF and its corresponding challenge-response behavior
using Lim's Linear Additive Delay Model (LADM).

# Usage:
```
    # As a library
    from apuf import APUF
    from challenges import generate_k_challenges

    # Initialize a 64-layer APUF, all weights from N(0,0.05)
    mypuf = APUF(d=64)

    # Generate 10 random 64-bit challenges
    chals = generate_k_challenges(10, 64)

    # Get a noisy 10-bit response
    resp = mypuf.get_noisy_responses(chals)
    print(resp)
```
"""
import numpy as np


class APUF:
    """Simulate an Arbiter Physically Unclonable Function (APUF) via LADM.

    This class samples independent biases of each layer (weights)
    and provides methods for:
      - thresholding a delay difference (float) into a bit,
      - simulating measurement noise using a Gaussian distribution
        and computing noisy responses for many challenges at once,
      - generating random challenges and mapping them to LADM phase vectors.
    """

    def __init__(self, d: int = 128, mean: float = 0.0,
                 std: float = 0.05):
        """Initialize an APUF with `d` layers and a weight distribution.

        Args:
            d (int): Number of layers of APUF. Internally `d+1` weights are used
                (one for each stage plus the arbiter). Defaults to `128`.
            weight_mean (float): Mean of the Gaussian distribution used to
                generate weights. Defaults to `0.0`.
            weight_std (float): Standard deviation of the Gaussian distribution. 
                Defaults to `0.05`.
        
        Raises:
            AssertionError: If `d` is not a positive integer, of `weight_mean`
                is not a float, or if `weight_std` is not a non-negative float.
        """
        # sanity‐checks
        assert isinstance(d, int) and d > 0, "d must be a positive integer"
        assert isinstance(mean, float), "mean must be a float"
        assert isinstance(std, float) and std >= 0, "std must be non-negative"


        # We represent a `d`-layer APUF using `d+1` weights
        self.d = d + 1
        self.weight_mean = mean
        self.weight_std = std
        self.weights = np.random.normal(
            loc=self.weight_mean,
            scale=self.weight_std,
            size=self.d)

        # Vectorized thresholding function.
        # Takes a float ndarray. Returns a byte (instead of int).
        self.__determine_responses_vectorized = np.vectorize(
            self.__determine_responses,
            "b")


    def __determine_responses(self, delay: float) -> int:
        """Threshold delay to obtain a response bit."""
        return 1 if delay > 0 else 0


    def get_noisy_responses(self, chals: np.ndarray,
                            mean: float = 0.0,
                            std: float = 0.005) -> np.ndarray:
        """Generate multi-bit (noisy) APUF responses.

        Args:
            chals (np.ndarray):
                Sequence of challenges (phase vectors). Shape `(d+1, k)`.
            mean (float):
                Mean of Gaussian noise added to each weight. Defaults to `0.0`.
            std (float):
                Standard deviation of Gaussian noise. Defaults to `0.005`.

        Returns:
            np.ndarray:
                Binary response vector of length `k` after noisy measurements.
        
        Raises:
            TypeError: If `chals` is not a np.ndarray.
            ValueError: If `chals` does not have shape (d+1, k).
            AssertionError: If `mean` is not a float, or 
                if `std` is not a non-negative float.
        """
        # types & ranges
        if not isinstance(chals, np.ndarray):
            raise TypeError("chals must be a numpy.ndarray")
        if chals.ndim != 2:
            raise ValueError(f"chals must be 2-D, got shape {chals.shape}")
        if chals.shape[0] != self.d:
            raise ValueError(f"Expected {self.d}-bit challenge,\
                                got {chals.shape[0]}")
        assert isinstance(mean, float), "mean must be numeric"
        assert isinstance(std, float) and std >= 0, "std must be non-negative"

        noise = np.random.normal(mean, std, self.d)

        resp = (self.weights + noise) @ chals

        return self.__determine_responses_vectorized(resp)


    @staticmethod
    def compact_responses(resp: np.ndarray) -> bytes:
        """Pack a 1-D array of 0/1 response bits into a bytes object.

        This method takes a numpy array of response bits (dtype np.int8)
        and returns the packed bits as raw bytes, grouping each consecutive
        8 bits into one byte (big-endian within each byte).

        Args:
            resp (np.ndarray): 1-D array of response bits, values must be
                0 or 1, and dtype must be np.int8.

        Returns:
            bytes: The packed bytes.

        Raises:
            AssertionError: If `resp` is not a 1-D np.ndarray of dtype np.int8,
                            or if it contains values other than 0 or 1.
        """
        # input validation
        assert isinstance(resp, np.ndarray), "resp must be a numpy.ndarray"
        assert resp.dtype == np.int8, f"resp bits must be np.int8,\
                                        got {resp.dtype}"
        assert resp.ndim == 1, f"resp must be 1-D array, got shape {resp.shape}"
        assert set(resp).issubset({0, 1}), f"resp must contain only 0 or 1,\
                                        got {set(resp)}"


        return bytes(np.packbits(resp))





def main():
    from challenges import generate_k_challenges    #pylint: disable=import-outside-toplevel
    # TESTING ONLY
    a = APUF(64)
    b = APUF(32)

    achal = generate_k_challenges(10, 64)

    bchal = generate_k_challenges(10, 32)

    aresp = a.get_noisy_responses(achal)
    bresp = b.get_noisy_responses(bchal)

    # bigchal = APUF.generate_n_k_challenges(5, 10, 64, 8)

    print(aresp, bresp)

if __name__ == "__main__":
    main()
