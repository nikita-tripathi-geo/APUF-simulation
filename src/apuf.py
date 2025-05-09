"""Simulate an delay-based PUFs and their challenge-response behaviors
using Lim's Linear Additive Delay Model (LADM).

# Usage: TODO UPDATE
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
from abc import ABC, abstractmethod
import numpy as np


class Response:
    """
    Immutable response vector for a delay-based PUF.
    """
    def __init__(self, data):
        # Convert input to 1D numpy bool array
        arr = np.asarray(data, dtype="?")
        if arr.ndim != 1:
            raise ValueError(f"Response must be 1D, got shape {arr.shape}")
        # Make it read-only
        arr.setflags(write=False)
        self._data = arr

    def __len__(self):
        return self._data.size

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def length(self) -> int:
        """Length of the response."""
        return len(self)

    @property
    def hw(self) -> int:
        """Hamming weight of the response."""
        return np.count_nonzero(self._data)

    def __str__(self) -> str:
        """Binary string, e.g. '1010010'."""
        return "".join("1" if bit else "0" for bit in self._data)

    def __repr__(self):
        """For debugging."""
        return f"Response('{self._data}')"

    def __bytes__(self):
        """Convert to a bytes object."""
        return bytes(np.packbits(self._data, bitorder="big"))

    def __invert__(self):
        """Bitwise NOT."""
        return Response(np.logical_not(self._data))

    def __and__(self, other):
        """Bitwise AND."""
        if not isinstance(other, Response):
            return NotImplemented
        if self.length != other.length:
            raise ValueError("Cannot AND Responses of different length")
        return Response(np.logical_and(self._data, other._data))

    def __or__(self, other):
        """Bitwise OR."""
        if not isinstance(other, Response):
            return NotImplemented
        if self.length != other.length:
            raise ValueError("Cannot OR Responses of different length")
        return Response(np.logical_or(self._data, other._data))

    def __xor__(self, other):
        """Bitwise XOR."""
        if not isinstance(other, Response):
            return NotImplemented
        if self.length != other.length:
            raise ValueError("Cannot XOR Responses of different length")
        return Response(self._data ^ other._data)

    def __add__(self, other):
        """Also bitwise XOR (addition)."""
        return self.__xor__(other)

    def __sub__(self, other):
        """Also bitwise XOR (subtraction)."""
        return self.__add__(other)

    def __eq__(self, other):
        if not isinstance(other, Response):
            return NotImplemented
        if self.length != other.length:
            raise ValueError("Cannot XOR Responses of different length")
        return np.array_equal(self._data, other._data)

    def dist(self, other) -> int:
        """
        Hamming distance between two Responses.
        """
        if not isinstance(other, Response):
            raise TypeError("distance requires another Response")
        # Length checking handled by xor/add/sub
        diff = self ^ other
        return diff.hw



class LADM(ABC):
    """Abstract interface for all delay-based PUFs using LADM.
    """

    @abstractmethod
    def get_responses(self, chals: np.ndarray,
                     nmean: float, nstd: float) -> Response:
        """Given `k` challenges and noise parameters, return a `k` bit response.
        """
        pass


    @staticmethod
    def compact_responses(resp: Response) -> bytes:
        """Pack a 1-D array of 0/1 response bits into a bytes object.

        This method takes a multi-bit response (Response)
        and returns the packed bits as raw bytes, grouping each consecutive
        8 bits into one byte (big-endian within each byte).

        Args:
            resp (Response): 1-D array of `k` response bits.

        Returns:
            bytes: The packed bytes.

        Raises:
            AssertionError: If `resp` is not a Response.
        """
        # input validation
        assert isinstance(resp, Response), "resp must be a Response"

        return bytes(resp)



class APUF(LADM):
    """Simulate an Arbiter Physically Unclonable Function (APUF) via LADM.

    This class samples independent biases of each layer (weights)
    and provides methods for:
      - thresholding an array of delay differences (float) into a Response,
      - simulating measurement noise using a Gaussian distribution
        and computing noisy responses for many challenges at once.
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


    def __determine_responses(self, delays: np.ndarray) -> Response:
        """Threshold delays to obtain response bits (Response)."""
        return Response(delays > 0)


    def get_responses(self, chals: np.ndarray,
                            nmean: float = 0.0,
                            nstd: float = 0.005) -> Response:
        """Generate multi-bit (noisy) APUF responses.

        Args:
            chals (np.ndarray):
                Sequence of challenges (phase vectors). Shape `(d+1, k)`.
            nmean (float):
                Mean of Gaussian noise added to each weight. Defaults to `0.0`.
            nstd (float):
                Standard deviation of Gaussian noise. Defaults to `0.005`.

        Returns:
            Response:
                Response vector of length `k` after noisy measurements.
        
        Raises:
            TypeError: If `chals` is not a np.ndarray.
            ValueError: If `chals` does not have shape (d+1, k).
            AssertionError: If `nmean` is not a float, or 
                if `nstd` is not a non-negative float.
        """
        # types & ranges
        if not isinstance(chals, np.ndarray):
            raise TypeError("chals must be a numpy.ndarray")
        if chals.ndim != 2:
            raise ValueError(f"chals must be 2-D, got shape {chals.shape}")
        if chals.shape[0] != self.d:
            raise ValueError(f"Expected {self.d}-bit challenge,\
                                got {chals.shape[0]}")
        assert isinstance(nmean, float), "nmean must be numeric"
        assert isinstance(nstd, float) and nstd >= 0, "nstd must be nonnegative"

        noise = np.random.normal(nmean, nstd, self.d)

        resp = (self.weights + noise) @ chals

        return self.__determine_responses(resp)



class XORPUF(LADM):
    """Simulate an XOR-PUF via LADM.

    This class samples independent biases of each layer (weights)
    and provides methods for:
      - thresholding an array of delay differences (float) into a Response,
      - simulating measurement noise using a Gaussian distribution
        and computing noisy responses for many challenges at once.
    """
    def __init__(self, children: list[APUF]):
        """
        children: list of APUF instances whose bit-wise
                  responses will be XOR-ed together.
        """
        self.children = children

    def get_responses(self, chals: np.ndarray) -> Response:
        # get each child’s response, then XOR them all
        child_responses = [puf.get_responses(chals) for puf in self.children]
        xor_response = child_responses[0]
        for cr in child_responses[1:]:
            xor_response ^= cr
        return xor_response


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
