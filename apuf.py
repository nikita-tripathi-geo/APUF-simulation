"""Simulate an Arbiter PUF and its corresponding challenge-response behavior
using Lim's Linear Additive Delay Model (LADM).
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

    def __init__(self, d: int = 128, weight_mean: float = 0.0, weight_std: float = 0.05):
        """Initialize an APUF with given number of layers and weight distribution.

        Args:
            d (int): Number of layers of APUF. Internally `d+1` weights are used
                (one for each stage plus the arbiter). Defaults to `128`.
            weight_mean (float): Mean of the Gaussian distribution used to generate
                weights. Defaults to `0.0`.
            weight_std (float): Standard deviation of the Gaussian distribution. 
                Defaults to `0.05`.
        """
        # We represent a `d`-layer APUF using `d+1` weights
        self.d = d + 1
        self.weight_mean = weight_mean
        self.weight_std = weight_std
        self.weights = np.random.normal(loc=weight_mean, scale=weight_std, size=self.d)

        # Vectorized thresholding function. Works on float ndarrays.
        self.determine_responses_vectorized = np.vectorize(self.determine_responses)


    def determine_responses(self, delay: float) -> int:
        """Threshold delay to obtain a response bit."""
        if delay > 0:
            return 1
        return 0


    def get_noisy_responses(self, chals: np.ndarray,
                            noise_mean: float = 0.0,
                            noise_std: float = 0.005) -> np.ndarray:
        """Generate multi-bit (noisy) APUF responses.

        Args:
            chals (np.ndarray): Sequence of challenges (phase vectors). Shape `(d+1, k)`
            noise_mean (float): Mean of Gaussian noise added to each weight. Defaults to `0.0`.
            noise_std (float): Standard deviation of Gaussian noise. Defaults to `0.005`.

        Returns:
            np.ndarray: Binary response vector of length `k` after noisy measurements.
        """
        noise = np.random.normal(noise_mean, noise_std, self.d)

        resp = (self.weights + noise) @ chals

        return self.determine_responses_vectorized(resp)


    @staticmethod
    def generate_k_challenges(k: int = 100, d: int = 128, seed: int = None) -> np.ndarray:
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

    # @staticmethod
    # TODO new static method that uses multiprocess to generate ell X k challenges
