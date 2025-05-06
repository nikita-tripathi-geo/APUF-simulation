"""TODO - Write module docstring
"""
import numpy as np


class APUF:
    """TODO - APUF docstring
    """
    def __init__(self, d: int = 128, weight_mean: float = 0.0, weight_std: float = 0.05):
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
        """Generate noisy APUF/XOR-PUF responses.

        Args:
            chals (np.ndarray): Sequence of challenges (phase vectors).
            noise_mean (float): Mean of Gaussian noise. Defaults to 0.0.
            noise_std (float): Standard deviation of Gaussian noise. Defaults to 0.005.

        Returns:
            np.ndarray: Binary response vector after noisy measurements.
        """
        noise = np.random.normal(noise_mean, noise_std, self.d)

        resp = (self.weights + noise) @ chals

        return self.determine_responses_vectorized(resp)


    @staticmethod
    def generate_k_challenges(k: int = 100, d: int = 128, seed: int = None) -> np.ndarray:
        """Generate `k` random challenges.

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
