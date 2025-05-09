# pylint: disable=missing-module-docstring
import unittest
import numpy as np

from src.challenges import (
    generate_k_challenges,
    generate_n_k_challenges,
    unit_generation,
    generate_challenges_mp
)


class TestGenerateKChallenges(unittest.TestCase):   # pylint: disable=missing-class-docstring
    def test_invalid_k_and_d(self):
        with self.assertRaises(AssertionError):
            generate_k_challenges(k=0, d=5)
        with self.assertRaises(AssertionError):
            generate_k_challenges(k=5, d=0)
        with self.assertRaises(AssertionError):
            generate_k_challenges(k=5, d=5, seed="not_int")

    def test_output_shape_and_values(self):
        k, d, seed = 3, 4, 123
        phi = generate_k_challenges(k=k, d=d, seed=seed)
        # phi should be shape (d+1, k)
        self.assertEqual(phi.shape, (d+1, k))
        # entries must be ±1
        unique = set(np.unique(phi))
        self.assertEqual(unique, {1.0, -1.0})
        # last row must be all +1
        np.testing.assert_array_equal(phi[-1], np.ones(k))

    def test_reproducibility(self):
        a = generate_k_challenges(k=5, d=6, seed=42)
        b = generate_k_challenges(k=5, d=6, seed=42)
        np.testing.assert_array_equal(a, b)
        # changing seed changes output
        c = generate_k_challenges(k=5, d=6, seed=43)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(a, c)


class TestGenerateNkChallenges(unittest.TestCase):  # pylint: disable=missing-class-docstring
    def test_invalid_n_k_d(self):
        with self.assertRaises(AssertionError):
            generate_n_k_challenges(n=0, k=1, d=1)
        with self.assertRaises(AssertionError):
            generate_n_k_challenges(n=1, k=0, d=1)
        with self.assertRaises(AssertionError):
            generate_n_k_challenges(n=1, k=1, d=0)
        with self.assertRaises(AssertionError):
            generate_n_k_challenges(n=2, k=2, d=2, seed="bad")

    def test_output_shape_and_content(self):
        n, k, d, seed = 2, 3, 4, 7
        arr = generate_n_k_challenges(n=n, k=k, d=d, seed=seed)
        # should be shape (n, d+1, k)
        self.assertEqual(arr.shape, (n, d+1, k))

        # all entries ±1 and last row = +1
        for i in range(n):
            unique = set(np.unique(arr[i]))
            self.assertEqual(unique, {1.0, -1.0})
            np.testing.assert_array_equal(arr[i][-1], np.ones(k))

    def test_reproducibility(self):
        x1 = generate_n_k_challenges(n=3, k=4, d=5, seed=99)
        x2 = generate_n_k_challenges(n=3, k=4, d=5, seed=99)
        for a, b in zip(x1, x2):
            np.testing.assert_array_equal(a, b)


class TestUnitGeneration(unittest.TestCase):    # pylint: disable=missing-class-docstring
    def test_unit_generation_matches_generate_k(self):
        # same parameters should match
        params = (4, 5, 2025)
        out1 = unit_generation(params)
        out2 = generate_k_challenges(*params)
        np.testing.assert_array_equal(out1, out2)


class TestGenerateChallengesMP(unittest.TestCase):  # pylint: disable=missing-class-docstring
    def test_invalid_params(self):
        with self.assertRaises(AssertionError):
            generate_challenges_mp(n=0, k=1, d=1, seed=1, proc=1)
        with self.assertRaises(AssertionError):
            generate_challenges_mp(n=1, k=0, d=1, seed=1, proc=1)
        with self.assertRaises(AssertionError):
            generate_challenges_mp(n=1, k=1, d=0, seed=1, proc=1)
        with self.assertRaises(AssertionError):
            generate_challenges_mp(n=1, k=1, d=1, seed="bad", proc=1)
        with self.assertRaises(AssertionError):
            generate_challenges_mp(n=1, k=1, d=1, seed=None, proc=0)

    def test_parallel_equivalence(self):
        n, k, d, seed = 3, 4, 5, 17
        # run with 1 process
        mp_res = generate_challenges_mp(n=n, k=k, d=d, seed=seed, proc=1)
        # expected by calling unit_generation for each batch
        expected = [unit_generation((k, d, seed + i)) for i in range(n)]
        # compare each array
        self.assertEqual(len(mp_res), n)
        for a, b in zip(mp_res, expected):
            np.testing.assert_array_equal(a, b)

    def test_default_parallel(self):
        n, k, d= 2, 3, 5
        # just ensure it runs without error with defaults
        res = generate_challenges_mp(n=n, k=k, d=d)
        self.assertEqual(len(res), n)
        for mat in res:
            self.assertEqual(mat.shape, (d+1, k))


if __name__ == "__main__":
    unittest.main()
