#pylint: disable=missing-module-docstring
import unittest
import numpy as np
from src.apuf import Response, APUF, XORPUF, LADM


class TestLADMAndCompact(unittest.TestCase):    #pylint: disable=missing-class-docstring
    def test_compact_responses_accepts_and_packs(self):
        r = Response([1, 0, 1, 0, 0, 0, 0, 1])
        # static method should accept a Response
        out = LADM.compact_responses(r)
        self.assertIsInstance(out, bytes)
        # same as bytes(r)
        self.assertEqual(out, bytes(r))

    def test_compact_responses_rejects_other(self):
        with self.assertRaises(AssertionError):
            # should fail
            LADM.compact_responses("not a Response")


class TestAPUF(unittest.TestCase):  #pylint: disable=missing-class-docstring
    def test_constructor_validation(self):
        with self.assertRaises(AssertionError):
            # d must be >0
            APUF(d=0)
        with self.assertRaises(AssertionError):
            # mean must be float
            APUF(d=1, mean="hello")
        with self.assertRaises(AssertionError):
            # std must be >=0
            APUF(d=1, std=-0.1)

    def test_get_responses_type_and_value_errors(self):
        apuf = APUF(d=1)
        # wrong type
        with self.assertRaises(TypeError):
            apuf.get_responses("not ndarray")
        # wrong ndim
        with self.assertRaises(ValueError):
            apuf.get_responses(np.zeros(3))
        # wrong first dimension
        arr = np.zeros((apuf.d + 1, 5))
        with self.assertRaises(ValueError):
            apuf.get_responses(arr)

    def test_get_responses_no_noise_logic(self):
        # d=1 -> self.d = 2 weights
        apuf = APUF(d=1)
        # override weights for predictability
        apuf.weights = np.array([2.0, 2.0])
        # make 3 “phase vectors”
        # col0:  [ 1,  1] -> 4 -> True
        # col1:  [ 1,  1] -> 4 -> True
        # col2:  [ 1, -1] -> 0 -> False
        chals = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ])
        resp = apuf.get_responses(chals, nmean=0.0, nstd=0.0)
        # test underlying bits
        expected = Response([True, True, False])
        self.assertEqual(resp, expected)

    def test_reproducibility_with_zero_noise(self):
        # zero noise should produce identical runs
        apuf = APUF(d=2)
        chals = np.ones((apuf.d, 4))
        r1 = apuf.get_responses(chals, nmean=0.0, nstd=0.0)
        r2 = apuf.get_responses(chals, nmean=0.0, nstd=0.0)
        self.assertEqual(r1, r2)


class TestXORPUF(unittest.TestCase):    #pylint: disable=missing-class-docstring
    def test_xorpuf_simple_composition(self):
        # dummy PUFs that return a fixed Response and ignores chals
        class Dummy:
            def __init__(self, resp: Response):
                self.resp = resp
            def get_responses(self, chals):
                _ = chals
                return self.resp

        resp1 = Response([1, 0, 1, 1])
        resp2 = Response([0, 1, 1, 0])
        puf = XORPUF(children=[Dummy(resp1), Dummy(resp2)])
        out = puf.get_responses(None)  # chals is ignored
        # bitwise XOR of resp1 and resp2
        expected = Response([True, True, False, True])
        self.assertEqual(out, expected)

    def test_xorpuf_length_mismatch(self):
        # dummy PUFs that return a fixed Response and ignores chals
        class Dummy:
            def __init__(self, resp: Response):
                self.resp = resp
            def get_responses(self, chals):
                _ = chals
                return self.resp

        r1 = Response([1, 0, 1])
        r2 = Response([1, 0, 1, 0])
        puf = XORPUF(children=[Dummy(r1), Dummy(r2)])
        with self.assertRaises(ValueError):
            puf.get_responses(None)


if __name__ == "__main__":
    unittest.main()
