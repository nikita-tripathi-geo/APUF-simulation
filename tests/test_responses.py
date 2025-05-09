#pylint: disable=missing-module-docstring
import unittest
from src.apuf import Response


class TestResponse(unittest.TestCase):  #pylint: disable=missing-class-docstring
    def test_init_invalid_dim(self):
        # must be 1-D
        with self.assertRaises(ValueError):
            Response([[1, 0], [0, 1]])

    def test_len_length_hw(self):
        bits = [1, 0, 1, 1]
        r = Response(bits)
        self.assertEqual(len(r), 4)
        self.assertEqual(r.length, 4)
        self.assertEqual(r.hw, 3)

    def test_str_and_repr(self):
        bits = [0, 1, 1, 0]
        r = Response(bits)
        self.assertEqual(str(r), "0110")
        # repr should show the bit-string
        self.assertEqual(repr(r), "Response('[False  True  True False]')")

    def test_bytes(self):
        # 1 0 1 0 0 0 0 1  → 0b10100001 = 0xa1
        r = Response([1, 0, 1, 0, 0, 0, 0, 1])
        self.assertEqual(bytes(r), b"\xa1")

    def test_bitwise_operators(self):
        a = Response([1, 1, 0, 0])
        b = Response([1, 0, 1, 0])

        # NOT
        self.assertEqual(str(~a), "0011")
        # AND
        self.assertEqual(str(a & b), "1000")
        # OR
        self.assertEqual(str(a | b), "1110")
        # XOR
        self.assertEqual(str(a ^ b), "0110")
        # aliases + and -
        self.assertEqual(str(a + b), "0110")
        self.assertEqual(str(a - b), "0110")

    def test_distance_and_equality(self):
        a = Response([1, 1, 1, 0, 0, 0])
        b = Response([1, 0, 1, 0, 1, 0])
        # Hamming distance = number of differing bits
        self.assertEqual(a.dist(b), 2)
        self.assertEqual(b.dist(a), 2)

        # equality same object
        self.assertTrue(a == a)
        # inequality different bit-pattern
        self.assertFalse(a == b)

    def test_eq_type_and_length_mismatch(self):
        a = Response([1, 0, 1])
        # comparing to non-Response returns False, not an exception
        self.assertFalse(a == 123)

        # comparing mismatched lengths should raise ValueError
        c = Response([0, 1])
        with self.assertRaises(ValueError):
            fail = a == c   #pylint: disable=unused-variable

    def test_and_or_xor_length_mismatch(self):
        a = Response([1, 0, 1])
        b = Response([1, 0, 1, 0])
        for op in (lambda x,y: x & y, lambda x,y: x | y, lambda x,y: x ^ y):
            with self.assertRaises(ValueError):
                op(a, b)

if __name__ == "__main__":
    unittest.main()
