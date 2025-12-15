import unittest
import simple_math

class TestSimpleMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(simple_math.add(2, 3), 5)

    def test_subtract(self):
        self.assertEqual(simple_math.subtract(5, 2), 3)

    def test_multiply(self):
        self.assertEqual(simple_math.multiply(3, 4), 12)

    def test_divide(self):
        self.assertEqual(simple_math.divide(10, 2), 5)

    def test_is_even(self):
        self.assertTrue(simple_math.is_even(4))
        self.assertFalse(simple_math.is_even(3))

    def test_factorial(self):
        self.assertEqual(simple_math.factorial(0), 1)
        self.assertEqual(simple_math.factorial(5), 120)

if __name__ == "__main__":
    unittest.main()
