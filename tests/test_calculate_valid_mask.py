import unittest
import torch as th
import os
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from losses import calculate_valid_mask, calculate_valid_pixels

# Unit test class
class TestInpaintingUtils(unittest.TestCase):
    def setUp(self):
        # Common test data (6x6)
        self.masks = th.tensor([[[[
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1
        ]]]], dtype=th.bool).reshape(1, 1, 6, 6)

        self.target = th.tensor([[[[
            1.0, 1.0, -999., 1.0, 1.0, 1.0,
            1.0, -999., 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, -999, 1.0, 1.0,
            -999., 1.0, 1.0, -999., 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, -999., 1.0, 1.0, 1.0, 1.0
        ]]]]).reshape(1, 1, 6, 6)
        
        self.expected_valid_mask = th.tensor([[[[
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 0, 0,
            1, 1, 0, 1, 0, 0,
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1
        ]]]], dtype=th.bool).reshape(1, 1, 6, 6)

        self.nan_placeholder = -999.0

    def test_calculate_valid_pixels(self):
        expected_valid_pixels = 6  # based on manual inspection
        actual = calculate_valid_pixels(self.masks.clone(), self.target, self.nan_placeholder)
        self.assertEqual(actual, expected_valid_pixels)

    def test_calculate_valid_mask_inv_true(self):
        mask = calculate_valid_mask(self.masks.clone(), self.target, self.nan_placeholder, inv=True)
        self.assertTrue(th.equal(mask, ~self.expected_valid_mask))

    def test_calculate_valid_mask_inv_false(self):
        mask = calculate_valid_mask(self.masks.clone(), self.target, self.nan_placeholder)
        self.assertTrue(th.equal(mask, self.expected_valid_mask))

# Run the tests
if __name__ == "__main__":
    unittest.main()
