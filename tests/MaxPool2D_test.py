import unittest
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from PartialPool import PartialMaxPool2D

class TestPartialMaxPool2D(unittest.TestCase):
    def setUp(self):
        self.pool = PartialMaxPool2D(kernel_size=2, stride=2)

    def test_basic_masking(self):
        """Test with partially masked input."""
        x = th.tensor([[
            [1, 2, 3, th.nan],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [th.nan, 14, 15, 16]
        ]], dtype=th.float32)
        mask = th.tensor([[
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]], dtype=th.float32)

        pooled_x, pooled_mask = self.pool(x, mask)
        
        # Expected outputs
        expected_x = th.tensor([[[6, 8], [14, 16]]], dtype=th.float32)
        expected_mask = th.tensor([[[1, 1], [1, 1]]], dtype=th.float32)

        self.assertTrue(th.allclose(pooled_x, expected_x, equal_nan=True))
        self.assertTrue(th.allclose(pooled_mask, expected_mask))

    def test_fully_masked_window(self):
        """Test with a fully masked window."""
        x = th.tensor([[
            [1, th.nan, 3, 4],
            [5, 6, th.nan, 8],
            [9, 10, 11, 12],
            [th.nan, 14, 15, 16]
        ]], dtype=th.float32)
        mask = th.tensor([[
            [0, 0, 0, 0],  # Fully masked window
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]], dtype=th.float32)

        pooled_x, pooled_mask = self.pool(x, mask)
        
        # Expected outputs
        expected_x = th.tensor([[[float("nan"), 8], [14, 16]]], dtype=th.float32)
        expected_mask = th.tensor([[[0, 1], [1, 1]]], dtype=th.float32)

        self.assertTrue(th.allclose(pooled_x, expected_x, equal_nan=True))
        self.assertTrue(th.allclose(pooled_mask, expected_mask))

    def test_nan_handling(self):
        """Test input with NaNs (should propagate)."""
        x = th.tensor([[
            [1, float("nan"), 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]], dtype=th.float32)
        mask = th.tensor([[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]], dtype=th.float32)

        pooled_x, pooled_mask = self.pool(x, mask)
        
        # Expected outputs
        expected_x = th.tensor([[[float("nan"), 8], [14, 16]]], dtype=th.float32)
        expected_mask = th.tensor([[[1, 1], [1, 1]]], dtype=th.float32)

        self.assertTrue(th.allclose(pooled_x, expected_x, equal_nan=True))
        self.assertTrue(th.allclose(pooled_mask, expected_mask))

if __name__ == "__main__":
    unittest.main()