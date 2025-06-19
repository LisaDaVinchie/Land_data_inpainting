import unittest
import torch as th
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from models import DummyModel


class TestDummyModel(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()

    def test_forward_pass_output_shape(self):
        """Test that the forward pass works and produces outputs of the correct shape."""
        
        self.dummy_input = th.rand(5, 13, 10, 10)  # Example input tensor
        self.dummy_mask = th.ones_like(self.dummy_input)  # Example mask tensor

        # Perform forward pass
        output_img = self.model(self.dummy_input, self.dummy_mask)

        # Check that the output has the correct shape
        self.assertEqual(output_img.shape, (5, 2, 10, 10))
    
    def test_nan_oresence(self):
        """Test that the forward pass works and produces outputs of the correct shape."""
        
        self.dummy_input = th.rand(5, 13, 10, 10)  # Example input tensor
        self.dummy_mask = th.ones_like(self.dummy_input)  # Example mask tensor

        # Perform forward pass
        output_img = self.model(self.dummy_input, self.dummy_mask)

        # Check that the output has no nan values
        self.assertFalse(th.isnan(output_img).any(), "Output contains NaN values")

if __name__ == "__main__":
    unittest.main()