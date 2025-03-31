import unittest
import torch as th
import torch.nn.functional as F
from torch import nn

import sys
import os

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from PartialConv import PartialConv2d

class TestPartialConv2dOutput(unittest.TestCase):
    def setUp(self):
        # Set up fixed input, mask, weights, and biases
        self.batch_size = 4
        self.in_channels = 2
        self.out_channels = 3
        self.kernel_size = (3, 3)
        self.nrows = 10
        self.ncols = 5
        self.input_size = (self.batch_size, self.in_channels, self.nrows , self.ncols)
        
        self.input_tensor = th.rand(self.input_size)
        self.input_tensor[:, :, 0:2, 0:2] = 0
        self.mask_tensor = th.ones_like(self.input_tensor)
        self.mask_tensor[:, :, 0:2, 0:2] = 0

        # Fixed weights and bias
        self.weight = th.tensor([[
            [[1, 0, -1],
             [1, 0, -1],
             [1, 0, -1]]
        ]], dtype=th.float32)
        self.bias = th.tensor([0.0], dtype=th.float32)
        
    def test_weight_initialization(self):
        partial_conv = PartialConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1,
            bias=True
        )
        
        # Check that the weights do not contain nans
        weights = partial_conv.weight
        self.assertFalse(th.isnan(weights).any(), "Weights contain NaN values.")
        
    def test_masked_input_invariance(self):
        """Test that the output is invariant to the input values in the masked regions."""
        input_tensor1 = th.tensor([[[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]],
                                  
                                  [[10, 11, 12],
                                  [13, 14, 15],
                                  [16, 17, 18]]], dtype=th.float32).unsqueeze(0)
        mask_tensor = th.tensor([[[1, 0, 1],
                                [1, 0, 1],
                                [1, 1, 1]],
                                 
                                [[1, 1, 1],
                                [1, 1, 0],
                                [1, 1, 0]]
                                 ], dtype=th.float32).unsqueeze(0)
        
        input_tensor2 = th.tensor([[[1, 20, 3],
                                  [4, 50, 6],
                                  [7, 8, 9]],
                                  
                                  [[10, 11, 12],
                                  [13, 14, 150],
                                  [16, 17, 180]]], dtype=th.float32).unsqueeze(0)
        
        in_size = input_tensor1.shape[1]
        out_size = 3
        kernel_size = (2, 2)
        
        weight = th.rand(out_size, in_size, *kernel_size)
        bias = th.zeros(out_size)
        
        model = PartialConv2d(
            in_channels=in_size,
            out_channels=out_size,
            kernel_size=kernel_size,
            padding=1,
            bias=True
        )
        
        # Override the weights and bias with fixed values
        model.weight = nn.Parameter(weight)
        model.bias = nn.Parameter(bias)
        
        # Forward pass with fixed input and mask
        output1, updated_mask1 = model(input_tensor1, mask_tensor)
        output2, updated_mask2 = model(input_tensor2, mask_tensor)
        
        # Check that the outputs are the same
        self.assertTrue(th.allclose(output1, output2, atol=1e-6), "Output does not match expected output.")
        # Check that the updated masks are the same
        self.assertTrue(th.allclose(updated_mask1, updated_mask2, atol=1e-6), "Updated mask does not match expected output.")


if __name__ == '__main__':
    unittest.main()