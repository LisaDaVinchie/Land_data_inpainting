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
        self.batch_size = 1
        self.in_channels = 1
        self.out_channels = 1
        self.kernel_size = (3, 3)
        self.width = 6
        self.height = 6
        self.input_size = (self.batch_size, self.in_channels, self.width , self.height)
        self.mask_size = (self.batch_size, 1, self.width , self.height)
        
        self.input_tensor = th.rand(self.input_size)
        self.input_tensor[:, :, 0:2, 0:2] = th.nan
        self.mask_tensor = th.ones_like(self.input_tensor)
        self.mask_tensor[th.isnan(self.input_tensor)] = 0

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

    def test_partial_conv2d_output(self):
        # Initialize PartialConv2d layer with fixed weights and bias
        partial_conv = PartialConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1,
            bias=True
        )

        # Override the weights and bias with fixed values
        partial_conv.weight = nn.Parameter(self.weight)
        partial_conv.bias = nn.Parameter(self.bias)

        # Forward pass with fixed input and mask
        output, updated_mask = partial_conv(self.input_tensor, self.mask_tensor)
        # print("Initial nan mask:\n", th.isnan(self.input_tensor))
        # print("updated mask:\n", updated_mask)
        # print("nan mask:\n", th.isnan(output))

        # Manually compute the expected output
        # Step 1: Apply the mask to the input
        masked_input = self.input_tensor * self.mask_tensor

        # Step 2: Perform standard convolution on the masked input
        expected_output = F.conv2d(
            masked_input, self.weight, bias=self.bias,
            stride=1, padding=1
        )

        # Step 3: Compute the mask ratio
        weight_mask_updater = th.ones(1, 1, 3, 3)
        update_mask = F.conv2d(
            self.mask_tensor, weight_mask_updater,
            stride=1, padding=1
        )
        mask_ratio = 9 / (update_mask + 1e-8)  # slide_winsize = 9 (3x3 kernel)
        mask_ratio = mask_ratio * th.clamp(update_mask, 0, 1)

        # Step 4: Apply the mask ratio to the output
        expected_output = expected_output * mask_ratio
        
        
        # Compare the output of PartialConv2d with the manually computed output
        
        # Check that nans are where they should be
        nan_mask_expected = th.isnan(expected_output)
        nan_mask_output = th.isnan(output)
        self.assertTrue(th.equal(nan_mask_expected, nan_mask_output), "NaN masks do not match.")

        # Check that non-NaN values match
        non_nan_mask = ~nan_mask_expected
        self.assertTrue(th.allclose(output[non_nan_mask], expected_output[non_nan_mask], atol=1e-6), "Non-NaN values do not match.")


if __name__ == '__main__':
    unittest.main()