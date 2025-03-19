import unittest
import torch as th
import torch.nn as nn
from pathlib import Path
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from models import simple_PartialConv
from PartialConv import PartialConv2d


class TestSimplePartialConv(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters."""
        # Create a temporary JSON file for model parameters
        self.temp_dir = Path("temp_test_dir")
        self.temp_dir.mkdir(exist_ok=True)
        
        self.params_path = self.temp_dir / "params.json"
        self.params = {
            "dataset": {
                "n_channels": 3,
                "cutted_width": 64,
                "cutted_height": 64
            },
            "simple_PartialConv": {
                "middle_channels": [10, 10, 10],
                "kernel_size": [1, 1, 1],
                "stride": [7, 7, 7],
                "padding": [5, 5, 5],
                "output_padding": [5, 5, 5]
            }
        }
        with open(self.params_path, "w") as f:
            json.dump(self.params, f)

        # Test input dimensions
        self.batch_size = 2
        self.n_channels = 3
        self.height = 64
        self.width = 64

    def tearDown(self):
        """Clean up temporary directory after tests."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that the network initializes correctly."""
        model = simple_PartialConv(self.params_path)

        # Check that the network has the correct attributes
        self.assertEqual(model.n_channels, 3)
        self.assertEqual(model.middle_channels, [10, 10, 10])
        self.assertEqual(model.kernel_size, [1, 1, 1])
        self.assertEqual(model.stride, [7, 7, 7])
        self.assertEqual(model.padding, [5, 5, 5])
        self.assertEqual(model.output_padding, [5, 5, 5])

        # Check that the layers are initialized correctly
        self.assertIsInstance(model.conv1, PartialConv2d)
        self.assertIsInstance(model.conv2, PartialConv2d)
        self.assertIsInstance(model.linear3, nn.Linear)
        self.assertIsInstance(model.relu, nn.ReLU)

    def test_forward_pass(self):
        """Test that the forward pass works and produces outputs of the correct shape."""
        model = simple_PartialConv(self.params_path)

        # Create dummy input and mask tensors
        x = th.rand(self.batch_size, self.n_channels, self.height, self.width)
        mask = th.ones(self.batch_size, self.n_channels, self.height, self.width)
        masked_idxs = th.randint(0, 2, (self.batch_size, self.n_channels, self.height, self.width))
        mask[masked_idxs == 0] = 0

        # Perform forward pass
        output = model(x, mask)

        # Check that the output has the correct shape
        expected_shape = (self.batch_size, self.n_channels)
        self.assertEqual(output.shape, expected_shape)

    def test_custom_parameters(self):
        """Test that the network can be initialized with custom parameters."""
        custom_middle_channels = [16, 16, 16]
        custom_kernel_size = [3, 3, 3]
        custom_stride = [2, 2, 2]
        custom_padding = [1, 1, 1]
        custom_output_padding = [1, 1, 1]

        model = simple_PartialConv(
            self.params_path,
            middle_channels=custom_middle_channels,
            kernel_size=custom_kernel_size,
            stride=custom_stride,
            padding=custom_padding,
            output_padding=custom_output_padding,
        )

        # Check that the network uses the custom parameters
        self.assertEqual(model.middle_channels, custom_middle_channels)
        self.assertEqual(model.kernel_size, custom_kernel_size)
        self.assertEqual(model.stride, custom_stride)
        self.assertEqual(model.padding, custom_padding)
        self.assertEqual(model.output_padding, custom_output_padding)


if __name__ == "__main__":
    unittest.main()