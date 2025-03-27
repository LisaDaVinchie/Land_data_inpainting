import unittest
import torch as th
import torch.nn as nn
from pathlib import Path
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from models import DINCAE_pconvs


class TestSimplePartialConv(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters."""
        # Create a temporary JSON file for model parameters
        self.temp_dir = Path("temp_test_dir")
        self.temp_dir.mkdir(exist_ok=True)
        
        self.params_path = self.temp_dir / "params.json"
        
        # Test input dimensions
        self.batch_size = 5
        self.n_channels = 3
        self.height = 20
        self.width = 15
        self.mask_dim = 4
        
        self.middle_channels = [16, 16, 16, 16, 16]
        self.kernel_sizes = [5, 3, 5, 3, 5]
        self.pooling_sizes = [2, 2, 2, 2, 2]
        self.interp_mode = "nearest"
        
        self.params = {
            "dataset": {
                "n_channels": self.n_channels,
                "cutted_width": self.width,
                "cutted_height": self.height
            },
            "DINCAE_pconvs": {
                "middle_channels": self.middle_channels,
                "kernel_sizes": self.kernel_sizes,
                "pooling_sizes": self.pooling_sizes,
                "interp_mode": self.interp_mode
            }
        }
        with open(self.params_path, "w") as f:
            json.dump(self.params, f)
        
        self.dummy_input = th.rand(self.batch_size, self.n_channels, self.width, self.height)
        self.dummy_input[:, :, 0:self.mask_dim, 0:self.mask_dim] = th.nan
        self.dummy_mask = th.ones_like(self.dummy_input)
        self.dummy_mask[th.isnan(self.dummy_input)] = 0
        
        self.model = DINCAE_pconvs(self.params_path)

    def tearDown(self):
        """Clean up temporary directory after tests."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that the network initializes correctly."""

        # Check that the network has the correct attributes
        self.assertEqual(self.model.n_channels, self.n_channels)
        self.assertEqual(self.model.middle_channels, self.middle_channels)
        self.assertEqual(self.model.kernel_sizes, self.kernel_sizes)
        self.assertEqual(self.model.pooling_sizes, self.pooling_sizes)
        self.assertEqual(self.model.interp_mode, self.interp_mode)

    def test_forward_pass_output_shape(self):
        """Test that the forward pass works and produces outputs of the correct shape."""

        # Perform forward pass
        output_img, output_mask = self.model(self.dummy_input, self.dummy_mask)

        # Check that the output has the correct shape
        self.assertEqual(output_img.shape, self.dummy_input.shape)
        self.assertEqual(output_mask.shape, self.dummy_mask.shape)
    
    def test_forward_pass_output_nans(self):
        """Test that the forward pass works and produces outputs with nans in the right places."""
        
        # Perform forward pass
        output_img, output_mask = self.model(self.dummy_input, self.dummy_mask)
        
        # print("input mask: ", self.dummy_mask)
        # print("output nans mask: ", th.isnan(output_img))
        
        # Check that the output has the correct nans
        self.assertTrue(th.isnan(output_img).any())
        self.assertFalse(th.isnan(output_mask).any())
        


if __name__ == "__main__":
    unittest.main()