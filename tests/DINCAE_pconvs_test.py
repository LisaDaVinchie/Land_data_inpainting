import unittest
import torch as th
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from models import DINCAE_pconvs, initialize_model_and_dataset_kind


class TestDINCAEPconvs(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters."""
        # Create a temporary JSON file for model parameters
        
        # Test input dimensions
        self.batch_size = 5
        self.channels_to_keep = ["c1", "c2", "c3"]
        self.n_channels = len(self.channels_to_keep) + 1
        self.ncols = 20
        self.nrows = 15
        self.mask_dim = 4
        
        self.middle_channels = [16, 16, 16, 16, 16]
        self.kernel_sizes = [5, 3, 5, 3, 5]
        self.pooling_sizes = [2, 2, 2, 2, 2]
        self.interp_mode = "nearest"
        
        self.params = {
            "dataset": {
                "cutted_nrows": self.nrows,
                "cutted_ncols": self.ncols,
                "dataset_kind": "test",
                "test": {
                    "n_channels": self.n_channels,
                    "channels_to_keep": self.channels_to_keep,
                }
            },
            "models": {
                "DINCAE_pconvs": {
                    "middle_channels": self.middle_channels,
                    "kernel_sizes": self.kernel_sizes,
                    "pooling_sizes": self.pooling_sizes,
                    "interp_mode": self.interp_mode
                }
            }
        }
        
        self.dummy_input = th.rand(self.batch_size, self.n_channels, self.nrows, self.ncols)
        self.dummy_input[:, :, 0:self.mask_dim, 0:self.mask_dim] = th.nan
        self.dummy_mask = th.ones_like(self.dummy_input)
        self.dummy_mask[th.isnan(self.dummy_input)] = 0
        
        self.model = DINCAE_pconvs(self.params)
        self.model.layers_setup()

    def test_initialization(self):
        """Test that the network initializes correctly."""

        # Check that the network has the correct attributes
        self.assertEqual(self.model.n_channels, self.n_channels)
        self.assertEqual(self.model.image_nrows, self.nrows)
        self.assertEqual(self.model.image_ncols, self.ncols)
        self.assertEqual(self.model.middle_channels, self.middle_channels)
        self.assertEqual(self.model.kernel_sizes, self.kernel_sizes)
        self.assertEqual(self.model.pooling_sizes, self.pooling_sizes)
        self.assertEqual(self.model.interp_mode, self.interp_mode)
    
    def test_initalization_priority(self):
        """Test that input is preferred over the Json file."""
        
        model = DINCAE_pconvs(self.params,
                                   n_channels = self.n_channels + 1,
                                   image_nrows= self.nrows + 1,
                                   image_ncols= self.ncols + 1,
                                   middle_channels = self.middle_channels + [1],
                                   kernel_sizes = self.kernel_sizes + [1],
                                   pooling_sizes = self.pooling_sizes + [1],
                                   interp_mode = "bilinear")
        model.layers_setup()
        # Check that the network has the correct attributes
        self.assertEqual(model.n_channels, self.n_channels + 1)
        self.assertEqual(model.image_ncols, self.ncols + 1)
        self.assertEqual(model.middle_channels, self.middle_channels + [1])
        self.assertEqual(model.kernel_sizes, self.kernel_sizes + [1])
        self.assertEqual(model.pooling_sizes, self.pooling_sizes + [1])
        self.assertEqual(model.interp_mode, "bilinear")

    def test_forward_pass_output_shape(self):
        """Test that the forward pass works and produces outputs of the correct shape."""

        # Perform forward pass
        output_img = self.model(self.dummy_input, self.dummy_mask)

        # Check that the output has the correct shape
        self.assertEqual(output_img.shape, self.dummy_input.shape)
        self.assertEqual(self.model.output_mask.shape, self.dummy_mask.shape)
    
    def test_forward_pass_output_nans(self):
        """Test that the forward pass works and produces outputs with nans in the right places."""
        
        # Perform forward pass
        output_img = self.model(self.dummy_input, self.dummy_mask)
        
        # print("input mask: ", self.dummy_mask)
        # print("output nans mask: ", th.isnan(output_img))
        
        # Check that the output has the correct nans
        self.assertTrue(th.isnan(output_img).any())
        self.assertFalse(th.isnan(self.model.output_mask).any())
        
    def test_automatic_initialization(self):
        channels_to_keep = ["c1", "c2", "c3", "c4"]
        dataset_params = {
            "dataset": {
                "cutted_nrows": self.nrows + 1,
                "cutted_ncols": self.ncols + 1,
                "dataset_kind": "test",
                "test": {
                    "n_channels": self.n_channels + 1,
                    "channels_to_keep": channels_to_keep
                }
            }
        }
        model, dataset_kind = initialize_model_and_dataset_kind(self.params, "DINCAE_pconvs", dataset_params)
        
        # Check that the network has the correct attributes
        self.assertIsInstance(model, DINCAE_pconvs)
        self.assertEqual(dataset_kind, "minimal")
        self.assertEqual(model.n_channels, self.n_channels + 1)
        self.assertEqual(model.image_nrows, self.nrows + 1)
        self.assertEqual(model.image_ncols, self.ncols + 1)
        self.assertEqual(model.middle_channels, self.middle_channels)
        self.assertEqual(model.kernel_sizes, self.kernel_sizes)
        self.assertEqual(model.pooling_sizes, self.pooling_sizes)
        self.assertEqual(model.interp_mode, self.interp_mode)

if __name__ == "__main__":
    unittest.main()