import unittest
import sys
import os
import torch as th

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.models import DINCAE_like, initialize_model_and_dataset_kind

class Test_DINCAE_model(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary JSON file with parameters."""
        self.middle_channels = [16, 30, 58, 110, 209]
        self.kernel_sizes = [3, 3, 3, 3, 3]
        self.pooling_sizes = [2, 2, 2, 2, 2]
        self.interp_mode = "bilinear"
        self.nrows = 168
        self.ncols = 144 
        self.channels_to_keep = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
        self.n_channels = len(self.channels_to_keep) + 1
        self.placeholder = 0.0
        
        self.batch_size = 32
        self.model_params = {
            "training": {
                "placeholder": self.placeholder,
            },
            "dataset": {
                "cutted_nrows": self.nrows,
                "cutted_ncols": self.ncols,
                "dataset_kind": "test",
                "test": {
                    "n_channels": self.n_channels,
                    "channels_to_keep": self.channels_to_keep
                },
                "placeholder": 0.0
            },
            "models": {
                "DINCAE_like": {
                    "middle_channels": self.middle_channels,
                    "kernel_sizes": self.kernel_sizes,
                    "pooling_sizes": self.pooling_sizes,
                    "interp_mode": self.interp_mode,
                    "n_channels": self.n_channels,
                }
            
            }
        }
        
        self.model = DINCAE_like(params=self.model_params)
        self.model.layers_setup()
        
        self.input_tensor = th.rand(self.batch_size, self.model.n_channels, self.nrows, self.ncols)
        self.masks = th.ones(self.batch_size, self.model.n_channels, self.nrows, self.ncols)
        self.masks[:, 0, 0:10, 0:10] = 0.0
        
    def test_model_initialization(self):
        """Test if the model initializes correctly"""
        # Test if the model initializes correctly
        
        # Check if the model has the correct attributes
        self.assertEqual(self.model.n_channels, self.n_channels)
        self.assertEqual(self.model.middle_channels, self.middle_channels)
        self.assertEqual(self.model.kernel_sizes, self.kernel_sizes)
        self.assertEqual(self.model.pooling_sizes, self.pooling_sizes)
        self.assertEqual(self.model.image_nrows, self.nrows)
        self.assertEqual(self.model.image_ncols, self.ncols)
        self.assertEqual(self.model.interp_mode, self.interp_mode)
        self.assertEqual(self.model.placeholder, self.placeholder)
    
    def test_initalization_priority(self):
        """Test that input is preferred over the Json file."""
        
        model = DINCAE_like(self.model_params,
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
        
    def test_forward_pass(self):
        """Test if the forward pass of the model works correctly"""
        # Test the forward pass of the model
        output = self.model(self.input_tensor, self.masks)
        
        # Check if the output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.model.n_channels, self.nrows, self.ncols))
    
    def test_automatic_initialization(self):
        channels_to_keep = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
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
        model, dataset_kind = initialize_model_and_dataset_kind(self.model_params, "DINCAE_like", dataset_params)
        
        # Check that the network has the correct attributes
        self.assertIsInstance(model, DINCAE_like)
        self.assertEqual(dataset_kind, "extended")
        self.assertEqual(model.n_channels, self.n_channels + 1)
        self.assertEqual(model.image_nrows, self.nrows + 1)
        self.assertEqual(model.image_ncols, self.ncols + 1)
        self.assertEqual(model.middle_channels, self.middle_channels)
        self.assertEqual(model.kernel_sizes, self.kernel_sizes)
        self.assertEqual(model.pooling_sizes, self.pooling_sizes)
        self.assertEqual(model.interp_mode, self.interp_mode)

if __name__ == "__main__":
    unittest.main()