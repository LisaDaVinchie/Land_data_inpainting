import unittest
import sys
import os
import torch as th
from pathlib import Path
import json
from tempfile import NamedTemporaryFile

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.models import DINCAE_like

class Test_DINCAE_model(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary JSON file with parameters."""
        self.middle_channels = [16, 30, 58, 110, 209]
        self.kernel_sizes = [3, 3, 3, 3, 3]
        self.pooling_sizes = [2, 2, 2, 2, 2]
        self.interp_mode = "bilinear"
        self.nrows = 168
        self.ncols = 144
        self.n_channels = 10
        
        self.batch_size = 32
        self.model_params = {
            "dataset": {
                "cutted_nrows": self.nrows,
                "cutted_ncols": self.ncols,
                "dataset_kind": "test",
                "test": {
                    "n_channels": self.n_channels,
                }
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
        
        
        # Create a temporary file
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.model_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        self.model = DINCAE_like(params_path=self.params_path)
        
        self.input_tensor = th.rand(self.batch_size, self.model.n_channels, self.nrows, self.ncols)
        
    
    def tearDown(self):
        """Delete the temporary JSON file after tests."""
        Path(self.temp_json.name).unlink()
        
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
    
    def test_initalization_priority(self):
        """Test that input is preferred over the Json file."""
        
        model = DINCAE_like(self.params_path,
                                   n_channels = self.n_channels + 1,
                                   image_nrows= self.nrows + 1,
                                   image_ncols= self.ncols + 1,
                                   middle_channels = self.middle_channels + [1],
                                   kernel_sizes = self.kernel_sizes + [1],
                                   pooling_sizes = self.pooling_sizes + [1],
                                   interp_mode = "bilinear")
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
        output = self.model(self.input_tensor)
        
        # Check if the output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.model.n_channels, self.nrows, self.ncols))

if __name__ == "__main__":
    unittest.main()