import unittest
import sys
import os
import torch as th
from pathlib import Path
import json
from tempfile import NamedTemporaryFile

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.models import simple_conv, conv_unet, conv_maxpool

class TestCNNModel(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary JSON file with reward parameters."""
        self.model_params = {
            "simple_conv": {
            "middle_channels": [64, 128, 256],
            "kernel_size": [3, 3, 3],
            "stride": [2, 2, 2],
            "padding": [1, 1, 1],
            "output_padding": [1, 1, 1]
            },
            "dataset_params":{
                "n_channels": 1
            }      
        }
        
        
        # Create a temporary file
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.model_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        self.model = simple_conv(params_path=self.params_path)
        
        self.batch_size = 2
        self.input_tensor = th.rand(self.batch_size, self.model.n_channels, 64, 64)
        
    
    def tearDown(self):
        """Delete the temporary JSON file after tests."""
        Path(self.temp_json.name).unlink()
        
    def test_model_initialization(self):
        """Test if the model initializes correctly"""
        # Test if the model initializes correctly
        
        # Check if the model has the correct attributes
        self.assertEqual(self.model.n_channels, 1)
        self.assertEqual(self.model.middle_channels, [64, 128, 256])
        self.assertEqual(self.model.kernel_size, [3, 3, 3])
        self.assertEqual(self.model.stride, [2, 2, 2])
        self.assertEqual(self.model.padding, [1, 1, 1])
        self.assertEqual(self.model.output_padding, [1, 1, 1])
        
    def test_forward_pass(self):
        """Test if the forward pass of the model works correctly"""
        # Test the forward pass of the model
        output = self.model(self.input_tensor)
        
        # Check if the output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.model.n_channels, 64, 64))

if __name__ == "__main__":
    unittest.main()