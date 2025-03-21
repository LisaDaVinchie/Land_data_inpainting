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
        self.model_params = {
            "DINCAE_like": {
                "middle_channels": [16, 30, 58, 110, 209],
                "kernel_sizes": [3, 3, 3, 3, 3],
                "pooling_sizes": [2, 2, 2, 2, 2],
                "interp_mode": "bilinear"
                },  
            "dataset":{
                "n_channels": 10,
                "cutted_width": 168,
                "cutted_height": 144
            }      
        }
        
        
        # Create a temporary file
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.model_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        self.model = DINCAE_like(params_path=self.params_path)
        
        self.batch_size = 32
        self.width = 168
        self.height = 144
        self.input_tensor = th.rand(self.batch_size, self.model.n_channels, self.width, self.height)
        
    
    def tearDown(self):
        """Delete the temporary JSON file after tests."""
        Path(self.temp_json.name).unlink()
        
    def test_model_initialization(self):
        """Test if the model initializes correctly"""
        # Test if the model initializes correctly
        
        # Check if the model has the correct attributes
        self.assertEqual(self.model.n_channels, 10)
        self.assertEqual(self.model.middle_channels, [16, 30, 58, 110, 209])
        self.assertEqual(self.model.kernel_sizes, [3, 3, 3, 3, 3])
        self.assertEqual(self.model.pooling_sizes, [2, 2, 2, 2, 2])
        self.assertEqual(self.model.image_width, 168)
        self.assertEqual(self.model.image_height, 144)
        self.assertEqual(self.model.interp_mode, "bilinear")
        
    def test_forward_pass(self):
        """Test if the forward pass of the model works correctly"""
        # Test the forward pass of the model
        output = self.model(self.input_tensor)
        
        # Check if the output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.model.n_channels, self.width, self.height))

if __name__ == "__main__":
    unittest.main()