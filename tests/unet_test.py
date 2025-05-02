import unittest
import sys
import os
import torch as th
import json

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.models import conv_unet

class TestCNNModel(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary JSON file with reward parameters."""
        self.model_params = {
            "models": {
                "conv_unet": {
                    "middle_channels": [12, 12, 12],
                    "kernel_size": [3, 3, 3],
                    "stride": [2, 2, 2],
                    "padding": [1, 1, 1],
                    "output_padding": [1, 1, 1]
                    }
            },
            "dataset": {
                "dataset_kind": "test",
                "test": {
                    "n_channels": 1,
                }
            }    
        }
        
        self.model = conv_unet(params=self.model_params)
        
        self.batch_size = 2
        self.input_tensor = th.rand(self.batch_size, self.model.n_channels, 64, 64)
        
    def test_model_initialization(self):
        """Test if the model initializes correctly"""
        # Test if the model initializes correctly
        
        # Check if the model has the correct attributes
        self.assertEqual(self.model.n_channels, 1)
        self.assertEqual(self.model.middle_channels, [12, 12, 12])
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