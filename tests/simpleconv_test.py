import unittest
import sys
import os
import torch as th

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.models import simple_conv

class Test_simpleconv_model(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary JSON file with reward parameters."""
        
        self.n_channels = 1
        self.ncols = 64
        self.nrows = 64
        self.middle_channels = [64, 128, 256]
        self.kernel_sizes = [3, 3, 3]
        self.stride = [2, 2, 2]
        self.padding = [1, 1, 1]
        self.output_padding = [1, 1, 1]
        
        self.batch_size = 2
        self.model_params = {
            "models": {
                "simple_conv": {
                    "middle_channels": self.middle_channels,
                    "kernel_size": self.kernel_sizes,
                    "stride": self.stride,
                    "padding": self.padding,
                    "output_padding": self.output_padding
                    }
            },  
            "dataset":{
                "dataset_kind": "test",
                "test": {
                    "n_channels": self.n_channels,
                }
            }      
        }
        
        self.model = simple_conv(params=self.model_params)
        
        self.input_tensor = th.rand(self.batch_size, self.model.n_channels, 64, 64)
        self.masks = th.ones(self.batch_size, self.model.n_channels, 64, 64)
        self.masks[:, 0, 0:10, 0:10] = 0.0
        
    def test_model_initialization(self):
        """Test if the model initializes correctly"""
        # Test if the model initializes correctly
        
        # Check if the model has the correct attributes
        self.assertEqual(self.model.n_channels, self.n_channels)
        self.assertEqual(self.model.middle_channels, self.middle_channels)
        self.assertEqual(self.model.kernel_size, self.kernel_sizes)
        self.assertEqual(self.model.stride, self.stride)
        self.assertEqual(self.model.padding, self.padding)
        self.assertEqual(self.model.output_padding, self.output_padding)
        
    def test_forward_pass(self):
        """Test if the forward pass of the model works correctly"""
        # Test the forward pass of the model
        output = self.model(self.input_tensor, self.masks)
        
        # Check if the output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.model.n_channels, 64, 64))

if __name__ == "__main__":
    unittest.main()