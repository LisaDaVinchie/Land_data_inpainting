import unittest
import sys
import os
import torch as th

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.models import simple_conv, initialize_model_and_dataset_kind

class Test_simpleconv_model(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary JSON file with reward parameters."""
        self.channels_to_keep = ["c1", "c2", "c3"]
        self.n_channels = len(self.channels_to_keep) + 1
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
                    "channels_to_keep": self.channels_to_keep
                }
            }      
        }
        
        self.model = simple_conv(params=self.model_params)
        self.model.layers_setup()
        
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
        model, dataset_kind = initialize_model_and_dataset_kind(self.model_params, "simple_conv", dataset_params)
        
        # Check that the network has the correct attributes
        self.assertIsInstance(model, simple_conv)
        self.assertEqual(dataset_kind, "extended")
        self.assertEqual(model.n_channels, self.n_channels + 1)
        
        self.assertEqual(self.model.middle_channels, self.middle_channels)
        self.assertEqual(self.model.kernel_size, self.kernel_sizes)
        self.assertEqual(self.model.stride, self.stride)
        self.assertEqual(self.model.padding, self.padding)
        self.assertEqual(self.model.output_padding, self.output_padding)

if __name__ == "__main__":
    unittest.main()