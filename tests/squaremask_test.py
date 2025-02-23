import unittest
import sys
import os
import torch as th
from pathlib import Path
import json
from tempfile import NamedTemporaryFile

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.mask_data import SquareMask

class TestSquareMask(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary JSON file with reward parameters."""
        self.model_params = {
            "dataset":{
                "image_width": 128,
                "image_height": 128
            },
            "square_mask": {
                "mask_percentage": 0.5
            }   
        }
        
        
        # Create a temporary file
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        
        json.dump(self.model_params, self.temp_json)
            
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        self.mask_class = SquareMask(self.params_path)
        
        
    def test_model_initialization(self):
        self.assertEqual(self.mask_class.image_width, 128)
        self.assertEqual(self.mask_class.image_height, 128)
        self.assertEqual(self.mask_class.mask_percentage, 0.5)
        
    # def tearDown(self):
    #     """Delete the temporary JSON file after tests."""
    #     Path(self.temp_json.name).unlink()

if __name__ == "__main__":
    unittest.main()