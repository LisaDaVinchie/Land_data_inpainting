import unittest
from tempfile import NamedTemporaryFile
import os
import sys
import json
import torch as th
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from preprocessing.mask_data import LinesMask

class TestLinesMask(unittest.TestCase):
    
    def setUp(self):
        self.image_nrows = 10
        self.image_ncols = 10
        self.num_lines = 5
        self.min_thickness = 1
        self.max_thickness = 3
        
        self.params = {
            "masks": {
                "lines": {
                    "num_lines": self.num_lines,
                    "min_thickness": self.min_thickness,
                    "max_thickness": self.max_thickness
                }
            },
            "dataset": {
                "cutted_nrows": self.image_nrows,
                "cutted_ncols": self.image_ncols
            }
        }
        
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.params, self.temp_json)
        self.temp_json.close()
        self.params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        # Create a LinesMask instance
        self.lines_mask = LinesMask(params_path=self.params_path)
        
    def tearDown(self):
        # Remove the temporary file
        os.remove(self.temp_json.name)
    
    def test_initialization(self):
        """Test if the LinesMask class initializes correctly."""
        self.assertEqual(self.lines_mask.image_nrows, self.image_nrows)
        self.assertEqual(self.lines_mask.image_ncols, self.image_ncols)
        self.assertEqual(self.lines_mask.num_lines, self.num_lines)
        self.assertEqual(self.lines_mask.min_thickness, self.min_thickness)
        self.assertEqual(self.lines_mask.max_thickness, self.max_thickness)
        
    def test_invalid_params(self):
        """Test if the LinesMask class raises an error with invalid parameters."""
        with self.assertRaises(ValueError):
            LinesMask(params_path=self.params_path, num_lines=-1)
        
        with self.assertRaises(ValueError):
            LinesMask(params_path=self.params_path, min_thickness=-2)
        
        with self.assertRaises(ValueError):
            LinesMask(params_path=self.params_path, max_thickness=0)
            
        with self.assertRaises(ValueError):
            LinesMask(params_path=self.params_path, min_thickness=5, max_thickness=3)
        
    def test_output_shape(self):
        """Test if the output shape of the mask is correct."""
        mask = self.lines_mask.mask()
        self.assertEqual(mask.shape, (self.image_nrows, self.image_ncols))
        
    def test_mask_dtype(self):
        """Test if the mask data type is correct."""
        mask = self.lines_mask.mask()
        self.assertEqual(mask.dtype, th.bool)
        

if __name__ == '__main__':
    unittest.main()