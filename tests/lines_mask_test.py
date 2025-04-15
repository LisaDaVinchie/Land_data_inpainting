import unittest
import torch as th
from tempfile import NamedTemporaryFile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from preprocessing.mask_data import LinesMask

class TestLinesMask(unittest.TestCase):
    
    def setUp(self):
        self.params_path = NamedTemporaryFile(delete=False)
        self.image_nrows = 10
        self.image_ncols = 10
        self.num_lines = 5
        self.min_thickness = 1
        self.max_thickness = 3
        
        self.params = {
            "lines_mask": {
                "num_lines": self.num_lines,
                "min_thickness": self.min_thickness,
                "max_thickness": self.max_thickness
            },
            "dataset": {
                "cutted_nrows": self.image_nrows,
                "cutted_ncols": self.image_ncols
            }
        }
        
        # Create a LinesMask instance
        self.lines_mask = LinesMask(params_path=self.params_path)
        
    def tearDown(self):
        # Remove the temporary file
        os.remove(self.params_path.name)
    
    def test_initialization(self):
        """Test if the LinesMask class initializes correctly."""
        self.assertEqual(self.lines_mask.image_nrows, self.image_nrows)
        self.assertEqual(self.lines_mask.image_ncols, self.image_ncols)
        self.assertEqual(self.lines_mask.num_lines, self.num_lines)
        self.assertEqual(self.lines_mask.min_thickness, self.min_thickness)
        self.assertEqual(self.lines_mask.max_thickness, self.max_thickness)
        

if __name__ == '__main__':
    unittest.main()