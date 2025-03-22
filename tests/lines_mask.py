import unittest
import torch as th
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from preprocessing.mask_data import create_lines_mask

class TestCreateLinesMask(unittest.TestCase):
    
    def test_mask_shape(self):
        """Test that the mask has the correct shape."""
        width, height = 100, 100
        mask = create_lines_mask(width, height, min_thickness=1, max_thickness=3, n_lines=5)
        self.assertEqual(mask.shape, (height, width))
    
    def test_mask_values(self):
        """Test that the mask contains only 0s and 1s."""
        width, height = 100, 100
        mask = create_lines_mask(width, height, min_thickness=1, max_thickness=3, n_lines=5)
        self.assertTrue(th.all((mask == 0) | (mask == 1)))
    
    def test_lines_are_drawn(self):
        """Test that lines are actually drawn on the mask."""
        width, height = 100, 100
        mask = create_lines_mask(width, height, min_thickness=1, max_thickness=3, n_lines=5)
        # Check that there are some zeros in the mask (indicating lines)
        self.assertTrue(th.any(mask == 0))
    
    def test_no_lines(self):
        """Test that no lines are drawn when n_lines is 0."""
        width, height = 100, 100
        mask = create_lines_mask(width, height, min_thickness=1, max_thickness=3, n_lines=0)
        # The mask should be all ones
        self.assertTrue(th.all(mask == 1))
    
    def test_thickness_range(self):
        """Test that the thickness of the lines is within the specified range."""
        width, height = 100, 100
        min_thickness, max_thickness = 2, 5
        mask = create_lines_mask(width, height, min_thickness, max_thickness, n_lines=5)
        # Convert mask to numpy for easier manipulation
        mask_np = mask.numpy()
        # Find contours of the lines
        contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Check that the thickness is within the specified range
            self.assertTrue(min_thickness <= w <= max_thickness or min_thickness <= h <= max_thickness)

if __name__ == '__main__':
    unittest.main()