import unittest
import sys
import os
import torch as th

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.mask_data import mask_image

class TestMaskImage(unittest.TestCase):
    def test_image_masking(self):
        images = th.rand(1, 3, 128, 128)
        mask = th.ones(1, 128, 128)
        mask[:, :3, :3] = 0
        masked_images = mask_image(images, mask, placeholder = -1)
        self.assertEqual(masked_images.shape, images.shape)
        self.assertTrue(th.all(masked_images[0, :, :3, :3] == -1))
