import unittest
import torch as th
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from preprocessing.mask_data import apply_mask_on_channel

class TestApplyMaskOnChannel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.height = 4
        self.width = 4
        self.placeholder = -1.0

        # Create dummy images (batch_size, channels, height, width)
        self.images = th.tensor([
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
             [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]],
             [[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]],

            [[[49, 50, 51, 52], [53, 54, 55, 56], [57, 58, 59, 60], [61, 62, 63, 64]],
             [[65, 66, 67, 68], [69, 70, 71, 72], [73, 74, 75, 76], [77, 78, 79, 80]],
             [[81, 82, 83, 84], [85, 86, 87, 88], [89, 90, 91, 92], [93, 94, 95, 96]]]
        ], dtype=th.float32)

        # Create a mask (batch_size, channels, height, width)
        self.masks = th.tensor([
            [[[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 1]],
             [[0, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]],
             [[1, 1, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 1]]],

            [[[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]],
             [[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0], [0, 1, 0, 1]],
             [[0, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]]]
        ], dtype=th.float32)

    def test_apply_mask_on_channel(self):
        masked_images = apply_mask_on_channel(self.images, self.masks, self.placeholder)

        # Ensure masked values are replaced with the placeholder
        self.assertTrue(th.all(masked_images[self.masks == 0] == self.placeholder))

        # Ensure unmasked values remain the same
        self.assertTrue(th.all(masked_images[self.masks == 1] == self.images[self.masks == 1]))

    def test_apply_mask_with_channel_mean(self):
        masked_images = apply_mask_on_channel(self.images, self.masks, None)

        # Compute expected per-channel mean ignoring masked pixels
        means = (self.images * self.masks).sum(dim=(2, 3), keepdim=True) / (self.masks.sum(dim=(2, 3), keepdim=True))
        means = means.expand(-1, -1, self.height, self.width)

        # Ensure masked values are replaced with per-channel mean
        self.assertTrue(th.allclose(masked_images[self.masks == 0], means[self.masks == 0]))

    def test_batch_independence(self):
        masked_images_1 = apply_mask_on_channel(self.images[0:1], self.masks[0:1], self.placeholder)
        masked_images_2 = apply_mask_on_channel(self.images[1:], self.masks[1:], self.placeholder)

        combined = th.cat([masked_images_1, masked_images_2], dim=0)
        full_masked = apply_mask_on_channel(self.images, self.masks, self.placeholder)

        # Ensure processing batches separately gives the same result as processing them together
        self.assertTrue(th.all(combined == full_masked))

if __name__ == '__main__':
    unittest.main()