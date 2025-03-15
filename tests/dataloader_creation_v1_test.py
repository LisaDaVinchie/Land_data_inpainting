import unittest
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils.create_dataloaders_v1 import create_dataloaders

class TestCreateDataloaders(unittest.TestCase):
    def setUp(self):
        # np.random.seed(42)
        # torch.manual_seed(42)

        # Create dummy dataset (100 samples, each with 3x64x64 shape)
        self.dataset = np.random.rand(100, 3, 64, 64).astype(np.float32)
        self.masks = np.random.randint(0, 2, (100, 64, 64), dtype=np.uint8)

    def test_dataloaders_split(self):
        train_perc = 0.8
        batch_size = 10

        train_loader, test_loader = create_dataloaders(self.dataset, self.masks, train_perc, batch_size)

        # Check train-test split sizes
        train_size = int(train_perc * len(self.dataset))
        test_size = len(self.dataset) - train_size

        self.assertEqual(len(train_loader.dataset), train_size)
        self.assertEqual(len(test_loader.dataset), test_size)

    def test_batch_shapes(self):
        train_loader, test_loader = create_dataloaders(self.dataset, self.masks, train_perc=0.8, batch_size=10)

        for images, masks in train_loader:
            self.assertEqual(images.shape, (10, 3, 64, 64))
            self.assertEqual(masks.shape, (10, 64, 64))
            break  # Check only first batch

        for images, masks in test_loader:
            self.assertEqual(images.shape, (10, 3, 64, 64))
            self.assertEqual(masks.shape, (10, 64, 64))
            break  # Check only first batch

if __name__ == '__main__':
    unittest.main()
