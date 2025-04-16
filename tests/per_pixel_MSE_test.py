import torch as th
import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from losses import PerPixelMSE

class TestPerPixelLoss(unittest.TestCase):
    
    def setUp(self):
        self.loss_function = PerPixelMSE()
        
    def test_no_masking(self):
        """Test case where no pixels are masked (all pixels are valid)."""
        prediction = th.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        target = th.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        mask = th.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])  # No masking
        loss = self.loss_function(prediction, target, mask)
        expected_loss = 0.0  # Prediction and target are identical
        self.assertEqual(loss, expected_loss)

    def test_full_masking(self):
        """Test case where all pixels are masked (no valid pixels)."""
        prediction = th.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        target = th.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
        mask = th.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])  # All pixels masked
        loss = self.loss_function(prediction, target, mask)
        expected_loss = 0.0  # No valid pixels, so loss should be 0
        self.assertEqual(loss, expected_loss)

    def test_partial_masking(self):
        """Test case where some pixels are masked."""
        prediction = th.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        target = th.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
        mask = th.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])  # Mask some pixels
        loss = self.loss_function(prediction, target, mask)
        # Only unmasked pixels are (0, 0) and (1, 1)
        # Differences: (1.0 - 1.0)**2 = 0.0 and (4.0 - 1.0)**2 = 9.0
        # Mean loss: (0.0 + 9.0) / 2 = 4.5
        expected_loss = th.tensor(4.5)
        self.assertTrue(th.allclose(loss, expected_loss))

    def test_random_inputs(self):
        """Test case with random inputs."""
        prediction = th.rand(1, 3, 256, 256)  # Random prediction tensor
        target = th.rand(1, 3, 256, 256)    # Random target tensor
        mask = th.randint(0, 2, (1, 3, 256, 256)).float()  # Random binary mask
        loss = self.loss_function(prediction, target, mask)
        # Ensure the loss is a scalar tensor
        self.assertEqual(loss.shape, ())
        # Ensure the loss is non-negative
        self.assertTrue(loss >= 0.0)
        
    def test_requires_grad(self):
        """Test that the loss requires gradients."""
        prediction = th.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
        target = th.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        mask = th.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
        loss = self.loss_function(prediction, target, mask)
        # Ensure the loss requires gradients
        self.assertTrue(loss.requires_grad)

if __name__ == "__main__":
    unittest.main()