import unittest
import torch
import numpy as np
import sys
import os

# Add the parent directory to the system path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from losses import DINCAE1Loss

class TestDINCAE1Loss(unittest.TestCase):
    def setUp(self):
        self.nan_placeholder = -2.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = DINCAE1Loss(nan_placeholder=self.nan_placeholder).to(self.device)
        
    def test_basic_loss_calculation(self):
        """Test loss calculation with simple known values"""
        # Create simple inputs where we can compute the expected loss manually
        
        # Prediction: [mean, stdev] for each pixel
        prediction = torch.tensor([[
            [[1.0, 2.0], [3.0, 4.0]],    # mean predictions
            [[0.5, 1.0], [1.5, 2.0]]     # stdev predictions
        ]], dtype=torch.float32, device=self.device)
        
        # Target: only mean values
        target = torch.tensor([[
            [[1.5, 2.5], [3.5, 4.5]]     # true means
        ]], dtype=torch.float32, device=self.device)
        
        # Mask: all valid (no masks, no NaNs)
        masks = torch.zeros_like(target, dtype=torch.bool, device=self.device)
        
        # Compute loss
        loss = self.loss_fn(prediction, target, masks)
        
        # Manually compute expected loss
        # For each pixel: 0.5*[(mean_pred - mean_true)/stdev_pred]^2 + log(stdev_pred^2)
        pixel1 = ((prediction[0, 0, 0, 0] - target[0, 0, 0, 0]) / prediction[0, 1, 0, 0])**2 + torch.log(prediction[0, 1, 0, 0]**2)
        pixel2 = ((prediction[0, 0, 0, 1] - target[0, 0, 0, 1]) / prediction[0, 1, 0, 1])**2 + torch.log(prediction[0, 1, 0, 1]**2)
        pixel3 = ((prediction[0, 0, 1, 0] - target[0, 0, 1, 0]) / prediction[0, 1, 1, 0])**2 + torch.log(prediction[0, 1, 1, 0]**2)
        pixel4 = ((prediction[0, 0, 1, 1] - target[0, 0, 1, 1]) / prediction[0, 1, 1, 1])**2 + torch.log(prediction[0, 1, 1, 1]**2)
        expected_loss = (pixel1 + pixel2 + pixel3 + pixel4) / (2*4)  # 4 valid pixels
        
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=6)
        
    def test_nan_handling(self):
        """Test that NaN placeholder values are properly ignored"""
        
        prediction = torch.tensor([[
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 1.0], [1.5, 2.0]]
        ]], dtype=torch.float32, device=self.device)
        
        # Set one pixel to NaN placeholder
        target = torch.tensor([[
            [[1.5, self.nan_placeholder], [3.5, 4.5]]  # second pixel is "NaN"
        ]], dtype=torch.float32, device=self.device)
        
        # No additional masks
        masks = torch.zeros_like(target, dtype=torch.bool, device=self.device)
        
        loss = self.loss_fn(prediction, target, masks)
        
        # Only 3 valid pixels (one was NaN)
        pixel1 = ((prediction[0, 0, 0, 0] - target[0, 0, 0, 0]) / prediction[0, 1, 0, 0])**2 + torch.log(prediction[0, 1, 0, 0]**2)
        pixel3 = ((prediction[0, 0, 1, 0] - target[0, 0, 1, 0]) / prediction[0, 1, 1, 0])**2 + torch.log(prediction[0, 1, 1, 0]**2)
        pixel4 = ((prediction[0, 0, 1, 1] - target[0, 0, 1, 1]) / prediction[0, 1, 1, 1])**2 + torch.log(prediction[0, 1, 1, 1]**2)
        expected_loss = (pixel1 + pixel3 + pixel4) / (2*3)
        
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=6)
        
    def test_mask_handling(self):
        """Test that masked values are properly ignored"""
        
        prediction = torch.tensor([[
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 1.0], [1.5, 2.0]]
        ]], dtype=torch.float32, device=self.device)
        
        target = torch.tensor([[
            [[1.5, 2.5], [3.5, 4.5]]
        ]], dtype=torch.float32, device=self.device)
        
        # Mask one pixel (second one)
        masks = torch.zeros_like(target, dtype=torch.bool, device=self.device)
        masks[0, 0, 0, 1] = True  # mask the second pixel
        
        loss = self.loss_fn(prediction, target, masks)
        
        # Only 3 valid pixels (one was masked)
        pixel1 = ((prediction[0, 0, 0, 0] - target[0, 0, 0, 0]) / prediction[0, 1, 0, 0])**2 + torch.log(prediction[0, 1, 0, 0]**2)
        pixel3 = ((prediction[0, 0, 1, 0] - target[0, 0, 1, 0]) / prediction[0, 1, 1, 0])**2 + torch.log(prediction[0, 1, 1, 0]**2)
        pixel4 = ((prediction[0, 0, 1, 1] - target[0, 0, 1, 1]) / prediction[0, 1, 1, 1])**2 + torch.log(prediction[0, 1, 1, 1]**2)
        expected_loss = (pixel1 + pixel3 + pixel4) / (2*3)
        
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=6)
        
    def test_all_invalid(self):
        """Test case where all pixels are invalid (masked or NaN)"""
        
        prediction = torch.tensor([[
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 1.0], [1.5, 2.0]]
        ]], dtype=torch.float32, device=self.device)
        
        # All pixels are NaN placeholders
        target = torch.full((1, 1, 2, 2), self.nan_placeholder, dtype=torch.float32, device=self.device)
        
        # No additional masks
        masks = torch.zeros_like(target, dtype=torch.bool, device=self.device)
        
        loss = self.loss_fn(prediction, target, masks)
        
        # Should return 0.0 when no valid pixels
        self.assertEqual(loss.item(), 0.0)
        
    def test_gradient_flow(self):
        """Test that gradients can flow back through the loss"""
        
        prediction = torch.tensor([[
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, 1.0], [1.5, 2.0]]
        ]], dtype=torch.float32, device=self.device, requires_grad=True)
        
        target = torch.tensor([[
            [[1.5, 2.5], [3.5, 4.5]]
        ]], dtype=torch.float32, device=self.device)
        
        masks = torch.zeros_like(target, dtype=torch.bool, device=self.device)
        
        loss = self.loss_fn(prediction, target, masks)
        loss.backward()
        
        # Check gradients exist and are not all zero
        self.assertTrue(prediction.grad is not None)
        self.assertFalse(torch.all(prediction.grad == 0))

if __name__ == '__main__':
    unittest.main()