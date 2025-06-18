import unittest
import torch as th
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from train5 import TrainModel
from losses import PerPixelMSE
from models import DINCAE_pconvs, DummierModel

class TestTrainingScript(unittest.TestCase):
    def setUp(self):
        loss_function = PerPixelMSE()
        model = DummierModel()
        optimizer = th.optim.Adam(model.parameters(), lr=0.0001)
        self.train = TrainModel(loss_function=loss_function, 
                           model=model, 
                           optimizer=optimizer)
        
    def test_validation_mask(self):
        """Test if the mask is correctly validated."""
        
        mask = th.tensor([
            [True, True, False, True],
            [True, False, True, False],
            [False, True, True, False],
            [True, True, False, False]
        ], dtype=th.bool).unsqueeze(0).unsqueeze(0)  # Add batch dimension
        
        nan_mask = th.tensor([
            [True, True, False, False],
            [False, False, True, True],
            [True, True, False, False],
            [False, False, True, True]
        ], dtype=th.bool).unsqueeze(0).unsqueeze(0)
        
        expected_result = th.tensor([
            [False, False, False, False],
            [False, False, False, True],
            [True, False, False, False],
            [False, False, True, True]
        ], dtype=th.bool).unsqueeze(0).unsqueeze(0)
        
        validation_mask = self.train.validation_mask(mask, nan_mask, False)
        
        self.assertTrue(th.equal(validation_mask, expected_result), "Validation mask does not match expected result.")
        
        inv_validation_mask = self.train.validation_mask(mask, nan_mask)
        
        self.assertTrue(th.equal(inv_validation_mask, ~expected_result), "Inverse validation mask does not match expected result.")
        
    def test_calculate_valid_pixels(self):
        """Test if the number of valid pixels is calculated correctly."""
        
        mask = th.tensor([
            [True, True, False, True],
            [True, False, True, False],
            [False, True, True, False],
            [True, True, False, False]
        ], dtype=th.bool).unsqueeze(0).unsqueeze(0)
        nan_mask = th.tensor([
            [True, True, False, False],
            [False, False, True, True],
            [True, True, False, False],
            [False, False, True, True]
        ], dtype=th.bool).unsqueeze(0).unsqueeze(0)
        
        expected_result = th.tensor([
            [False, False, False, False],
            [False, False, False, True],
            [True, False, False, False],
            [False, False, True, True]
        ], dtype=th.bool).unsqueeze(0).unsqueeze(0)
        
        expected_n_valid_pixels = expected_result.sum().item()
        
        output = self.train.calculate_valid_pixels(mask, nan_mask)
        
        self.assertEqual(output, expected_n_valid_pixels, "Number of valid pixels does not match expected value.")
        
    def test_compute_loss(self):
        """Test if the loss is computed correctly."""
        
        images = th.randn(10, 13, 4, 4)
        masks = (th.randn(10, 13, 4, 4) > 0.5).bool()
        nanmasks = (th.randn(10, 13, 4, 4) > 0.5).bool()
        
        loss = self.train._compute_loss(images, masks, nanmasks)
        
        self.assertIsInstance(loss, th.Tensor, "Loss should be a tensor.")
        self.assertEqual(loss.shape, (), "Loss should be a scalar tensor.")
        self.assertTrue(loss.requires_grad, "Loss should require gradient for backpropagation.")
        
        # Test output with dummy data
        images = th.ones(1, 13, 4, 4) * 2
        masks = th.ones(1, 13, 4, 4, dtype=th.bool)
        masks[0, 4, 0:2, 0:2] = False
        nanmasks = th.ones(1, 13, 4, 4, dtype=th.bool)
        
        # MSE for 4 pixels with value 2
        expected_loss = 4 * 4
        
        loss = self.train._compute_loss(images, masks, nanmasks)
        
        self.assertAlmostEqual(loss.item(), expected_loss, places=5)

if __name__ == '__main__':
    unittest.main()