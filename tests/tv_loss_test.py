import torch as th
import unittest

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from losses import TotalVariationLoss

class TestTVLoss(unittest.TestCase):
    
    def setUp(self):
        self.nan_placeholder = -2.0
        self.tvloss = TotalVariationLoss(nan_placeholder=self.nan_placeholder)
    
    def test_dilation(self):
        sample_mask = th.tensor([[[
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1]
        ]]], dtype=th.bool)
        
        expected_mask = th.tensor([[[
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ]]], dtype=th.bool)
        
        dilated_mask = self.tvloss._dilate_mask(sample_mask, 1)
        self.assertEqual(dilated_mask.dtype, th.bool)
        
        self.assertTrue(th.equal(dilated_mask, expected_mask))
        inv_dilated_mask = self.tvloss._dilate_mask(sample_mask, 1, True)
        self.assertTrue(th.equal(inv_dilated_mask, ~expected_mask))
        self.assertEqual(inv_dilated_mask.dtype, th.bool)
        
    def test_image_composition(self):
        """Test that the image is composed correctly"""
        
        pred = th.tensor([[[
            [1., 2., 3., 1.],
            [4., 5., 6., 1.],
            [7., 8., 9., 1.],
            [1., 1., 1., 1.]
        ]]])
        target = th.tensor([[[
            [2., 2., 2., 2.],
            [2., 2., 2., 2.],
            [2., 2., 2., 2.],
            [2., 2., 2., 2.]
        ]]])
        masks = th.tensor([[[
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, False],
            [False, False, False, False]
        ]]], dtype=th.bool)
        
        expected_image = th.tensor([[[
            [1., 2., 3., 2.],
            [4., 5., 6., 2.],
            [7., 8., 9., 2.],
            [2., 2., 2., 2.]
        ]]])
        
        composed_image = self.tvloss._compose_image(pred, target, masks)
        
        self.assertTrue(th.allclose(composed_image, expected_image))
    
    def test_nan_exlcusion(self):
        image = th.tensor([[[
            [self.nan_placeholder, self.nan_placeholder, 3., 4.],
            [self.nan_placeholder, self.nan_placeholder, 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]], dtype=th.float32)
        mask = th.tensor([[[
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ]]], dtype=th.bool)
        
        expected_mask = th.tensor([[[
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ]]], dtype=th.bool)
        
        output_mask = self.tvloss._exclude_nans_from_mask(image, mask)
        
        self.assertTrue(th.equal(output_mask, expected_mask))
    
    def test_tv_loss_no_nans(self):
        image = th.tensor([[[
            [1., 2., 3., 2.],
            [4., 5., 6., 2.],
            [7., 8., 9., 2.],
            [2., 2., 2., 2.]
        ]]], dtype=th.float32)
        
        masks = th.tensor([[[
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0]
        ]]], dtype=th.bool)
        
        loss = self.tvloss._tv_loss(image, masks)
        
        # The expected loss is calculated as:
        # (2 - 1 + 3 - 2 + 5 - 4 + 6 - 5 + 8 - 7 + 9 - 8 + 
        # + 4 - 1 + 7 - 4 + 8 - 5 + 5 - 2 + 9 - 6 + 6 - 3) / 12
        # = (6 + 3 * 6) / 12 = 24 / 12 = 2.0  
        
        self.assertAlmostEqual(loss.item(), 2.0, places=4)
        
    def test_all_masked_smooth(self):
        """Test when all pixels are masked (= evaluated), but there are no variations"""
        B, C, H, W = 1, 3, 4, 4
        pred = th.ones(B, C, H, W, requires_grad=True)
        target = th.zeros(B, C, H, W)
        masks = th.zeros((B, C, H, W), dtype=th.bool)  # All masked => loss calculated on all pixels
        
        loss = self.tvloss(pred, target, masks)
        
        self.assertAlmostEqual(loss.item(), 0.0, places=4)
        self.assertTrue(loss.requires_grad)
        
    def test_no_masked_pixels(self):
        """Test when no pixels are masked"""
        B, C, H, W = 1, 3, 6, 6
        pred = th.rand(B, C, H, W)
        target = th.rand(B, C, H, W)
        masks = th.ones((B, C, H, W), dtype = th.bool)  # None masked => loss calculated on no pixels
        
        loss = self.tvloss(pred, target, masks)
        
        self.assertEqual(loss.item(), 0.0)
        
    def test_partial_mask(self):
        """Test with partial masking"""
        pred = th.tensor([[[
            [1., 2., 3., 1.],
            [4., 5., 6., 1.],
            [7., 8., 9., 1.],
            [1., 1., 1., 1.]
        ]]])
        target = th.tensor([[[
            [2., 2., 2., 2.],
            [2., 2., 2., 2.],
            [2., 2., 2., 2.],
            [2., 2., 2., 2.]
        ]]])
        masks = th.tensor([[[
            [1., 1., 1., 1.],
            [1., 0., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]], dtype=th.bool)
        
        loss = self.tvloss(pred, target, masks)
        
        self.assertAlmostEqual(loss.item(), 2.0, places=4)
    
    def test_border_case(self):
        """Test with partial masking"""
        pred = th.tensor([[[
            [1., 2., 3., 1.],
            [4., 5., 6., 1.],
            [7., 8., 9., 1.],
            [1., 1., 1., 1.]
        ]]])
        target = th.tensor([[[
            [2., 2., 2., 2.],
            [2., 2., 2., 2.],
            [2., 2., 2., 2.],
            [2., 2., 2., 2.]
        ]]])
        masks = th.tensor([[[
            [0., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]], dtype=th.bool)
        
        loss = self.tvloss(pred, target, masks)
        # (2 - 1 + 5 - 4 + 4 - 1 + 5 - 2) / 4 = 8 / 4 = 2
        self.assertAlmostEqual(loss.item(), 2.0, places=4)
        
    def test_gradient_flow(self):
        """Verify gradients can flow back through the loss"""
        B, C, H, W = 1, 1, 3, 3
        pred = th.rand(B, C, H, W, requires_grad=True)
        target = th.rand(B, C, H, W)
        masks = th.zeros((B, C, H, W), dtype=th.bool)  # All masked => loss calculated on all pixels
        
        loss = self.tvloss(pred, target, masks)
        self.assertTrue(loss.requires_grad)
        
        loss.backward()
    
        self.assertIsNotNone(pred.grad)
        self.assertTrue(th.any(pred.grad != 0))

if __name__ == '__main__':
    unittest.main()