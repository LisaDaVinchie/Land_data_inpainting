import unittest
import torch as th
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from metrics import PCC

class TestPCC(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.nan_placeholder = -2.0
        self.pcc = PCC(self.nan_placeholder)
        
        # # Test case with no correlation
        # self.x3 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        # self.y3 = torch.tensor([[[[4.0, 3.0], [2.0, 1.0]]]])
        
        # # Test case with batch and channel dimensions
        # self.x4 = torch.tensor([
        #     [[[1.0, 2.0], [3.0, 4.0]],  # Channel 0
        #      [[0.5, 1.5], [2.5, 3.5]]], # Channel 1
        #     [[[2.0, 4.0], [6.0, 8.0]],  # Channel 0
        #      [-0.5, -1.5], [-2.5, -3.5]]] # Channel 1
        #                        ])  # shape (2, 2, 2, 2)
        
        # self.y4 = torch.tensor([
        #     [[[2.0, 4.0], [6.0, 8.0]],  # Channel 0 (x4*2)
        #      [[1.0, 3.0], [5.0, 7.0]]],  # Channel 1 (x4*2)
        #     [[[-2.0, -4.0], [-6.0, -8.0]],  # Channel 0 (x4*-1)
        #      [0.5, 1.5], [2.5, 3.5]]]  # Channel 1 (x4*-1)
        # ])
        
        # # Random test case
        # torch.manual_seed(42)
        # self.x5 = torch.randn(3, 2, 4, 4)  # shape (3, 2, 4, 4)
        # self.y5 = self.x5 * 1.5 + 0.3  # Linear transformation
        
    def test_get_nan_mask(self):
        target = th.tensor([[[[1.0, self.nan_placeholder, self.nan_placeholder], [self.nan_placeholder, self.nan_placeholder, 6.0]]]])
        expected_mask = th.tensor([[[[False, True, True], [True, True, False]]]], dtype=th.bool)
        
        nan_mask = self.pcc._get_nan_mask(target)
        self.assertEqual(nan_mask.shape, expected_mask.shape)
        self.assertTrue(th.all(nan_mask == expected_mask))
        
    def test_get_valid_mask(self):
        masks = th.tensor([[[[True, True, False, False],
                             [True, True, False, False],
                             [True, True, True, True],
                             [True, True, True, True],
                             ]]], dtype=th.bool)
        
        nan_mask = th.tensor([[[[False, False, False, False],
                                [False, True, True, False],
                                [False, True, True, False],
                                [False, False, False, False]
                                ]]], dtype=th.bool)
        
        expected_mask = th.tensor([[[[False, False, True, True],
                                     [False, False, False, True],
                                     [False, False, False, False],
                                     [False, False, False, False]
                                     ]]], dtype=th.bool)
        
        valid_mask = self.pcc._get_valid_mask(masks, nan_mask)
        self.assertEqual(valid_mask.shape, expected_mask.shape)
        self.assertTrue(th.all(valid_mask == expected_mask))
        
    def test_perfect_positive_correlation_no_mask(self):
        """Test perfect positive correlation (PCC=1)"""
        # Simple test case - perfect correlation
        image = th.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # shape (1, 1, 2, 2)
        target = image * 2.0
        mask = th.tensor([[[[False, False], [False, False]]]], dtype=th.bool)
        pcc = self.pcc(image, target, mask)
        self.assertEqual(pcc.shape, (1, 1))
        self.assertAlmostEqual(pcc.item(), 1.0, places=6)
        
    def test_perfect_positive_correlation_mask(self):
        """Test perfect positive correlation (PCC=1)"""
        # Simple test case - perfect correlation
        image = th.tensor([[[[1.0, 2.0, 3.0, 4.0],
                             [5.0, self.nan_placeholder, 7.0, 8.0],
                             [9.0, self.nan_placeholder, 11.0, 12.0],
                             [13.0, 14.0, 15.0, 16.0]]]])
        
        target = th.tensor([[[[1.0, 2.0, 3.0, 4.0],
                             [5.0, self.nan_placeholder, 14.0, 16.0],
                             [9.0, self.nan_placeholder, 22.0, 24.0],
                             [13.0, 14.0, 15.0, 16.0]]]])
        mask = th.tensor([[[[False, False, False, False],
                             [False, True, True, True],
                             [False, True, True, True],
                             [False, False, False, False]]]], dtype=th.bool)
        pcc = self.pcc(image, target, mask)
        self.assertEqual(pcc.shape, (1, 1))
        self.assertAlmostEqual(pcc.item(), 1.0, places=6)
    
    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation (PCC=-1)"""
        image = th.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # shape (1, 1, 2, 2)
        target = image * (-2.0)  # image * -2
        mask = th.tensor([[[[False, False], [False, False]]]], dtype=th.bool)
        pcc = self.pcc(image, target, mask)
        self.assertEqual(pcc.shape, (1, 1))
        self.assertAlmostEqual(pcc.item(), -1.0, places=6)
    
    # def test_no_correlation_no_mask(self):
    #     """Test no correlation (PCC≈0)"""
    #     shape = (1, 1, 2, 2)
    #     image = th.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # shape (1, 1, 2, 2)
    #     target = th.tensor([[[[9.0, 8.0], [7.0, 6.0]]]])  # shape (1, 1, 2, 2)
    #     mask = th.tensor([[[[False, False], [False, False]]]], dtype=th.bool)
        
    #     pcc = self.pcc(image, target, mask)
    #     print(f"Pearson Correlation Coefficient: {pcc.item()}")
    #     self.assertEqual(pcc.shape, (1, 1))
    #     self.assertTrue(abs(pcc.item()) < 0.1)
    
    def test_batch_channel_handling(self):
        """Test handling of batch and channel dimensions"""
         # # Test case with batch and channel dimensions
        image = th.tensor([[[[1.0, 2.0], [3.0, 4.0]],  # Channel 0
                            [[0.5, 1.5], [2.5, 3.5]]], # Channel 1
                           [[[2.0, 4.0], [6.0, 8.0]],  # Channel 0
                            [[-0.5, -1.5], [-2.5, -3.5]]]# Channel 1
                           ])  # shape (2, 2, 2, 2)
        
        target = th.tensor([[[[2.0, 4.0],
                               [6.0, 8.0]],  # Channel 0 (x4*2)
                              [[1.0, 3.0],
                               [5.0, 7.0]]],  # Channel 1 (x4*2)
                             [[[-2.0, -4.0],
                               [-6.0, -8.0]],  # Channel 0 (x4*-1)
                              [[0.5, 1.5], [2.5, 3.5]]]  # Channel 1 (x4*-1)
                            ])
        mask = th.tensor([[[[False, False],
                             [False, False]],  # Channel 0
                            [[False, False],
                             [False, False]]],  # Channel 1
                           [[[False, False],
                             [False, False]],  # Channel 0
                            [[False, False],
                             [False, False]]]])
        pcc = self.pcc(image, target, mask)
        self.assertEqual(pcc.shape, (2, 2))
        
        # Check first batch
        self.assertAlmostEqual(pcc[0, 0].item(), 1.0, places=6)  # Channel 0
        self.assertAlmostEqual(pcc[0, 1].item(), 1.0, places=6)  # Channel 1
        
        # Check second batch
        self.assertAlmostEqual(pcc[1, 0].item(), -1.0, places=6)  # Channel 0
        self.assertAlmostEqual(pcc[1, 1].item(), -1.0, places=6)  # Channel 1

if __name__ == '__main__':
    unittest.main()