import unittest
import torch as th
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from mask_data import apply_mask_on_channel

class TestApplyMaskOnChannel(unittest.TestCase):
    def test_apply_mask(self):
        images = th.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]])
        masks = th.tensor([[[[1, 0], [1, 0]], [[0, 1], [0, 1]]],
                           [[[1, 1], [0, 0]], [[0, 0], [1, 1]]]])
        channels = [0, 1]
        placeholder = -1.0
        
        expected_output = th.tensor([[[[1.0, -1.0], [3.0, -1.0]], [[-1.0, 6.0], [-1.0, 8.0]]],
                                     [[[9.0, 10.0], [-1.0, -1.0]], [[-1.0, -1.0], [15.0, 16.0]]]])
        
        output = apply_mask_on_channel(images, channels, masks, placeholder)
        
        self.assertTrue(th.allclose(output, expected_output))
    
    def test_empty_channels(self):
        images = th.rand(2, 2, 2, 2)
        masks = th.ones_like(images)
        placeholder = 0.0
        
        output = apply_mask_on_channel(images, [], masks, placeholder)
        self.assertTrue(th.allclose(output, th.zeros_like(images)))

if __name__ == "__main__":
    unittest.main()
