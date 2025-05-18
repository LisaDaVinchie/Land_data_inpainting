import torch as th
import unittest
from ignite.metrics import SSIM as IgniteSSIM
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from metrics import SSIM

class TestSSIM(unittest.TestCase):
    
    def setUp(self):
        self.nan_placeholder = -2.0
        
        # Create simple 3x3 pred and target tensors
        self.pred = th.tensor([[0.9, 0.8, 0.7],
                            [0.8, 0.9, 0.8],
                            [0.7, 0.8, 0.9]], dtype=th.float32).unsqueeze(0).unsqueeze(0)  # 1x1x3x3

        self.target = th.tensor([[1.0, 1.0, 0.9],
                            [1.0, 1.0, 0.9],
                            [0.9, 0.9, 1.0]], dtype=th.float32).unsqueeze(0).unsqueeze(0)  # 1x1x3x3

        self.mask = th.tensor([[0, 0, 0],
                          [0, 1, 1],
                          [0, 0, 0]], dtype=th.bool).unsqueeze(0).unsqueeze(0)  # 1x1x3x3
        
    def test_get_nan_mask(self):
        image = th.tensor([[[
            [self.nan_placeholder, self.nan_placeholder, 3., 4.],
            [self.nan_placeholder, self.nan_placeholder, 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]], dtype=th.float32)
        
        expected_mask = th.tensor([[[
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]]], dtype=th.bool)
        
        ssim = SSIM(nan_placeholder=self.nan_placeholder)
        output_mask = ssim._get_nan_mask(image)
        
        self.assertTrue(th.equal(output_mask, expected_mask))
    
    def test_get_valid_mask(self):
        mask = th.tensor([[[
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1]
        ]]], dtype=th.bool)
        
        image = th.tensor([[[
            [self.nan_placeholder, self.nan_placeholder, 3., 4.],
            [self.nan_placeholder, self.nan_placeholder, 7., 8.],
            [self.nan_placeholder, self.nan_placeholder, 11., 12.],
            [13., 14., 15., 16.]
        ]]], dtype=th.float32)
        
        expected_mask = th.tensor([[[
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0]
        ]]], dtype=th.bool)
        
        ssim = SSIM(nan_placeholder=self.nan_placeholder)
        
        output_mask = ssim._get_valid_mask(image, mask)
        
        self.assertTrue(th.equal(output_mask, expected_mask))
    
    def test_uniform_kernel(self):
        ssim = SSIM(nan_placeholder=self.nan_placeholder, gaussian=False)
        kernel = ssim._uniform(11)
        self.assertEqual(kernel.shape[0], 11)
        self.assertAlmostEqual(kernel.sum().item(), 1.0, places=4)

    def test_gaussian_kernel(self):
        ssim = SSIM(nan_placeholder=self.nan_placeholder, gaussian=True)
        kernel = ssim._gaussian(11, 1.5)
        self.assertEqual(kernel.shape[0], 11)
        self.assertAlmostEqual(kernel.sum().item(), 1.0, places=4)
    
    # def test_apply_pad(self):
    #     ssim = SSIM(nan_placeholder=self.nan_placeholder, data_range=1.0, kernel_size=3, gaussian=False)

    #     # Input: 1x1x2x2 image
    #     x = th.tensor([[[[1.0, 2.0],
    #                     [3.0, 4.0]]]])

    #     # Expected padding for kernel_size=3 with reflect mode (pad=1):
    #     # Reflect padding gives:
    #     # [[4, 3, 4, 3]
    #     #  [2, 1, 2, 1]
    #     #  [4, 3, 4, 3]
    #     #  [2, 1, 2, 1]]
    #     # But PyTorch uses (left, right, top, bottom) padding order.

    #     result = ssim.apply_pad(x)
        
    #     expected = th.tensor([[[[4.0, 3.0, 4.0, 3.0],
    #                             [2.0, 1.0, 2.0, 1.0],
    #                             [4.0, 3.0, 4.0, 3.0],
    #                             [2.0, 1.0, 2.0, 1.0]]]])

    #     self.assertTrue(th.allclose(result, expected, atol=1e-4))

    
    def test_compute_mu_components(self):
        ssim = SSIM(nan_placeholder=self.nan_placeholder, data_range=1.0, kernel_size=3, gaussian=False)
        ssim._kernel = th.ones((1, 1, 3, 3)) / 9  # Uniform 3x3 kernel
        ssim.nb_channel = 1
        ssim.norm = th.ones((1, 1, 3, 3))  # No normalization effect for this test

        # Small 3x3 input
        image = th.tensor([[[[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]]]])

        target = th.tensor([[[[9.0, 8.0, 7.0],
                            [6.0, 5.0, 4.0],
                            [3.0, 2.0, 1.0]]]])

        mu_pred_sq, mu_target_sq, mu_pred_target = ssim.compute_mu_components(image, target)

        # Manually compute expected values at center pixel (1,1)
        # Average of 3x3 region: mean(image) = 5.0, mean(target) = 5.0
        # So mu_pred = mu_target = 5.0
        # mu_pred_sq = 25, mu_target_sq = 25, mu_pred_target = 25

        self.assertTrue(th.allclose(mu_pred_sq[..., 1, 1], th.tensor(25.0), atol=1e-4))
        self.assertTrue(th.allclose(mu_target_sq[..., 1, 1], th.tensor(25.0), atol=1e-4))
        self.assertTrue(th.allclose(mu_pred_target[..., 1, 1], th.tensor(25.0), atol=1e-4))

    def test_compute_covariance_components(self):
        ssim = SSIM(nan_placeholder=self.nan_placeholder, data_range=1.0, kernel_size=3, gaussian=False)
        ssim._kernel = th.ones((1, 1, 3, 3)) / 9  # Uniform kernel
        ssim.nb_channel = 1
        ssim.norm = th.ones((1, 1, 3, 3))  # No normalization effect

        # Image and target (same as before)
        image = th.tensor([[[[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]]]])

        target = th.tensor([[[[9.0, 8.0, 7.0],
                            [6.0, 5.0, 4.0],
                            [3.0, 2.0, 1.0]]]])

        mu_pred_sq = th.full((1, 1, 3, 3), 25.0)
        mu_target_sq = th.full((1, 1, 3, 3), 25.0)
        mu_pred_target = th.full((1, 1, 3, 3), 25.0)

        # E[x^2], E[y^2], E[xy]
        expected_sq_image = (image ** 2).mean().item()  # mean([1^2 ... 9^2]) = 285 / 9 = 31.6667
        expected_sq_target = (target ** 2).mean().item()  # same
        expected_image_target = (image * target).mean().item()  # mean([1*9, 2*8, ...]) = 165 / 9 = 18.3333

        cov_pred = expected_sq_image - 25.0
        cov_target = expected_sq_target - 25.0
        cov_cross = expected_image_target - 25.0

        cov_pred_tensor, cov_target_tensor, cov_cross_tensor = ssim.compute_covariance_components(
            image, target, mu_pred_sq, mu_target_sq, mu_pred_target
        )

        self.assertTrue(th.allclose(cov_pred_tensor[..., 1, 1], th.tensor(cov_pred), atol=1e-4))
        self.assertTrue(th.allclose(cov_target_tensor[..., 1, 1], th.tensor(cov_target), atol=1e-4))
        self.assertTrue(th.allclose(cov_cross_tensor[..., 1, 1], th.tensor(cov_cross), atol=1e-4))

    def test_ssim_identical_images(self):
        img = th.ones((1, 1, 8, 8))
        masks = th.zeros((1, 1, 8, 8), dtype=th.bool)
        ssim = SSIM(nan_placeholder=self.nan_placeholder, kernel_size=3, sigma=1.0)
        
        output = ssim(img, img, masks)
        expected_output = 1.0
        self.assertAlmostEqual(output, expected_output, places=4)
    
    def test_ssim_different_images(self):
        img1 = th.ones((1, 1, 8, 8))
        img2 = th.zeros((1, 1, 8, 8))
        masks = th.zeros((1, 1, 8, 8), dtype=th.bool)
        ssim = SSIM(nan_placeholder=self.nan_placeholder, kernel_size=3, sigma=1.0)
        output = ssim(img1, img2, masks)
        self.assertTrue(output < 1.0)
        
    def test_unmasked_image(self):
        shape = (1, 5, 20, 20)
        pred = th.rand(shape, dtype=th.float32)
        target = th.rand(shape, dtype=th.float32)
        masks = th.zeros(shape, dtype=th.bool)
        
        module = IgniteSSIM(data_range=1.0, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03)
        module.update((pred, target))
        expected_ssim = module.compute()
        
        myssim = SSIM(kernel_size=11, sigma=1.5, k1=0.01, k2=0.03, data_range=1.0, nan_placeholder=self.nan_placeholder)
        output = myssim(pred, target, masks)
        
        self.assertAlmostEqual(output, expected_ssim, places=4)
        
    def test_fully_masked_image(self):
        shape = (10, 3, 20, 20)
        pred = th.rand(shape, dtype=th.float32)
        target = th.rand(shape, dtype=th.float32)
        masks = th.ones(shape, dtype=th.bool)
        
        myssim = SSIM(nan_placeholder=self.nan_placeholder, data_range=1.0, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03)
        output = myssim(pred, target, masks)
        
        self.assertAlmostEqual(output.item(), 0.0, places=4)
    
    def test_compute_mu_components(self):
        ssim = SSIM(nan_placeholder=self.nan_placeholder, data_range=1.0, kernel_size=3, gaussian=False)
        ssim._kernel = th.ones((1, 1, 3, 3)) / 9  # Uniform 3x3 kernel
        ssim.nb_channel = 1
        ssim.norm = th.ones((1, 1, 3, 3))  # No normalization effect for this test

        # Small 3x3 input
        image = th.tensor([[[[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]]]])

        target = th.tensor([[[[9.0, 8.0, 7.0],
                            [6.0, 5.0, 4.0],
                            [3.0, 2.0, 1.0]]]])

        mu_pred_sq, mu_target_sq, mu_pred_target = ssim.compute_mu_components(image, target)

        # Manually compute expected values at center pixel (1,1)
        # Average of 3x3 region: mean(image) = 5.0, mean(target) = 5.0
        # So mu_pred = mu_target = 5.0
        # mu_pred_sq = 25, mu_target_sq = 25, mu_pred_target = 25

        self.assertTrue(th.allclose(mu_pred_sq[..., 1, 1], th.tensor(25.0), atol=1e-4))
        self.assertTrue(th.allclose(mu_target_sq[..., 1, 1], th.tensor(25.0), atol=1e-4))
        self.assertTrue(th.allclose(mu_pred_target[..., 1, 1], th.tensor(25.0), atol=1e-4))
    
    # def test_partial_masked_image_uniform(self):
    #     k1 = 0.01
    #     k2 = 0.03
        
    #     c1 = k1 ** 2
    #     c2 = k2 ** 2
        
    #     n_valid_pixels = th.sum(~self.mask).item()
    #     mu_pred = th.sum(self.pred[~self.mask]) / n_valid_pixels
    #     mu_target = th.sum(self.target[~self.mask]) / n_valid_pixels
        
    #     sigma_pred = th.sum((self.pred[~self.mask] - mu_pred) ** 2) / n_valid_pixels
    #     sigma_target = th.sum((self.target[~self.mask] - mu_target) ** 2) / n_valid_pixels
    #     sigma_pred_target = th.sum((self.pred[~self.mask] - mu_pred) * (self.target[~self.mask] - mu_target)) / n_valid_pixels
        
    #     numerator = (2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)
    #     denominator = (mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2)
    #     expected_ssim = (numerator / denominator).item()
        
    #     myssim = SSIM(nan_placeholder=self.nan_placeholder, data_range=1.0, kernel_size=3, sigma=1.5, k1=k1, k2=k2, gaussian=False)
    #     output = myssim(self.pred, self.target, self.mask)
    #     print(f"Output: {output}, Expected: {expected_ssim}")
    #     self.assertAlmostEqual(output, expected_ssim, places=4)
        
    def test_partial_masked_image_uniform(self):
        k1 = 0.01
        k2 = 0.03

        c1 = k1 ** 2
        c2 = k2 ** 2

        # Use scalar mean/variance
        n_valid_pixels = th.sum(~self.mask).item()
        mu_pred = th.sum(self.pred[~self.mask]) / n_valid_pixels
        mu_target = th.sum(self.target[~self.mask]) / n_valid_pixels

        sigma_pred = th.sum((self.pred[~self.mask] - mu_pred) ** 2) / n_valid_pixels
        sigma_target = th.sum((self.target[~self.mask] - mu_target) ** 2) / n_valid_pixels
        sigma_pred_target = th.sum(
            (self.pred[~self.mask] - mu_pred) * (self.target[~self.mask] - mu_target)
        ) / n_valid_pixels

        numerator = (2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)
        denominator = (mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2)
        expected_ssim = (numerator / denominator).item()

        # Now make the SSIM match this scalar behavior
        myssim = SSIM(
            nan_placeholder=self.nan_placeholder,
            data_range=1.0,
            kernel_size=3,
            gaussian=False,  # << key change
            k1=k1,
            k2=k2
        )

        output = myssim(self.pred, self.target, self.mask)
        print(f"Output: {output}, Expected: {expected_ssim}")
        self.assertAlmostEqual(output, expected_ssim, places=4)


if __name__ == '__main__':
    unittest.main()