import torch as th
import torch.nn as nn
import torch.nn.functional as F

class PartialMaxPool2D(nn.Module):
    """Mask aware max pooling layer for 2D data, that ignores masked values."""
    
    def __init__(self, kernel_size, stride = None, padding = 0, ceil_mode = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
    
    def forward(self, x: th.Tensor, mask: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass.

        Args:
            x (th.Tensor): tensor of shape (batch_size, n_channels, image_width, image_height), can contain NaNs
            mask (th.Tensor): tensor of shape (batch_size, n_channels, image_width, image_height), 1 where x is valid, 0 where x is masked

        Returns:
            pooled_x (th.Tensor): Max-pooled output, masked regions set to NaN.
            pooled_mask (th.Tensor): Pooled mask (1 where output is valid, else 0).
        """
        
        # Replace masked values with -inf where mask is 0
        x_masked = th.where(mask != 0, x, float("-inf"))
        
        # Apply max pooling
        pooled_x = F.max_pool2d(input=x_masked, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, ceil_mode=self.ceil_mode)
        pooled_mask = F.max_pool2d(input=mask, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, ceil_mode=self.ceil_mode)
        
        # Replace pooled values with NaN where mask is 0
        pooled_x[pooled_mask == 0] = th.nan
        
        return pooled_x, pooled_mask