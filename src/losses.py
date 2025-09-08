import torch as th
import torch.nn as nn
        
class PerPixelMSE(nn.Module):
    def __init__(self):
        """Initialize the Per Pixel MSE loss module."""
        super(PerPixelMSE, self).__init__()
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Calculate the per-pixel loss between the prediction and the target on masked pixels.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): binary mask with 0 where the loss must be calculated, shape (batch_size, channels, height, width).

        Returns:
            th.Tensor: per-pixel loss calculated only on the masked pixels.
        """
        
        # Calculate squared differences for all images at once
        squared_diff = (prediction - target) ** 2
        
        masked_diff = squared_diff.masked_fill(masks, 0.0)  # Set masked pixels to 0
        
        # Sum over spatial dimensions and channels (keeping batch dimension)
        diff_sums = masked_diff.sum(dim=(1, 2, 3))
        
        # Count valid pixels for each image in batch
        n_valid_pixels = (~masks).float().sum(dim=(1, 2, 3))
        
        # Compute normalized loss for each image
        per_image_losses = diff_sums / (n_valid_pixels + 1e-8)
        
        # Sum all individual image losses (matches original logic)
        total_loss = per_image_losses.sum()
            
        return total_loss