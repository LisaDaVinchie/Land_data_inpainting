import torch as th
import torch.nn as nn

def get_loss_function(loss_kind: str) -> nn.Module:
    """Get the loss function based on the specified kind.

    Args:
        loss_kind (str): type of loss function to use. Options are "per_pixel_loss" or "per_pixel_mse".
        nan_placeholder (float): value used to represent NaN pixels in the target tensor.

    Returns:
        nn.Module: loss function
    """
    
    if loss_kind == "per_pixel":
        return PerPixelL1()
    elif loss_kind == "per_pixel_mse":
        return PerPixelMSE()
    else:
        raise ValueError(f"Loss kind {loss_kind} not recognized")
        
class PerPixelMSE(nn.Module):
    def __init__(self):
        """Initialize the Per Pixel MSE loss module.
        
        Args:
            nan_placeholder (float, optional): placeholder for nan pixels. Defaults to -2.0.
        """
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
        
        # Apply mask (using ~masks to select pixels where mask is 0)
        masked_diff = squared_diff * (~masks).float()
        
        # Sum over spatial dimensions and channels (keeping batch dimension)
        diff_sums = masked_diff.sum(dim=(1, 2, 3))
        
        # Count valid pixels for each image in batch
        n_valid_pixels = (~masks).float().sum(dim=(1, 2, 3))
        
        # Compute normalized loss for each image
        per_image_losses = diff_sums / (n_valid_pixels + 1e-8)
        
        # Sum all individual image losses (matches original logic)
        total_loss = per_image_losses.sum()
            
        return total_loss
    
class PerPixelL1(nn.Module):
    def __init__(self):
        """Initialize the Per Pixel L1 loss module.

        Args:
            nan_placeholder (float, optional): placeholder for nan pixels. Defaults to -2.0.
        """
        super(PerPixelL1, self).__init__()
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor, normalize: bool = True) -> th.Tensor:
        """Calculate the per-pixel loss between the prediction and the target, ignoring masked pixels.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): th.bool mask with False where the loss must be calculated, shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: per-pixel loss
        """
        
        diff = th.abs(prediction - target)
        masked_diff = diff.masked_fill(masks, 0.0)
        diff_sum = masked_diff.sum()
        
        if normalize:
            n_valid_pixels = (~masks).float().sum()
            
        return diff_sum / (n_valid_pixels + 1e-8) if normalize else diff_sum