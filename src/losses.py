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
    
    
def calculate_valid_pixels(masks: th.Tensor, target: th.Tensor, nan_placeholder: float) -> int:
    
    # Create a mask that is 1 where the pixel is valid and 0 otherwise
    masks = calculate_valid_mask(masks, target, nan_placeholder, inv=True)
    
    return masks.float().sum().item()

def calculate_valid_mask(masks: th.Tensor, target: th.Tensor, nan_placeholder: float, inv: bool = False) -> th.Tensor:
    """Get the pixels to be used for inpainting, i.e. the pixels that are masked and not NaN.

    Args:
        masks (th.Tensor): binary mask with 0 where the pixel is masked, shape (batch_size, channels, height, width).
        target (th.Tensor): ground truth, shape (batch_size, channels, height, width).
        nan_placeholder (float): value used to represent NaN pixels in the target tensor.
        inv (bool, optional): whether to return the inverse mask. Defaults to False.

    Returns:
        th.Tensor: mask with 0 where the pixel is valid and 1 otherwise if inv is False, or 1 where the pixel is valid and  otherwise if inv is True.
    """
    
    # Create a mask that is 1 where the pixel is masked and 0 otherwise
    nans_mask = th.where(target == nan_placeholder, th.ones_like(target), th.zeros_like(target)).bool()
    
    # Create a mask that is 0 where the pixel is masked and 1 otherwise
    masks |= nans_mask  # Combine the masks with the NaN mask
    
    # If inv, create a mask that is 1 where the pixel is masked and 0 otherwise
    return ~masks if inv else masks
        
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
        
        # Calculate the squared difference, for each pixel
        diff = (prediction - target) ** 2 
        
        # Set the masked pixels to 0 where the mask is 1, i.e. where the pixel is not masked
        masked_diff = diff.masked_fill(masks, 0.0)
        return masked_diff.sum() # Return the mean of the squared differences over the number of valid pixels

class PerPixelL1(nn.Module):
    def __init__(self):
        """Initialize the Per Pixel L1 loss module.

        Args:
            nan_placeholder (float, optional): placeholder for nan pixels. Defaults to -2.0.
        """
        super(PerPixelL1, self).__init__()
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
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
        return masked_diff.sum()