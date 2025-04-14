import torch as th

def per_pixel_loss(prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
    """Calculate the per-pixel loss between the prediction and the target, ignoring masked pixels.

    Args:
        prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
        target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
        masks (th.Tensor): binary mask with 0s for masked pixels, shape (batch_size, channels, height, width)

    Returns:
        th.Tensor: per-pixel loss
    """
    
    n_valid_pixels = (~masks.bool()).sum().float()
    if n_valid_pixels == 0:
        return th.tensor(0.0, requires_grad=True)
    
    diff = th.abs(prediction - target)
    masked_diff = diff.masked_fill(masks.bool(), 0.0)
    return masked_diff.sum() / n_valid_pixels

def per_pixel_mse(prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
    """Calculate the per-pixel loss between the prediction and the target on masked pixels.

    Args:
        prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
        target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
        masks (th.Tensor): binary mask with 0 where the pixel is masked, shape (batch_size, channels, height, width).
        The loss is calculated on the masked (0) pixels.

    Returns:
        th.Tensor: per-pixel loss
    """
    
    n_valid_pixels = (~masks.bool()).sum().float() # count the number of masked (0) pixels, by inverting the mask
    if n_valid_pixels == 0: # if all pixels are masked, return 0
        return 0
    
    diff = (prediction - target) ** 2 # Calculate the squared difference, for each pixel
    masked_diff = diff.masked_fill(masks.bool(), 0.0) # Set the masked pixels to 0 where the mask is 1, i.e. where the pixel is not masked
    return masked_diff.sum() / n_valid_pixels # Return the mean of the squared differences over the number of valid pixels