import torch as th

def per_pixel_loss(prediction: th.tensor, target: th.tensor, masks: th.tensor) -> th.tensor:
    """Calculate the per-pixel loss between the prediction and the target, ignoring masked pixels.

    Args:
        prediction (th.tensor): output of the model, shape (batch_size, channels, height, width)
        target (th.tensor): ground truth, shape (batch_size, channels, height, width)
        masks (th.tensor): binary mask with 0s for masked pixels, shape (batch_size, channels, height, width)

    Returns:
        th.tensor: per-pixel loss
    """
    
    n_valid_pixels = (~masks.bool()).sum().float()
    if n_valid_pixels == 0:
        return 0
    
    diff = th.abs(prediction - target)
    masked_diff = diff.masked_fill(masks.bool(), 0.0)
    return masked_diff.sum() / n_valid_pixels

def per_pixel_mse(prediction: th.tensor, target: th.tensor, masks: th.tensor) -> th.tensor:
    """Calculate the per-pixel loss between the prediction and the target, ignoring masked pixels.

    Args:
        prediction (th.tensor): output of the model, shape (batch_size, channels, height, width)
        target (th.tensor): ground truth, shape (batch_size, channels, height, width)
        masks (th.tensor): binary mask with 0s for masked pixels, shape (batch_size, channels, height, width)

    Returns:
        th.tensor: per-pixel loss
    """
    
    n_valid_pixels = (~masks.bool()).sum().float()
    if n_valid_pixels == 0:
        return 0
    
    diff = (prediction - target) ** 2
    masked_diff = diff.masked_fill(masks.bool(), 0.0)
    return masked_diff.sum() / n_valid_pixels