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
        return 0
    
    diff = (prediction - target) ** 2
    masked_diff = diff.masked_fill(masks.bool(), 0.0)
    return masked_diff.sum() / n_valid_pixels