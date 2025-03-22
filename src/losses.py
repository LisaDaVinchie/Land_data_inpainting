import torch as th

def per_pixel_loss(prediction: th.tensor, target: th.tensor, mask: th.tensor) -> th.tensor:
    """Calculate the per-pixel loss between the prediction and the target, using a mask."""
    
    n_valid_pixels = (~mask.bool()).sum().float()
    if n_valid_pixels == 0:
        return 0
    
    diff = th.abs(prediction - target)
    masked_diff = diff.masked_fill(mask.bool(), 0.0)
    return masked_diff.sum() / n_valid_pixels