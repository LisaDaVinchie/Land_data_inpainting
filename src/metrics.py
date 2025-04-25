import torch as th

class PCC(th.nn.Module):
    """
    Computes the Pearson Correlation Coefficient (PCC) between two tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: th.Tensor, y: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """
        Computes the PCC between two tensors.

        Args:
            x (th.Tensor): First tensor, shape (B, C, H, W).
            y (th.Tensor): Second tensor, shape (B, C, H, W).
            masks (th.Tensor): Binary mask with 0s for masked pixels, shape (B, C, H, W).
        Returns:
            th.Tensor: The PCC between the two tensors.
        """
        
        n_valid_pixels = (~masks.bool()).sum().float() # count the number of masked (0) pixels, by inverting the mask
        if n_valid_pixels == 0: # if all pixels are masked, return 0
            return 0
        
        # Set the masked pixels to NaN, where
        x = x.masked_fill(masks.bool(), th.nan)
        y = y.masked_fill(masks.bool(), th.nan) # Set the masked pixels to 0 where the mask is 1, i.e. where the pixel is not masked
        
        # Flatten the tensors for each channel
        # Obtain the shape (B, C, H*W)
        x_flat = x.view(x.shape[0], x.shape[1], -1)
        y_flat = y.view(y.shape[0], y.shape[1], -1)
        
        # Compute the mean of each image and channel
        # Obtain the shape (B, C, 1)
        x_mean = x_flat.mean(dim=1, keepdim=True)
        y_mean = y_flat.mean(dim=1, keepdim=True)
        
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
        
        cov = (x_centered * y_centered).mean(dim=2)
        
        x_std = th.sqrt((x_centered ** 2).sum(dim=2))
        y_std = th.sqrt((y_centered ** 2).sum(dim=2))
        
        if th.any(x_std == 0) or th.any(y_std == 0):
            raise ValueError("Standard deviation is zero, cannot compute PCC.")
        
        return cov / (x_std * y_std)
        
        
        
        