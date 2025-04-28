import torch as th

class PCC(th.nn.Module):
    """
    Computes the Pearson Correlation Coefficient (PCC) between two tensors.
    """

    def __init__(self, nan_placeholder: float):
        super().__init__()
        
        self.nan_placeholder = nan_placeholder

    def forward(self, image: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """
        Computes the PCC between two tensors.

        Args:
            image (th.Tensor): image tensor, shape (B, C, H, W).
            target (th.Tensor): Second tensor, shape (B, C, H, W).
            masks (th.Tensor): th.bool mask with False for masked (= to consider) pixels, shape (B, C, H, W).
        Returns:
            th.Tensor: The PCC between the two tensors.
        """
        
        # Create a mask for the NaN values, True where the NaN values are
        nan_mask = self._get_nan_mask(target)
        
        # Create a mask for the valid pixels, True for valid pixels
        # Pixels are valid if they are masked (masks False) and not NaN (nan_mask False)
        valid_mask = self._get_valid_mask(masks, nan_mask)
        
        # Count the number of valid pixels
        n_valid_pixels = valid_mask.sum().float()
        
        n_valid_pixels_inv = 1.0 / n_valid_pixels
        
        if n_valid_pixels == 0: # if all pixels are masked, return 0
            return 0
        
        # Set the non valid pixels to NaN
        image = image.masked_fill(~valid_mask, th.nan)
        target = target.masked_fill(~valid_mask, th.nan)
        
        # Flatten the tensors for each channel
        # Obtain the shape (B, C, H*W)
        x_flat = image.view(image.shape[0], image.shape[1], -1)
        y_flat = target.view(target.shape[0], target.shape[1], -1)
        
        # Compute the mean for each channel, shape (B, C, 1)
        x_mean = th.nansum(x_flat, dim=2, keepdim=True) * n_valid_pixels_inv
        y_mean = th.nansum(y_flat, dim=2, keepdim=True) * n_valid_pixels_inv
        
        # Center the tensors by subtracting the mean
        x_flat -= x_mean
        y_flat -= y_mean
        
        # cov = (x_centered * y_centered).mean(dim=2)
        cov = th.nansum(x_flat * y_flat, dim=2) * n_valid_pixels_inv
        
        # x_std = th.sqrt((x_flat ** 2).sum(dim=2))
        # y_std = th.sqrt((x_flat ** 2).sum(dim=2))
        
        x_std = th.sqrt(th.nansum(x_flat ** 2, dim=2) * n_valid_pixels_inv)
        y_std = th.sqrt(th.nansum(y_flat ** 2, dim=2) * n_valid_pixels_inv)
        
        if th.any(x_std == 0) or th.any(y_std == 0):
            raise ValueError("Standard deviation is zero, cannot compute PCC.")
        
        return cov / (x_std * y_std)
    
    def _get_nan_mask(self, target: th.Tensor) -> th.Tensor:
        """
        Computes the NaN mask for the tensors, True where the NaN values are.

        Args:
            target (th.Tensor): Second tensor, shape (B, C, H, W).

        Returns:
            th.Tensor: The NaN mask for the tensors.
        """
        
        return th.where(target == self.nan_placeholder, th.ones_like(target), th.zeros_like(target)).bool()
    
    def _get_valid_mask(self, masks: th.Tensor, nan_mask: th.Tensor) -> th.Tensor:
        """
        Computes the valid mask for the tensors, True for valid pixels.

        Args:
            masks (th.Tensor): th.bool mask with True for pixels to consider, shape (B, C, H, W).
            nan_mask (th.Tensor): Mask for NaN values, True where NaN is shape (B, C, H, W).

        Returns:
            th.Tensor: The valid mask, True for valid pixels.
        """
        
        # Pixels are valid if they are masked (masks False) and not NaN (nan_mask False)
        return ~(masks | nan_mask)
        
        
        
        