import torch as th
import torch.nn as nn

def get_loss_function(loss_kind: str) -> nn.Module:
    """Get the loss function based on the specified kind.

    Args:
        loss_kind (str): type of loss function to use. Options are "per_pixel_loss" or "per_pixel_mse".

    Returns:
        nn.Module: loss function
    """
    
    if loss_kind == "per_pixel_loss":
        return PerPixelL1()
    elif loss_kind == "per_pixel_mse":
        return PerPixelMSE()
    elif loss_kind == "tv_loss":
        return TotalVariationLoss()
    elif loss_kind == "custom1":
        return CustomLoss1()
    else:
        raise ValueError(f"Loss kind {loss_kind} not recognized")

class PerPixelMSE(nn.Module):
    def __init__(self):
        """Initialize the Per Pixel MSE loss module."""
        super(PerPixelMSE, self).__init__()
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
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

class PerPixelL1(nn.Module):
    def __init__(self):
        """Initialize the Per Pixel L1 loss module."""
        super(PerPixelL1, self).__init__()
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
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

class TotalVariationLoss(nn.Module):
    def __init__(self):
        """Initialize the Total Variation Loss module."""
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Calculate the total variation loss between the prediction and the target, ignoring masked pixels.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): binary mask with 0s for masked pixels, shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: total variation loss
        """

        # Dilate the mask by 1 pixel
        inv_dilated_mask = self._dilate_mask(masks, 1, True)

        if inv_dilated_mask.sum() == 0: # if all pixels are masked, return 0
            return prediction.sum() * 0.0 # Trick to preserve grad
        
        # calculate the image to be used for the loss
        image_comp = self._compose_image(prediction, target, inv_dilated_mask.bool())
        
        return self._tv_loss(image_comp, inv_dilated_mask)

    def _tv_loss(self, image: th.Tensor, mask: th.Tensor) -> th.Tensor:
        """Calculate the total variation loss on the masked image, only where the mask is 1.

        Args:
            image (th.Tensor): image to be used for the loss, shape (batch_size, channels, height, width)
            mask (th.Tensor): binary mask of shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: total variation loss
        """
        # calculate the valid pixels in the x and y direction
        valid_x = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        valid_y = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        
        # count the number of differences performed
        n_diffs = valid_x.sum() + valid_y.sum()
        
        # calculate the difference between adjacent pixels in the x and y direction
        diff_rows = image[:, :, 1:, :] - image[:, :, :-1, :]
        diff_cols = image[:, :, :, 1:] - image[:, :, :, :-1]
        
        # calculate the absolute difference between adjacent pixels in the x and y direction
        # Keep only the valid pixels
        valid_norm_x = th.abs(diff_rows) * valid_x
        valid_norm_y = th.abs(diff_cols) * valid_y
        # sum the absolute differences in the x and y direction and divide by the number of valid pixels
        return (valid_norm_x.sum() + valid_norm_y.sum()) / n_diffs
    
    def _compose_image(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Compose the prediction and target images using the masks.
        Where the mask is True, use the target, otherwise use the prediction.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): binary mask with True where the pixel is masked, shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: composed image, shape (batch_size, channels, height, width)
        """
        return th.where(masks, prediction, target)
        
    
    def _dilate_mask(self, masks: th.Tensor, dilation: int, inverse: bool = False) -> th.Tensor:
        """Dilate a binary mask by a given number of pixels. The epanded regions will be the 0 ones.

        Args:
            masks (th.Tensor): binary mask with 0 where the pixel is masked, 1 otherwise, shape (batch_size, channels, height, width).
            dilation (int): number of pixels to dilate the mask.
            inverse (bool, optional): whether to return the dilated mask or the inverse dilated mask. Defaults to False.

        Returns:
            th.Tensor: _description_
        """
        kernel_size = 2 * dilation + 1
        
        B, C, H, W = masks.shape
        
        kernel = th.ones(1, 1, kernel_size, kernel_size, dtype=th.float32, device =masks.device)
        
        inv_mask = (1 - masks).view(B * C, 1, H, W).float()  # Invert the mask to dilate the unmasked pixels
        
        dilated_mask = th.nn.functional.conv2d(inv_mask, kernel, padding=dilation, groups=1) > 0.5
        dilated_mask = dilated_mask.view(B, C, H, W).float()
        
        return dilated_mask if inverse else 1 - dilated_mask
   
class CustomLoss1(nn.Module):
    def __init__(self, per_pixel_weight: float = 1.0, tv_weight: float = 0.1):
        """Combine per-pixel loss and total variation loss.

        Args:
            per_pixel_weight (float, optional): weight for the per-pixel loss. Defaults to 1.0.
            tv_weight (float, optional): weight for the total variation loss. Defaults to 0.1.
        """
        super(CustomLoss1, self).__init__()
        self.per_pixel_weight = per_pixel_weight
        self.tv_weight = tv_weight
        self.per_pixel_loss = PerPixelMSE()
        self.tv_loss = TotalVariationLoss()
        
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Calculate the combined loss.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): binary mask with 0 where the pixel is masked, shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: combined loss
        """
        per_pixel_loss = self.per_pixel_loss(prediction, target, masks)
        tv_loss = self.tv_loss(prediction, target, masks)
        
        return self.per_pixel_weight * per_pixel_loss + self.tv_weight * tv_loss