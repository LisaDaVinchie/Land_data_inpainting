import torch as th
import torch.nn as nn

def get_loss_function(loss_kind: str, nan_placeholder: float) -> nn.Module:
    """Get the loss function based on the specified kind.

    Args:
        loss_kind (str): type of loss function to use. Options are "per_pixel_loss" or "per_pixel_mse".
        nan_placeholder (float): value used to represent NaN pixels in the target tensor.

    Returns:
        nn.Module: loss function
    """
    
    if loss_kind == "per_pixel":
        return PerPixelL1(nan_placeholder=nan_placeholder)
    elif loss_kind == "per_pixel_mse":
        return PerPixelMSE(nan_placeholder=nan_placeholder)
    elif loss_kind == "tv_loss":
        return TotalVariationLoss(nan_placeholder=nan_placeholder)
    elif loss_kind == "custom1":
        return CustomLoss1(nan_placeholder=nan_placeholder)
    else:
        raise ValueError(f"Loss kind {loss_kind} not recognized")

class DINCAE1Loss(nn.Module):
    
    def __init__(self, nan_placeholder: float = -2.0):
        """Initialize the DINCAE1Loss module.

        Args:
            loss_kind (str): type of loss function to use. Options are "per_pixel_loss" or "per_pixel_mse".
            nan_placeholder (float, optional): value used to represent NaN pixels in the target tensor. Defaults to -2.0.
        """
        super(DINCAE1Loss, self).__init__()
        self.nan_placeholder = nan_placeholder
        self.eps = 1e-6 # Small constant to avoid division by zero
        
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        
        # Select the first channel of the target tensor
        target = target[:, 0:1, :, :]
        mean_pred = prediction[:, 0:1, :, :]
        stdev_pred = prediction[:, 1:2, :, :]
        
        # Create a mask that is 1 where the target is the NaN placeholder and 0 otherwise
        nans_mask = th.where(target == self.nan_placeholder, th.ones_like(target), th.zeros_like(target)).bool()
        
        valid_mask = ~(masks[:, 0, :, :] | nans_mask) # Create a mask that is 1 where the pixel is valid and 0 otherwise
        
        N = valid_mask.sum().float() # count the number of valid pixels
        if N == 0:
            return th.tensor(0.0, requires_grad=True)
        
        # Add small constant to stdev for numerical stability
        stdev_pred = stdev_pred + self.eps
        
        # Compute squared term
        squared_term = ((mean_pred - target) / stdev_pred).pow(2)
        
        # Compute log term (more numerically stable than squaring first)
        log_term = th.log(stdev_pred.pow(2))
        
        # Combine terms
        # Non valid pixels are set to 0
        loss_terms = (squared_term + log_term) * valid_mask.float()
        
        # Sum and normalize
        total_loss = loss_terms.sum() / (2 * N)
        
        return total_loss
        
class PerPixelMSE(nn.Module):
    def __init__(self, nan_placeholder: float):
        """Initialize the Per Pixel MSE loss module.
        
        Args:
            nan_placeholder (float, optional): placeholder for nan pixels. Defaults to -2.0.
        """
        super(PerPixelMSE, self).__init__()
        
        self.nan_placeholder = nan_placeholder
    
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
        
        n_valid_pixels = (~masks).sum().float() # count the number of masked (0) pixels, by inverting the mask
        
        # Create a mask that is 1 where the target is the NaN placeholder and 0 otherwise
        # This is used to count the number of NaN pixels
        n_nans = th.where(target == self.nan_placeholder, th.ones_like(target), th.zeros_like(target)).sum().float() # count the number of NaN pixels
        # Subtract the number of NaN pixels from the number of valid pixels
        n_valid_pixels -= n_nans 
        if n_valid_pixels == 0: # if all pixels are masked, return 0
            return th.tensor(0.0, requires_grad=True)
        
        diff = (prediction - target) ** 2 # Calculate the squared difference, for each pixel
        masked_diff = diff.masked_fill(masks, 0.0) # Set the masked pixels to 0 where the mask is 1, i.e. where the pixel is not masked
        return masked_diff.sum() / n_valid_pixels # Return the mean of the squared differences over the number of valid pixels

class PerPixelL1(nn.Module):
    def __init__(self, nan_placeholder: float = -2.0):
        """Initialize the Per Pixel L1 loss module.

        Args:
            nan_placeholder (float, optional): placeholder for nan pixels. Defaults to -2.0.
        """
        super(PerPixelL1, self).__init__()
        
        self.nan_placeholder = nan_placeholder
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Calculate the per-pixel loss between the prediction and the target, ignoring masked pixels.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): th.bool mask with False for masked pixels, shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: per-pixel loss
        """
        
        n_valid_pixels = (~masks).sum().float()
        n_nans = th.where(target == self.nan_placeholder, th.ones_like(target), th.zeros_like(target)).sum().float() # count the number of NaN pixels
        
        n_valid_pixels -= n_nans # Subtract the number of NaN pixels from the number of valid pixels
        if n_valid_pixels == 0:
            return th.tensor(0.0, requires_grad=True)
        
        diff = th.abs(prediction - target)
        masked_diff = diff.masked_fill(masks, 0.0)
        return masked_diff.sum() / n_valid_pixels

class TotalVariationLoss(nn.Module):
    def __init__(self, nan_placeholder: float):
        """Initialize the Total Variation Loss module."""
        super(TotalVariationLoss, self).__init__()
        self.nan_placeholder = nan_placeholder
    
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Calculate the total variation loss between the prediction and the target, ignoring masked pixels.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): th.bool mask with False for masked pixels, shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: total variation loss
        """

        # Dilate the mask by 1 pixel
        inv_dilated_mask = self._dilate_mask(masks, 1, True)
        # Get a mask that is 1 where the image is masked and 0 otherwise

        if inv_dilated_mask.sum() == 0: # if all pixels are masked, return 0
            return prediction.sum() * 0.0 # Trick to preserve grad
        
        # Calculate the image to be used for the loss
        image_comp = self._compose_image(prediction, target, inv_dilated_mask)
        
        # Exclude NaN pixels from the mask
        inv_dilated_mask = self._exclude_nans_from_mask(prediction, inv_dilated_mask)
        
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

    def _exclude_nans_from_mask(self, image: th.Tensor, mask: th.Tensor) -> th.Tensor:
        """Exclude NaN pixels from the valid pixels in the mask.

        Args:
            image (th.Tensor): image to be used for the loss, shape (batch_size, channels, height, width)
            mask (th.Tensor): th.bool mask with True where the pixel is masked, shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: mask with NaN pixels excluded, True where the pixel is masked, False otherwise
        """
        nan_mask = th.where(image == self.nan_placeholder, th.ones_like(image), th.zeros_like(image)).bool() # Create a mask that is 1 where the target is the NaN placeholder and 0 otherwise
        
        # Where both mask and nan_mask are 0, assign value 1, otherwise keep the value of mask
        return th.where(mask & nan_mask, th.zeros_like(mask), mask).bool() # Invert the mask to exclude the NaN pixels
    
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
            masks (th.Tensor): th.bool mask with False where the pixel is masked, True otherwise, shape (batch_size, channels, height, width).
            dilation (int): number of pixels to dilate the mask.
            inverse (bool, optional): whether to return the dilated mask or the inverse dilated mask. Defaults to False.

        Returns:
            th.Tensor: dilated mask, with False where the pixel is masked, True otherwise if not inverse.
            The contrary if inverse is True.
        """
        kernel_size = 2 * dilation + 1
        
        B, C, H, W = masks.shape
        
        kernel = th.ones(1, 1, kernel_size, kernel_size, dtype=th.float32, device = masks.device)
        
        inv_mask = (~masks).view(B * C, 1, H, W).float()  # Invert the mask to dilate the unmasked pixels
        
        dilated_mask = th.nn.functional.conv2d(inv_mask, kernel, padding=dilation, groups=1) > 0.5
        dilated_mask = dilated_mask.view(B, C, H, W).bool()
        
        return dilated_mask if inverse else ~dilated_mask
   
# class StyleLoss(nn.Module):
#     def __init__(self, nan_placeholder: float):
#         """Initialize the Style Loss module.

#         Args:
#             target_feature (th.Tensor): target feature map, shape (batch_size, channels, height, width)
#         """
#         super(StyleLoss, self).__init__()
#         self.target = target_feature.detach()
        
#     def _gram_matrix(features):
#         b, c, h, w = features.size()
#         features = features.view(b, c, h * w)
#         gram = th.bmm(features, features.transpose(1, 2))  # Batch matrix multiplication
#         return gram / (c * h * w)
        
#     def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
#         """Calculate the style loss between the input and the target feature map.

#         Args:
#             x (th.Tensor): input feature map, shape (batch_size, channels, height, width)

#         Returns:
#             th.Tensor: style loss
#         """
#         return nn.functional.mse_loss(x, self.target)

class CustomLoss1(nn.Module):
    def __init__(self, nan_placeholder: float, per_pixel_weight: float = 1.0, tv_weight: float = 0.1):
        """Combine per-pixel loss and total variation loss.

        Args:
            per_pixel_weight (float, optional): weight for the per-pixel loss. Defaults to 1.0.
            tv_weight (float, optional): weight for the total variation loss. Defaults to 0.1.
        """
        super(CustomLoss1, self).__init__()
        self.per_pixel_weight = per_pixel_weight
        self.tv_weight = tv_weight
        self.per_pixel_loss = PerPixelMSE(nan_placeholder=nan_placeholder)
        self.tv_loss = TotalVariationLoss(nan_placeholder=nan_placeholder)
        
    def forward(self, prediction: th.Tensor, target: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Calculate the combined loss.

        Args:
            prediction (th.Tensor): output of the model, shape (batch_size, channels, height, width)
            target (th.Tensor): ground truth, shape (batch_size, channels, height, width)
            masks (th.Tensor): th.bool mask with False where the pixel is masked, shape (batch_size, channels, height, width)

        Returns:
            th.Tensor: combined loss
        """
        per_pixel_loss = self.per_pixel_loss(prediction, target, masks)
        tv_loss = self.tv_loss(prediction, target, masks)
        
        return self.per_pixel_weight * per_pixel_loss + self.tv_weight * tv_loss