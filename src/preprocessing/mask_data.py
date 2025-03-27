import torch as th
from pathlib import Path
import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.import_params_json import load_config

def apply_mask_on_channel(images: th.Tensor, masks: th.Tensor, placeholder: float = None) -> th.Tensor:
    """Mask the image with the mask, using a placeholder. If the placeholder is none, use the mean of the level"""
    new_images = images.clone()
    if placeholder is not None:
        return new_images * masks + placeholder * (1 - masks)
    
    means = (images * masks).sum(dim=(2, 3), keepdim=True) / (masks.sum(dim=(2, 3), keepdim=True))
    return new_images * masks + means * (1 - masks)

def mask_inversemask_image(images: th.Tensor, masks: th.Tensor, placeholder: float = None) -> tuple:
    """Mask the image with a placeholder value.
    If the placeholder is none, use the mean of the level and, if there are nans, mask them as well.

    Args:
        images (th.Tensor): images to mask, of shape (batch_size, channels, height, width)
        masks (th.Tensor): masks to apply, of shape (batch_size, channels, height, width)
        placeholder (float, optional): number to use for masked pixels. Defaults to None.

    Returns:
        tuple: masked_images, inverse masked images
    """
    new_images = images.clone()
    
    masks[images.isnan()] = 0
    inverse_masks = 1 - masks
    inverse_masks[images.isnan()] = 0
    
    new_images[new_images.isnan()] = 0
    
    if placeholder is None:
        placeholder, inv_placeholder = 0, 0
        
        if masks.sum() > 0:
            placeholder = (new_images * masks).sum(dim=(2, 3), keepdim=True) / (masks.sum(dim=(2, 3), keepdim=True))
        
        if inverse_masks.sum() > 0:
            inv_placeholder = (new_images * inverse_masks).sum(dim=(2, 3), keepdim=True) / (inverse_masks.sum(dim=(2, 3), keepdim=True))
        
    else:
        inv_placeholder = placeholder
    
    masked_img = new_images * masks + placeholder * (1 - masks)
    inverse_masked_img = new_images * inverse_masks + inv_placeholder * (1 - inverse_masks)
    return masked_img, inverse_masked_img


def create_square_mask(image_width: int, image_height: int, mask_percentage: float) -> th.Tensor:
    """Create a square mask of n_pixels in the image"""
    n_pixels = int(mask_percentage * image_width * image_height)
    square_width = int(n_pixels ** 0.5)
    mask = th.ones((image_width, image_height), dtype=th.float32)
    
    # Get a random top-left corner for the square
    row_idx = th.randint(0, image_height - square_width, (1,)).item()
    col_idx = th.randint(0, image_width - square_width, (1,)).item()
    
    mask[
        row_idx: row_idx + square_width,
        col_idx: col_idx + square_width
    ] = 0
    
    return mask

# def create_lines_mask(image_width: int, image_height: int, min_thickness: int, max_thickness: int, n_lines: int) -> th.Tensor:
#     """Create a mask with lines of random thickness"""
    
#     mask = th.ones((image_width, image_height), dtype=th.float32)
    
#     for i in range(n_lines):
#         # Get random x locations to start line
#         x1, x2 = th.randint(1, image_width, (2,))
#         # Get random y locations to start line
#         y1, y2 = th.randint(1, image_height, (2,))
#         # Get random thickness of the line drawn
#         thickness = th.randint(min_thickness, max_thickness, (1,)).item()
#         # Draw black line on the white mask
#         mask[
#             x1: x2,
#             y1: y2
#         ] = 0
        
#         cv2.line(mask.numpy(), (x1, y1), (x2, y2), (0, 0, 0), thickness)
        
#     return th.Tensor(mask)