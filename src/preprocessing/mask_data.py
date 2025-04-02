import torch as th
import os
import sys
import random
import math

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
        images (th.Tensor): images to mask, of shape (batch_size, channels, nrows, ncols)
        masks (th.Tensor): masks to apply, of shape (batch_size, channels, nrows, ncols)
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

def create_square_mask(image_nrows: int, image_ncols: int, mask_percentage: float) -> th.Tensor:
    """Create a square mask of n_pixels in the image"""
    n_pixels = int(mask_percentage * image_nrows * image_ncols)
    square_nrows = int(n_pixels ** 0.5)
    mask = th.ones((image_nrows, image_ncols), dtype=th.float32)
    
    # Get a random top-left corner for the square
    row_idx = th.randint(0, image_ncols - square_nrows, (1,)).item()
    col_idx = th.randint(0, image_nrows - square_nrows, (1,)).item()
    
    mask[
        row_idx: row_idx + square_nrows,
        col_idx: col_idx + square_nrows
    ] = 0
    
    return mask

def generate_random_lines_mask(nrows, ncols, num_lines=1, min_thickness=1, max_thickness=5):
    """
    Generate an INVERTED binary mask with random lines (0=line, 1=background).
    
    Args:
        nrows (int): Mask height
        width (int): Mask width
        num_lines (int): Number of random lines to generate
        min_thickness (int): Minimum line thickness
        max_thickness (int): Maximum line thickness
        
    Returns:
        torch.Tensor: Inverted binary mask of shape (height, width)
    """
    # Start with all ones (background)
    mask = th.ones((nrows, ncols), dtype=th.float32)
    
    for _ in range(num_lines):
        # Random start and end points
        start_point = (random.randint(0, nrows-1), random.randint(0, ncols-1))
        end_point = (random.randint(0, nrows-1), random.randint(0, ncols-1))
        
        # Random thickness
        thickness = random.randint(min_thickness, max_thickness)
        
        # Generate the line and subtract from mask (lines become 0)
        mask = mask * (1 - generate_single_line(nrows, ncols, start_point, end_point, thickness))
        
    return mask

def generate_single_line(nrows, ncols, start_point, end_point, thickness):
    """
    Helper function to generate a single line (1=line, 0=background).
    """
    line_mask = th.zeros((nrows, ncols), dtype=th.float32)
    
    y1, x1 = start_point
    y2, x2 = end_point
    
    # Vector from start to end
    dx = x2 - x1
    dy = y2 - y1
    
    # Normalize
    length = max(math.sqrt(dx**2 + dy**2), 1e-8)
    dx /= length
    dy /= length
    
    # Generate points along the line
    num_samples = max(int(length * 2), 2)
    t_values = th.linspace(0, 1, num_samples)
    
    for t in t_values:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        # Create a grid for the thickness circle
        radius = thickness / 2
        i_values = th.arange(-thickness//2, thickness//2 + 1, dtype=th.float32)
        j_values = th.arange(-thickness//2, thickness//2 + 1, dtype=th.float32)
        
        for i in i_values:
            for j in j_values:
                if (i**2 + j**2) <= radius**2:
                    yi = int(th.round(y + i).item())
                    xi = int(th.round(x + j).item())
                    if 0 <= yi < nrows and 0 <= xi < ncols:
                        line_mask[yi, xi] = 1
                        
    return line_mask