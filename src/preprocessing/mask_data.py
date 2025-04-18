import torch as th
import random
import math
from pathlib import Path
import json

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

def initialize_mask_kind(params_path: Path, mask_kind: str):
    """Initialize the mask kind based on the provided parameters."""
    if mask_kind == "square":
        return SquareMask(params_path)
    elif mask_kind == "lines":
        return LinesMask(params_path)
    else:
        raise ValueError(f"Unknown mask kind: {mask_kind}")

class SquareMask:
    def __init__(self, params_path: Path = None, image_nrows: int = None, image_ncols: int = None, mask_percentage: float = None):
        """Initialize the SquareMask class."""
        
        self._initialize_parameters(params_path, image_nrows, image_ncols, mask_percentage)
        
        self._check_parameters()

    def _check_parameters(self):
        if self.image_nrows <= 0 or self.image_ncols <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        
        if self.mask_percentage <= 0 or self.mask_percentage >= 1:
            raise ValueError("Mask percentage must be between 0 and 1.")

    def _initialize_parameters(self, params_path, image_nrows, image_ncols, mask_percentage):
        
        if params_path is not None:
            with open(params_path, 'r') as f:
                params = json.load(f)
        
        self.image_nrows = image_nrows if image_nrows is not None else params['dataset']['cutted_nrows']
        self.image_ncols = image_ncols if image_ncols is not None else params['dataset']['cutted_ncols']
        self.mask_percentage = mask_percentage if mask_percentage is not None else params['square_mask']['mask_percentage']
        
        if self.image_nrows is None or self.image_ncols is None or self.mask_percentage is None:
            raise ValueError("Missing one of the following required parameters: image_nrows, image_ncols, mask_percentage")
        
    def mask(self):
        """Create a square mask of n_pixels in the image"""
        n_pixels = int(self.mask_percentage * self.image_nrows * self.image_ncols)
        square_nrows = int(n_pixels ** 0.5)
        image_mask = th.ones((self.image_nrows, self.image_ncols), dtype=th.float32)
        
        # Get a random top-left corner for the square
        row_idx = th.randint(0, self.image_ncols - square_nrows, (1,)).item()
        col_idx = th.randint(0, self.image_nrows - square_nrows, (1,)).item()
        
        image_mask[
            row_idx: row_idx + square_nrows,
            col_idx: col_idx + square_nrows
        ] = 0
        
        return image_mask

class LinesMask:
    def __init__(self, params_path: Path = None, image_nrows: int = None, image_ncols: int = None, num_lines: int = None, min_thickness: int = None, max_thickness: int = None):
        
        self._initialize_parameters(params_path, image_nrows, image_ncols, num_lines, min_thickness, max_thickness)
        
        self._check_parameters()

    def _check_parameters(self):
        if self.image_nrows <= 0 or self.image_ncols <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        
        if self.num_lines <= 0:
            raise ValueError("Number of lines must be a positive integer.")
        
        if self.min_thickness <= 0 or self.max_thickness <= 0:
            raise ValueError("Line thickness must be positive integers.")
        
        if self.min_thickness > self.max_thickness:
            raise ValueError("Minimum thickness cannot be greater than maximum thickness.")
        

    def _initialize_parameters(self, params_path, image_nrows, image_ncols, num_lines, min_thickness, max_thickness):
        
        if params_path is not None:
            with open(params_path, 'r') as f:
                params = json.load(f)
                
            self.image_nrows = params['dataset']['cutted_nrows']
            self.image_ncols = params['dataset']['cutted_ncols']
            self.num_lines = params['lines_mask']['num_lines']
            self.min_thickness = params['lines_mask']['min_thickness']
            self.max_thickness = params['lines_mask']['max_thickness']
        else:
            self.image_nrows = image_nrows
            self.image_ncols = image_ncols
            self.num_lines = num_lines
            self.min_thickness = min_thickness
            self.max_thickness = max_thickness
        
        if self.image_nrows is None or self.image_ncols is None or self.num_lines is None or self.min_thickness is None or self.max_thickness is None:
            raise ValueError("Missing one of the following required parameters: image_nrows, image_ncols, num_lines, min_thickness, max_thickness")
        
    def mask(self):
        """
        Generate an INVERTED binary mask with random lines (0=line, 1=background).
        
        Args:
            nrows (int): Image height
            width (int): Image width
            num_lines (int): Number of random lines to generate
            min_thickness (int): Minimum line thickness
            max_thickness (int): Maximum line thickness
            
        Returns:
            torch.Tensor: Inverted binary mask of shape (height, width)
        """
        # Start with all ones (background)
        mask = th.ones((self.image_nrows, self.image_ncols), dtype=th.float32)
        
        for _ in range(self.num_lines):
            # Random start and end points
            start_point = (random.randint(0, self.image_nrows-1), random.randint(0, self.image_ncols-1))
            end_point = (random.randint(0, self.image_nrows-1), random.randint(0, self.image_ncols-1))
            
            # Random thickness
            thickness = random.randint(self.min_thickness, self.max_thickness)
            
            # Generate the line and subtract from mask (lines become 0)
            mask = mask * (1 - self._generate_single_line(start_point, end_point, thickness))
            
        return mask

    def _generate_single_line(self, start_point: tuple, end_point: tuple, thickness: int):
        """Helper function to generate a single line (1=line, 0=background).

        Args:
            start_point (tuple): start point of the line, as (row, col)
            end_point (tuple): end point of the line, as (row, col)
            thickness (int): thickness of the line, in pixels

        Returns:
            _type_: _description_
        """
        line_mask = th.zeros((self.image_nrows, self.image_ncols), dtype=th.float32)
        
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
                        if 0 <= yi < self.image_nrows and 0 <= xi < self.image_ncols:
                            line_mask[yi, xi] = 1
                            
        return line_mask