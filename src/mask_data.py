import torch as th
from pathlib import Path
import cv2

from utils.import_params_json import load_config

def apply_mask_on_channel(images: th.tensor, masks: th.tensor, placeholder: float = None) -> th.tensor:
    """Mask the image with the mask, using a placeholder. If the placeholder is none, use the mean of the level"""
    new_images = images.clone()
    if placeholder is not None:
        return new_images * masks + placeholder * (1 - masks)
    
    means = (images * masks).sum(dim=(2, 3), keepdim=True) / (masks.sum(dim=(2, 3), keepdim=True))
    return new_images * masks + means * (1 - masks)

def mask_inversemask_image(images: th.tensor, masks: th.tensor, placeholder: float = None) -> th.tensor:
    """Mask the image with the mask, using a placeholder. If the placeholder is none, use the mean of the level"""
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

def create_square_mask(image_width: int, image_height: int, mask_percentage: float) -> th.tensor:
    """Create a square mask of n_pixels in the image"""
    n_pixels = int(mask_percentage * image_width * image_height)
    square_width = int(n_pixels ** 0.5)
    mask = th.ones((image_width, image_height), dtype=th.int)
    
    # Get a random top-left corner for the square
    row_idx = th.randint(0, image_height - square_width, (1,)).item()
    col_idx = th.randint(0, image_width - square_width, (1,)).item()
    
    mask[
        row_idx: row_idx + square_width,
        col_idx: col_idx + square_width
    ] = 0
    
    return mask

class SquareMask():
    def __init__(self, params_path: Path, image_width: int = None, image_height: int = None, mask_percentage: float = None):
        """Create a square mask of n_pixels in the image"""
        
        params = load_config(params_path, ["dataset"]).get("dataset", {})
        self.image_width = image_width if image_width is not None else params.get("image_width", 10)
        self.image_height = image_height if image_height is not None else params.get("image_height", 10)
        
        params = load_config(params_path, ["square_mask"]).get("square_mask", {})
        self.mask_percentage = mask_percentage if mask_percentage is not None else params.get("mask_percentage", 10)
        

        n_pixels = int(self.mask_percentage * self.image_width * self.image_height)
        square_width = int(n_pixels ** 0.5)
        self.half_square_width = square_width // 2
    
    def mask(self) -> th.tensor:
        """Mask the image with a square"""
        mask = th.ones((self.image_width, self.image_height), dtype=th.int)
        center_row = th.randint(self.half_square_width, self.image_height - self.half_square_width, (1,)).item()
        center_col = th.randint(self.half_square_width, self.image_width - self.half_square_width, (1,)).item()
        
        mask[
            max(0, center_row - self.half_square_width): min(self.image_height, center_row + self.half_square_width),
            max(0, center_col - self.half_square_width): min(self.image_width, center_col + self.half_square_width)
        ] = 0
        
        return mask
        

# class LineMask:
#     def __init__(self, image_width: int = None, image_height: int = None, min_thickness: int = None, max_thickness: int = None, n_lines: int = None):
#         self.image_width = image_width
#         self.image_height = image_height
#         self.max_tichkness = max_thickness
#         self.min_thickness = min_thickness
#         self.n_lines = n_lines

#     def create_mask(self) -> np.ndarray:
#         ## Prepare masking matrix
        
#         mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
#         for _ in range(np.random.randint(1, self.n_lines)):
#             # Get random x locations to start line
#             x1, x2 = np.random.randint(1, self.image_width), np.random.randint(1, self.image_width)
#             # Get random y locations to start line
#             y1, y2 = np.random.randint(1, self.image_height), np.random.randint(1, self.image_height)
#             # Get random thickness of the line drawn
#             thickness = np.random.randint(self.min_thickness, self.max_tichkness)
#             # Draw black line on the white mask
#             cv2.line(mask,(x1,y1),(x2,y2),(1,1,1),thickness)

#         return mask.astype(bool)