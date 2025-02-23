import torch as th
from pathlib import Path
import cv2
from pathlib import Path

from utils.import_params_json import load_config

class SquareMask():
    def __init__(self, params_path: Path, image_width: int = None, image_height: int = None, mask_percentage: int = None):
        """Create a square mask of n_pixels in the image"""
        
        params = load_config(params_path, ["dataset"]).get("dataset", {})
        self.image_width = image_width if image_width is not None else params.get("image_width", 10)
        self.image_height = image_height if image_height is not None else params.get("image_height", 10)
        
        params = load_config(params_path, ["square_mask"]).get("square_mask", {})
        self.mask_percentage = mask_percentage if mask_percentage is not None else params.get("mask_percentage", 10)
        

        n_pixels = int(self.mask_percentage * self.image_width * self.image_height)
        square_width = int(n_pixels ** 0.5)
        self.half_square_width = square_width // 2
    
    def mask_image(self, image: th.tensor) -> th.tensor:
        """Mask the image with a square"""
        with th.no_grad():
            mask = th.zeros((self.image_width, self.image_height), dtype=th.bool)
            center_row = th.randint(self.image_height - self.half_square_width, self.half_square_width, (1,)).item()
            center_col = th.randint(self.image_width - self.half_square_width, self.half_square_width, (1,)).item()
        
        start_row = center_row - self.half_square_width
        end_row = center_row + self.half_square_width
        start_col = center_col - self.half_square_width
        end_col = center_col + self.half_square_width
        
        mask[start_row:end_row, start_col:end_col] = True
        
        masked_image = image * mask
        
        return masked_image, mask
        

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