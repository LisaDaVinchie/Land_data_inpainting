import torch as th
from time import time
from pathlib import Path
import os
import sys
import json
import argparse
# import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.import_params_json import load_config

parser = argparse.ArgumentParser(description='Create square masks for the input data')
parser.add_argument('--params', type=Path, help='Path to the parameters file', required=True)
parser.add_argument('--paths', type=Path, help='Path to the output directory', required=True)

args = parser.parse_args()

paths_file_path = args.paths
with open(paths_file_path, 'r') as file:
    paths = json.load(file)
    
masks_dir = Path(paths["data"]["masks_dir"])
masks_basename: str = paths["data"]["masks_basename"]
masks_file_ext: str = paths["data"]["masks_file_ext"]

if not masks_dir.exists():
    raise FileNotFoundError(f"Directory {masks_dir} does not exist")

params_file_path = args.params

n_images: int = None
n_channels: int = None
image_width: int = None
image_height: int = None
params = load_config(params_file_path, ["dataset"]).get("dataset", {})
locals().update(params)

mask_file_path = masks_dir / f"{masks_basename}_n{n_images}_c{n_channels}{masks_file_ext}"


mask_percentage: float = None
params = load_config(params_file_path, ["square_mask"]).get("square_mask", {})
locals().update(params)

start_time = time()

class SquareMask():
    def __init__(self, image_width: int, image_height: int, mask_percentage: float):
        """Create a square mask of n_pixels in the image"""
        self.image_width = image_width
        self.image_height = image_height
        self.mask_percentage = mask_percentage
        
        n_pixels = int(self.mask_percentage * self.image_width * self.image_height)
        self.square_width = int(n_pixels ** 0.5)
        self.half_square_width = self.square_width // 2
        
    def generate_masks(self, n_masks: int) -> th.Tensor:
        """Generate `n_masks` unique square masks as a PyTorch tensor"""
        # Initialize a tensor of ones with shape (n_masks, image_width, image_height)
        masks = th.ones((n_masks, self.image_width, self.image_height), dtype=th.uint8)
        
        # Randomly generate centers for each mask
        center_rows = th.randint(
            self.half_square_width, self.image_height - self.half_square_width, size=(n_masks,))
        center_cols = th.randint(
            self.half_square_width, self.image_width - self.half_square_width, size=(n_masks,))
        
        # Apply the square mask to each mask
        for i in range(n_masks):
            row_start = center_rows[i] - self.half_square_width
            row_end = center_rows[i] + self.half_square_width
            col_start = center_cols[i] - self.half_square_width
            col_end = center_cols[i] + self.half_square_width
            
            masks[i, row_start:row_end, col_start:col_end] = 0
        
        return masks

# Initialize the mask generator
mask_generator = SquareMask(image_width=image_height, image_height=image_width, mask_percentage=mask_percentage)

# Generate all masks at once
n_masks = n_images * n_channels
all_masks = mask_generator.generate_masks(n_masks=n_masks)

# Reshape the masks to match the tensor shape (n_images, n_channels, n_lons, n_lats)
mask_tensor = all_masks.reshape(n_images, n_channels, image_height, image_width)



th.save(mask_tensor, mask_file_path)
print(f"Saved square masks to {mask_file_path} in {time() - start_time:.2f} seconds")

