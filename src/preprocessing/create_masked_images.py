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
    
processed_data_dir: Path = Path(paths["data"]["processed_data_dir"])

if not processed_data_dir.exists():
    raise FileNotFoundError(f"Directory {processed_data_dir} does not exist")

params_file_path = args.params

n_images: int = None
n_channels: int = None
image_width: int = None
image_height: int = None
params = load_config(params_file_path, ["dataset"]).get("dataset", {})
locals().update(params)

dataset_path = processed_data_dir.parent / f"dataset_n{n_images}_c{n_channels}.pt"


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
mask_generator = SquareMask(image_width=image_width, image_height=image_height, mask_percentage=mask_percentage)

# Generate all masks at once
n_masks = n_images * n_channels
all_masks = mask_generator.generate_masks(n_masks=n_masks)

# Reshape the masks to match the tensor shape (n_images, n_channels, n_lons, n_lats)
mask_tensor = all_masks.reshape(n_images, n_channels, image_width, image_height)

print("Masks created in {:.2f} seconds".format(time() - start_time))

# def apply_mask_on_channel(images: th.tensor, masks: th.tensor, placeholder: float = None) -> th.tensor:
#     """Mask the image with the mask, using a placeholder. If the placeholder is none, use the mean of the level"""
#     new_images = images.clone()
#     if placeholder is not None:
#         return new_images * masks + placeholder * (1 - masks)
    
#     masked_sum = (images * masks).sum(dim=(2, 3))
#     means = (images * masks).sum(dim=(2, 3), keepdim=True) / (masks.sum(dim=(2, 3), keepdim=True))
#     return new_images * masks + means * (1 - masks)

def apply_mask_on_channel(images: th.Tensor, masks: th.Tensor, placeholder: float = None) -> tuple[th.Tensor, th.Tensor]:
    """
    Mask the image with the mask, using a placeholder for both masked and inverse-masked regions.
    Returns a tuple of (masked_images, inverse_masked_images).
    """
    # Compute image * mask and image * (1 - mask)
    masked_images = images * masks
    inverse_masked_images = images * (1 - masks)
    
    if placeholder is not None:
        # Apply placeholder to both masked and inverse-masked regions
        masked_images = masked_images + placeholder * (1 - masks)
        inverse_masked_images = inverse_masked_images + placeholder * masks
    else:
        # Compute the mean of the masked regions efficiently
        masked_sum = masked_images.sum(dim=(2, 3), keepdim=True)
        mask_sum = masks.sum(dim=(2, 3), keepdim=True)
        means = masked_sum / mask_sum
        
        # Apply the mean to the inverse-masked regions
        inverse_masked_images = inverse_masked_images + means * masks
    
    return masked_images, inverse_masked_images

images_paths = list(processed_data_dir.glob("*.pt"))[:n_images]

print("Stacking images...", flush=True)
s_time = time()
images = th.stack([th.load(file) for file in images_paths])
print(f"Images stacked in {time() - s_time:.2f} seconds", flush=True)

m_time = time()
masked_images, inverse_masked_images = apply_mask_on_channel(images, mask_tensor)
print(f"Images masked in {time() - m_time:.2f} seconds", flush=True)

# i_time = time()
# inverse_masked_images = apply_mask_on_channel(masked_images, 1 - mask_tensor)
# print(f"Images inverse masked in {time() - i_time:.2f} seconds", flush=True)

s_time = time()
th.save({"masked_images": masked_images, "inverse_masked_images": inverse_masked_images, "masks": mask_tensor}, dataset_path)
print(f"Dataset saved in {time() - s_time:.2f} seconds", flush=True)

print(f"Dataset created in {time() - start_time:.2f} seconds", flush=True)




