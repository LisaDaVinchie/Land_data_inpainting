import torch as th
from pathlib import Path
import argparse
from time import time
from utils.import_params_json import load_config
from mask_data import mask_inversemask_image
import random

import matplotlib.pyplot as plt


start_time = time()

parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')

args = parser.parse_args()

params_path = args.params
paths_path = args.paths

processed_data_dir: Path = None
cutted_images_dir: Path = None
cutted_images_basename: str = None
cutted_images_file_ext: str = None
paths = load_config(paths_path, ["data", "results"])
locals().update(paths["data"])

n_images = None
image_width = None
image_height = None
n_cutted_images = None
cutted_width = None
cutted_height = None
n_channels = None
params = load_config(params_path, ["dataset"])
locals().update(params["dataset"])


processed_data_dir = Path(processed_data_dir)
cutted_images_dir = Path(cutted_images_dir)

if not processed_data_dir.exists():
    raise FileNotFoundError(f"Path {processed_data_dir} does not exist.")

if not cutted_images_dir.exists():
    raise FileNotFoundError(f"Path {cutted_images_dir} does not exist.")

# Select n_images random images from the processed images
processed_images_paths = list(processed_data_dir.glob(f"*.pt"))

if len(processed_images_paths) == 0:
    raise FileNotFoundError(f"No images found in {processed_data_dir}")

print(f"Found {len(processed_images_paths)} images in {processed_data_dir}")


# Select some random points, to use as centers for the cutted images

idx_time = time()
random_x = th.randint(0, image_width - cutted_width, (n_cutted_images,))
random_y = th.randint(0, image_height - cutted_height, (n_cutted_images,))
random_points = th.stack([random_x, random_y], dim = 1)

path_to_indices = {}
for point in random_points:
    path = random.choice(processed_images_paths)
    
    if path not in path_to_indices:
        path_to_indices[path] = []
    
    path_to_indices[path].append(point)
    
print(f"Selected random points for cutted images in {time() - idx_time} seconds")

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

d_time = time()     
keys = ["masked_images", "inverse_masked_images", "masks"]
dataset = {cls: th.empty((n_cutted_images, n_channels, cutted_width, cutted_height), dtype=th.float32) for cls in keys}

idx = 0
for path, indices in path_to_indices.items():
    image = th.load(path)
    
    for index in indices:
        cutted_img = image[:, index[0]:index[0] + cutted_width, index[1]:index[1] + cutted_height].unsqueeze(0)
        masks = th.stack([create_square_mask(cutted_width, cutted_height, 0.05).unsqueeze(0) for _ in range(n_channels)], dim=1)
        dataset["masked_images"][idx], dataset["inverse_masked_images"][idx] = mask_inversemask_image(cutted_img, masks, 0)
        dataset["masks"][idx] = masks
        idx += 1


print(f"Cutted images in {time() - d_time} seconds")

# Save the cutted images
save_time = time()

cutted_images_path = cutted_images_dir / f"{cutted_images_basename}_n{n_cutted_images}_c{n_channels}{cutted_images_file_ext}"
th.save(dataset, cutted_images_path)

print("Elapsed time: {:.2f} seconds".format(time() - start_time))



