import torch as th
from pathlib import Path
import argparse
from time import time
from utils.import_params_json import load_config
from mask_data import mask_inversemask_image, create_square_mask
import random
import json

import matplotlib.pyplot as plt


start_time = time()

parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')

args = parser.parse_args()

params_path = args.params
paths_path = args.paths

processed_data_dir: Path = None
next_cutted_images_path: Path = None
cutted_txt_path: Path = None
paths = load_config(paths_path, ["data", "results"])
locals().update(paths["data"])

n_images = None
image_width = None
image_height = None
n_cutted_images = None
cutted_width = None
cutted_height = None
n_channels = None
mask_percentage = None
params = load_config(params_path, ["dataset", "mask"])
locals().update(params["dataset"])
locals().update(params["mask"])


processed_data_dir = Path(processed_data_dir)

if not processed_data_dir.exists():
    raise FileNotFoundError(f"Path {processed_data_dir} does not exist.")

# Select n_images random images from the processed images
processed_images_paths = list(processed_data_dir.glob(f"*.pt"))

if len(processed_images_paths) == 0:
    raise FileNotFoundError(f"No images found in {processed_data_dir}")

print(f"\nFound {len(processed_images_paths)} images in {processed_data_dir}\n", flush=True)


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
    
print(f"Selected random points for cutted images in {time() - idx_time} seconds\n", flush=True)

d_time = time()     
keys = ["masked_images", "inverse_masked_images", "masks"]
dataset = {cls: th.empty((n_cutted_images, n_channels, cutted_width, cutted_height), dtype=th.float32) for cls in keys}

idx = 0
for path, indices in path_to_indices.items():
    image = th.load(path)
    
    for index in indices:
        cutted_img = image[:, index[0]:index[0] + cutted_width, index[1]:index[1] + cutted_height].unsqueeze(0)
        masks = th.stack([create_square_mask(cutted_width, cutted_height, mask_percentage).unsqueeze(0) for _ in range(n_channels)], dim=1)
        dataset["masked_images"][idx], dataset["inverse_masked_images"][idx] = mask_inversemask_image(cutted_img, masks, 0)
        dataset["masks"][idx] = masks
        idx += 1


print(f"Cutted images in {time() - d_time} seconds\n", flush=True)

# Save the cutted images
save_time = time()
th.save(dataset, next_cutted_images_path)

with open(params_path, 'r') as json_file:
    data = json.load(json_file)

# Extract the "dataset" and "mask" sections
dataset_section = data.get('dataset', {})
mask_section = data.get('mask', {})

# Combine the sections into a single dictionary
sections_to_save = {
    'dataset': dataset_section,
    'mask': mask_section
}

# Save the combined sections to a text file
with open(cutted_txt_path, 'w') as txt_file:
    json.dump(sections_to_save, txt_file, indent=4)

print("Elapsed time: {:.2f} seconds\n".format(time() - start_time), flush=True)



