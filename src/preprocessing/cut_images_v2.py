import torch as th
from pathlib import Path
import argparse
from time import time
import random
import json
import os
import sys

path_to_append = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_to_append)
print("appended path in cut images: ", path_to_append)
from preprocessing.mask_data import mask_inversemask_image, create_square_mask


def select_random_points(original_width: int, original_height: int, n_points: int, final_width: int, final_height: int) -> th.Tensor:
    """Select random points in the original image, to use as top-left corners for the cutted images

    Args:
        original_width (int): x shape of the original image
        original_height (int): y shape of the original image
        n_points (int): number of random points to select
        final_width (int): x shape of the cutted image
        final_height (int): y shape of the cutted image

    Returns:
        th.Tensor: tensor with the selected random points, as (x, y) coordinates
    """
    random_x = th.randint(0, original_width - final_width, (n_points,))
    random_y = th.randint(0, original_height - final_height, (n_points,))
    random_points = th.stack([random_x, random_y], dim = 1)
    return random_points

def map_random_points_to_images(image_file_paths: list, selected_random_points: th.Tensor) -> dict:
    """Assign the points to some random images

    Args:
        image_file_paths (list): paths to the images
        selected_random_points (th.Tensor): points from a tensor

    Returns:
        dict: dictionary with the paths to the images as keys and the points as values
    """
    path_to_indices = {}
    for point in selected_random_points:
        path = random.choice(image_file_paths)
    
        if path not in path_to_indices:
            path_to_indices[path] = []
    
        path_to_indices[path].append(point)
    return path_to_indices
    
def generate_masked_image_dataset(original_width: int, original_height: int, n_images: int, final_width: int, final_height: int, n_channels: int, masked_fraction: float, non_masked_channels_list: list, path_to_indices_map: dict, placeholder: float = None) -> dict:
    """Generate a dataset of masked images, inverse masked images and masks

    Args:
        original_width (int): x shape of the original image
        original_height (int): y shape of the original image
        n_images (int): number of images to generate
        final_width (int): x shape of the cutted image
        final_height (int): y shape of the cutted image
        n_channels (int): total number of channels in the image
        masked_fraction (float): percentage of masked pixels in each channel
        non_masked_channels_list (list): list of channels that should not be masked
        placeholder (float): value to use as placeholder for masked pixels, None if the mean of the image should be used
        path_to_indices_map (dict): dictionary with the paths to the images as keys and the points as values

    Returns:
        dict: dictionary with the masked images, inverse masked images and masks as keys and the corresponding tensors as values
    """
    keys = ["masked_images", "inverse_masked_images", "masks"]
    dataset = {cls: th.empty((n_images, n_channels, final_width, final_height), dtype=th.float32) for cls in keys}
    non_masked_channels_list = list(non_masked_channels_list)

    idx = 0
    n_pixels = final_width * final_height * n_channels
    threshold = 0.5 * n_pixels
    for path, indices in path_to_indices_map.items():
        image = th.load(path)
    
        for index in indices:
            cutted_img = image[:, index[0]:index[0] + final_width, index[1]:index[1] + final_height].unsqueeze(0)
            nan_count = th.isnan(cutted_img).sum().item()
            if nan_count > threshold:
                while nan_count > threshold:
                    index = select_random_points(original_width, original_height, 1, final_width, final_height)[0]
                    cutted_img = image[:, index[0]:index[0] + final_width, index[1]:index[1] + final_height].unsqueeze(0)
                    nan_count = th.isnan(cutted_img).sum().item()
                
            masks = th.stack([create_square_mask(final_width, final_height, masked_fraction).unsqueeze(0) for _ in range(n_channels)], dim=1)
            masks[:, non_masked_channels_list, :, :] = th.ones((1, final_width, final_height), dtype=th.float32)
        
        
            dataset["masked_images"][idx], dataset["inverse_masked_images"][idx] = mask_inversemask_image(cutted_img, masks, placeholder)
            dataset["masks"][idx] = masks
            idx += 1
    return dataset

def main():
    start_time = time()
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')

    args = parser.parse_args()

    params_path = args.params
    paths_path = args.paths
    
    with open(paths_path, 'r') as json_file:
        paths = json.load(json_file)
        
    processed_data_dir = paths["data"]["processed_data_dir"]
    next_cutted_images_path = paths["data"]["next_cutted_images_path"]
    cutted_txt_path = paths["data"]["cutted_txt_path"]
    
    
    
    print(f"Found paths: {processed_data_dir}, {next_cutted_images_path}, {cutted_txt_path}\n", flush=True)

    with open(params_path, 'r') as json_file:
        params = json.load(json_file)
    
    x_shape_raw = params["dataset"]["x_shape_raw"]
    y_shape_raw = params["dataset"]["y_shape_raw"]
    n_cutted_images = params["dataset"]["n_cutted_images"]
    cutted_width = params["dataset"]["cutted_width"]
    cutted_height = params["dataset"]["cutted_height"]
    n_channels = params["dataset"]["n_channels"]
    non_masked_channels = params["dataset"]["non_masked_channels"]
    
    mask_percentage = params["mask"]["mask_percentage"]
    placeholder = params["mask"]["placeholder"]

    if placeholder == False:
        placeholder = None


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
    random_points = select_random_points(original_width=x_shape_raw, original_height=y_shape_raw,
                                        n_points=n_cutted_images, final_width=cutted_width,
                                        final_height=cutted_height)
    print(f"Selected random points for cutted images in {time() - idx_time} seconds\n", flush=True)

    path_to_indices = map_random_points_to_images(processed_images_paths, random_points)

    d_time = time() 
    dataset = generate_masked_image_dataset(original_width=x_shape_raw, original_height=y_shape_raw,
                                            n_images=n_cutted_images, final_width=cutted_width,
                                            final_height=cutted_height, n_channels=n_channels,
                                            masked_fraction=mask_percentage, non_masked_channels_list=non_masked_channels,
                                            path_to_indices_map=path_to_indices, placeholder=placeholder)

    print(f"Cutted images in {time() - d_time} seconds\n", flush=True)

    # Save the cutted images
    save_time = time()
    th.save(dataset, next_cutted_images_path)

    with open(params_path, 'r') as json_file:
        paths = json.load(json_file)

    # Extract the "dataset" and "mask" sections
    dataset_section = paths.get('dataset', {})
    mask_section = paths.get('mask', {})

    # Combine the sections into a single dictionary
    sections_to_save = {
        'dataset': dataset_section,
        'mask': mask_section
    }

    # Save the combined sections to a text file
    with open(cutted_txt_path, 'w') as txt_file:
        json.dump(sections_to_save, txt_file, indent=4)
    print(f"Saved cutted images in {time() - save_time} seconds\n", flush=True)

    print("Elapsed time: {:.2f} seconds\n".format(time() - start_time), flush=True)
    
if __name__ == "__main__":
    main()