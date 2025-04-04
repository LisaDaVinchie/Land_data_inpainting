import torch as th
from pathlib import Path
import argparse
from time import time
from datetime import datetime
import random
import json
import os
import sys
import math

path_to_append = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_to_append)
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
    
    chosen_paths = [random.choice(image_file_paths) for _ in range(len(selected_random_points))]
    path_to_indices = {}
    
    for path, point in zip(chosen_paths, selected_random_points):
        if path not in path_to_indices:
            path_to_indices[path] = []
    
        path_to_indices[path].append(point)
    return path_to_indices

def generate_image_dataset(original_width: int, original_height: int, n_images: int, final_width: int, final_height: int, n_channels: int, masked_fraction: float, masked_channels_list: list, path_to_indices_map: dict, nans_threshold: float, extended_data: bool, minimal_data: bool, placeholder: float = None) -> tuple[dict, dict, th.Tensor]:
    """Generate a dataset of masked images, inverse masked images and masks

    Args:
        original_width (int): x shape of the original image
        original_height (int): y shape of the original image
        n_images (int): number of images to generate
        final_width (int): x shape of the cutted image
        final_height (int): y shape of the cutted image
        n_channels (int): final number of channels in the image
        masked_fraction (float): percentage of masked pixels in each channel
        masked_channels_list (list): list of channels that should not be masked
        path_to_indices_map (dict): dictionary with the paths to the images as keys and the points as values
        nans_threshold (float): threshold of nans in the image
        extended_data (bool): True if the extended dataset should be generated
        minimal_data (bool): True if the minimal dataset should be generated
        placeholder (float): value to use as placeholder for masked pixels, None if the mean of the image should be used

    Returns:
        tuple: dictionary with the masked images, inverse masked images and masks as keys and the corresponding tensors as values
                and dictionary with the images and masks as keys and the corresponding tensors as values
    """
    
    # Initialize the two datasets
    dataset_ext, dataset_min = None, None
    if extended_data:
        keys_ext = ["masked_images", "inverse_masked_images", "masks"]
        dataset_ext = {cls: th.empty((n_images, n_channels, final_width, final_height), dtype=th.float32) for cls in keys_ext}
    if minimal_data:
        keys_min = ["images", "masks"]
        dataset_min = {cls: th.empty((n_images, n_channels, final_width, final_height), dtype=th.float32) for cls in keys_min}
        
    nans_masks = th.ones((n_images, n_channels, final_width, final_height), dtype=th.float32)
    
    masked_channels_list = list(masked_channels_list)
    
    # Calculate the interval between the oldest and newest date, in days
    selected_dates = [Path(path).stem for path in path_to_indices_map.keys()]
    dates_parsed = [datetime.strptime(d, "%Y_%m_%d").date() for d in selected_dates]
    oldest_date, newest_date = min(dates_parsed), max(dates_parsed)
    interval = (newest_date - oldest_date).days

    idx_start = 0 # index of the first cutted image of this raw image
    idx_end = 0 # index of the last cutted image of this raw image
    n_original_channels = n_channels - 1 # The last channel is the time layer
    n_pixels = final_width * final_height * n_original_channels # number of pixels in the raw image
    threshold = nans_threshold * n_pixels # threshold of nans in the image
    for path, indices in path_to_indices_map.items():
        image = th.load(path)
        n_indices = len(indices)
        idx_end = idx_start + n_indices
        
        # Add the time layer to the images
        days_from_inital_date = (datetime.strptime(Path(path).stem, "%Y_%m_%d").date() - oldest_date).days
        encoded_time = math.cos(math.pi * days_from_inital_date / interval)
        
        # Generate the cutted images, adding the time layer
        cutted_imgs = th.stack([cut_valid_image(original_width, original_height, final_width, final_height, threshold, image, index) for index in indices], dim=0)
        time_layers = th.ones((n_indices, 1, final_width, final_height), dtype=th.float32) * encoded_time
        cutted_imgs = th.cat((cutted_imgs, time_layers), dim=1)
        
        # Find where the nans are in the cutted images
        cutted_img_nans = ~th.isnan(cutted_imgs)
        nans_masks[idx_start:idx_end, :, :, :] = cutted_img_nans.float()
        
        # Create square masks. 0 where the values are masked, 1 where the values are not masked
        masks = th.ones((n_indices, n_channels, final_width, final_height), dtype=th.float32)
        for mc in masked_channels_list:
            masks[:, mc, :, :] = create_square_mask(final_width, final_height, masked_fraction)
        
        # Set masks to 0 where the nan mask is 0
        masks = th.where(cutted_img_nans == 0, th.tensor(0, dtype=masks.dtype), masks)
        
        # Save the images to the minimal dataset, substituting the nans with the placeholder
        if minimal_data:
            dataset_min[keys_min[0]][idx_start:idx_end] = th.nan_to_num(cutted_imgs, nan=placeholder)
            dataset_min[keys_min[1]][idx_start:idx_end] = masks
        
        # Save the images to the extended dataset
        if extended_data:
            masked_images, inverse_masked_images = mask_inversemask_image(cutted_imgs, masks, placeholder)
            dataset_ext[keys_ext[0]][idx_start:idx_end] = masked_images
            dataset_ext[keys_ext[1]][idx_start:idx_end] = inverse_masked_images
            dataset_ext[keys_ext[2]][idx_start:idx_end] = masks
        
        idx_start = idx_end
        
    return dataset_ext, dataset_min, nans_masks

def cut_valid_image(original_width: int, original_height: int, final_width: int, final_height: int, threshold: float, image: th.Tensor, index: int):
    """Cut a valid image from the original image

    Args:
        original_width (int): _description_
        original_height (int): _description_
        final_width (int): _description_
        final_height (int): _description_
        threshold (float): _description_
        image (th.Tensor): _description_
        index (int): _description_

    Returns:
        _type_: _description_
    """
    cutted_img = image[:, index[0]:index[0] + final_width, index[1]:index[1] + final_height]
    nan_count = th.isnan(cutted_img).sum().item()
    if nan_count > threshold:
        while nan_count > threshold:
            index = select_random_points(original_width, original_height, 1, final_width, final_height)[0]
            cutted_img = image[:, index[0]:index[0] + final_width, index[1]:index[1] + final_height]
            nan_count = th.isnan(cutted_img).sum().item()
    return cutted_img

def check_dirs_existance(dirs: list[Path]):
    """Check if the directories exist

    Args:
        dirs (list[Path]): list of directories to check

    Raises:
        FileNotFoundError: if a directory does not exist
    """
    for dir in dirs:
        if not dir.exists():
            raise FileNotFoundError(f"Folder {dir} does not exist.")

def main():
    start_time = time()
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')

    args = parser.parse_args()

    params_path = args.params
    paths_path = args.paths
    
    with open(paths_path, 'r') as json_file:
        json_paths = json.load(json_file)
        
    processed_data_dir = Path(json_paths["data"]["processed_data_dir"])
    next_extended_dataset_path = Path(json_paths["data"]["next_extended_dataset_path"])
    next_minimal_dataset_path = Path(json_paths["data"]["next_minimal_dataset_path"])
    dataset_specs_path = Path(json_paths["data"]["dataset_specs_path"])
    next_nans_masks_path = Path(json_paths["data"]["next_nans_masks_path"])
    
    # Check if the directories exist
    check_dirs_existance([processed_data_dir, next_extended_dataset_path.parent, next_minimal_dataset_path.parent, dataset_specs_path.parent])

    with open(params_path, 'r') as json_file:
        params = json.load(json_file)
    
    x_shape_raw = int(params["dataset"]["x_shape_raw"])
    y_shape_raw = int(params["dataset"]["y_shape_raw"])
    n_cutted_images = int(params["dataset"]["n_cutted_images"])
    cutted_width = int(params["dataset"]["cutted_width"])
    cutted_height = int(params["dataset"]["cutted_height"])
    n_channels = int(params["dataset"]["n_channels"])
    masked_channels = list(params["dataset"]["masked_channels"])
    nans_threshold = float(params["dataset"]["nans_threshold"])
    minimal_dataset = bool(params["dataset"]["minimal_dataset"])
    extended_dataset = bool(params["dataset"]["extended_dataset"])
    
    mask_percentage = float(params["mask"]["mask_percentage"])
    placeholder = float(params["mask"]["placeholder"])
    
    if minimal_dataset == False and extended_dataset == False:
        raise ValueError("Both minimal_dataset and extended_dataset are False. At least one of them should be True.")

    if placeholder == False:
        placeholder = None

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
    
    # Generate the dataset
    dataset_ext, dataset_min, nans_masks = generate_image_dataset(original_width=x_shape_raw, original_height=y_shape_raw,
                                                    n_images=n_cutted_images, final_width=cutted_width,
                                                    final_height=cutted_height, n_channels=n_channels,
                                                    masked_fraction=mask_percentage, masked_channels_list=masked_channels,
                                                    path_to_indices_map=path_to_indices, nans_threshold=nans_threshold,
                                                    minimal_data=minimal_dataset, extended_data=extended_dataset,
                                                    placeholder=placeholder)
    print(f"Generated the dataset in {time() - d_time} seconds\n", flush=True)
    
    if dataset_ext is not None:
        th.save(dataset_ext, next_extended_dataset_path)
        print(f"Saved the extended dataset to {next_extended_dataset_path}\n", flush=True)
    if dataset_min is not None:
        th.save(dataset_min, next_minimal_dataset_path)
        print(f"Saved the minimal dataset to {next_minimal_dataset_path}\n", flush=True)
        
    th.save(nans_masks, next_nans_masks_path)
    print(f"Saved the nans masks to {next_nans_masks_path}\n", flush=True)

    # Extract the "dataset" and "mask" sections
    dataset_section = json_paths.get('dataset', {})
    mask_section = json_paths.get('mask', {})

    # Combine the sections into a single dictionary
    sections_to_save = {
        'dataset': dataset_section,
        'mask': mask_section
    }

    # Save the combined sections to a text file
    with open(dataset_specs_path, 'w') as txt_file:
        json.dump(sections_to_save, txt_file, indent=4)

    print("Elapsed time: {:.2f} seconds\n".format(time() - start_time), flush=True)
    
if __name__ == "__main__":
    main()