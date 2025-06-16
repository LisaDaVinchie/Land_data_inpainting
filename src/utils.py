from pathlib import Path
import json
import argparse
import torch as th

def change_dataset_idx(dataset_path: Path, dataset_specs_path: Path, new_idx: int, dataset_specs_name = "dataset_specs") -> tuple:
    """Change the dataset index in the file names.

    Args:
        dataset_path (Path): latest dataset path
        dataset_specs_path (Path): latest dataset specs path
        new_idx (int): new dataset index

    Returns:
        tuple: new dataset path, new dataset specs path
    """
    dataset_ext = dataset_path.suffix
    dataset_name = dataset_path.stem.split("_")[0]
    new_dataset_path = dataset_path.parent / f"{dataset_name}_{new_idx}{dataset_ext}"
    
    dataset_specs_ext = dataset_specs_path.suffix
    
    new_dataset_specs_path = dataset_specs_path.parent / f"{dataset_specs_name}_{new_idx}{dataset_specs_ext}"
    
    return new_dataset_path, new_dataset_specs_path

def parse_params() -> tuple:
    """Parse command line arguments for training parameters and paths.

    Raises:
        FileNotFoundError: if the specified parameters or paths files do not exist.

    Returns:
        tuple: containing dictionaries with the parameters and paths loaded from JSON files.
    """
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')
    
    args = parser.parse_args()
    
    params_file_path = Path(args.params)
    paths_file_path = Path(args.paths)
    if not params_file_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file_path}")
    if not paths_file_path.exists():
        raise FileNotFoundError(f"Paths file not found: {paths_file_path}")
    with open(params_file_path, 'r') as f:
        params = json.load(f)
    with open(paths_file_path, 'r') as f:
        paths = json.load(f)
    
    return params, paths

def find_min_max_values(minmax_path: Path, dataset) -> tuple:
    if minmax_path.exists():
        print(f"Loading min and max values from {minmax_path}", flush=True)
        boundaries = th.load(minmax_path)
        min_val = boundaries[0]
        max_val = boundaries[1]
    else:
        print(f"Calculating min and max values from dataset", flush=True)
        images = dataset["images"]
        masks = dataset["masks"]
        images *= masks.float()  # Apply masks to images to exclude masked pixels
        
        min_val = images[:, 0:9][images[:, 0:9] > 0].min()
        max_val = images[:, 0:9][images[:, 0:9] > 0].max()
        th.save(th.tensor([min_val, max_val]), minmax_path)
        print(f"Min and max values saved to {minmax_path}", flush=True)
    return min_val, max_val