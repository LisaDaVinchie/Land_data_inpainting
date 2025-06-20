import torch as th
from pathlib import Path
from losses import PerPixelMSE
from utils import parse_params, change_dataset_idx
import matplotlib.pyplot as plt
import gc

def main():
    """Main function to calculate the mean loss of a dataset."""
    params, paths = parse_params()
    
    plot = False
    batch_size = 500  # Adjust based on your available memory
    
    dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
    dataset_specs_path = Path(paths["data"]["current_dataset_specs_path"])
    dataset_idx = int(params["training"]["dataset_idx"])
    
    if dataset_idx >= 0:
        dataset_path, dataset_specs_path = change_dataset_idx(dataset_path, dataset_specs_path, dataset_idx)
    
    for path in [dataset_path, dataset_specs_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file {path} not found")
        
    loss_func = PerPixelMSE()
    
    # Memory mapping for large datasets
    dataset = th.load(dataset_path, map_location='cpu')
    print(f"Loaded dataset from {dataset_path}", flush=True)
    
    dataset_keys = list(dataset.keys())
    total_samples = dataset[dataset_keys[0]].shape[0]
    
    # Process in batches to reduce memory usage
    c = 4
    known_channels = [0, 1, 2, 3, 5, 6, 7, 8]
    
    # Initialize variables for accumulating results
    total_loss = 0.0
    total_valid_pixels = 0
    
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        print(f"Processing batch {i} to {batch_end-1}", flush=True)
        
        # Load only the current batch
        batch_images = dataset[dataset_keys[0]][i:batch_end]
        batch_masks = dataset[dataset_keys[1]][i:batch_end]
        batch_nan_mask = dataset[dataset_keys[2]][i:batch_end]
        
        # Process known channels
        batch_known = batch_images[:, known_channels, :, :]
        batch_known = th.where(batch_nan_mask[:, known_channels, :, :].bool(), batch_known, th.nan)
        
        # Calculate batch mean
        batch_mean_img = th.nanmean(batch_known, dim=1, keepdim=True)
        batch_mean_img = th.nan_to_num(batch_mean_img, nan=-300.0)
        batch_mean_img = th.where(batch_masks[:, c:c+1, :, :], batch_images[:, c:c+1, :, :], batch_mean_img)
        
        # Calculate validation mask and loss for this batch
        batch_val_mask = ~batch_masks[:, c:c+1, :, :] & batch_nan_mask[:, c:c+1, :, :]
        # batch_loss = loss_func(
        #     batch_mean_img,
        #     batch_images[:, c:c+1, :, :],
        #     batch_val_mask[:, c:c+1, :, :]).item()
        
        squared_errors = (batch_mean_img - batch_images[:, c:c+1, :, :].float()) ** 2
        valid_errors = squared_errors[batch_val_mask]  # This flattens automatically
        
        total_loss += valid_errors.sum().item()
        # total_valid_pixels += batch_val_mask.float().sum().item()
        
        # Clean up batch variables
        del batch_images, batch_masks, batch_nan_mask, batch_known, batch_mean_img, batch_val_mask
        gc.collect()
    
    total_loss /= total_valid_pixels
        
        # Final calculation
    print(f"Final loss: {total_loss}", flush=True)
    
    th.save(th.tensor(total_loss), dataset_path.parent / f"mean_loss_{dataset_idx}.pt")
    
    if plot:
        # Plotting code remains the same, but would need to load a sample batch
        pass

if __name__ == "__main__":
    main()