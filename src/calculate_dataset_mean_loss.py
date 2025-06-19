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
    n_images = dataset[dataset_keys[0]].shape[0]
    
    # Process in batches to reduce memory usage
    c = 4
    known_channels = [0, 1, 2, 3, 5, 6, 7, 8]
    
    n_samples = 100
    random_indexes = th.randperm(n_images)[:n_samples]
    print(f"Randomly selected {n_samples} samples from {n_images} total images", flush=True)
    # print("random_indexes:", random_indexes, flush=True)
    
    if n_images < batch_size:
        images = dataset[dataset_keys[0]][random_indexes]
        masks = dataset[dataset_keys[1]][random_indexes]
        nan_mask = dataset[dataset_keys[2]][random_indexes]
        print("Loaded entire dataset into memory", flush=True)
    
        known_images = images[:, known_channels, :, :]
        
        known_images = th.where(nan_mask[:, known_channels, :, :].bool(), known_images, th.nan)
        print("Found known images", flush=True)
        
        # Calculate the mean image for the known channels only on non-NaN values
        mean_image = th.nanmean(known_images, dim=1, keepdim=True)
        print("Calculated mean image", flush=True)
        mean_image = th.nan_to_num(mean_image, nan=-300.0)
        print("Replaced NaN values in mean image", flush=True)
        mean_image = th.where(masks[:, c:c+1, :, :], images[:, c:c+1, :, :], mean_image)
        print("Applied masks to mean image", flush=True)
        
        if plot:
            random_idx = th.randint(0, images.shape[0], (1,)).item()
            nan_mean_image = th.where(nan_mask[random_idx, c, :, :],  mean_image[random_idx, 0], th.nan)
            nan_original_image = th.where(nan_mask[random_idx, c, :, :], images[random_idx, c, :, :], th.nan)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(nan_original_image.cpu().numpy())
            axs[0].set_title("Original Image")
            axs[1].imshow(nan_mean_image.cpu().numpy())
            axs[1].set_title("Mean Image")
            axs[2].imshow(masks[random_idx, c, :, :].cpu().numpy())
            axs[2].set_title("Mask")
            plt.show()
            
        val_mask = ~(~masks & nan_mask)
        print("Calculated validation mask", flush=True)
        
        loss = loss_func(mean_image, images[:, c:c+1, :, :], val_mask[:, c:c+1, :, :]).item()
        print("Calculated loss", flush=True)
        n_valid_pixels = (~val_mask[:, c, :, :]).float().sum().item()
        print("Number of valid pixels:", n_valid_pixels, flush=True)
        
        total_loss = loss / n_valid_pixels
        print("Normalized loss by number of valid pixels", flush=True)
    
    
        
        
    else:
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