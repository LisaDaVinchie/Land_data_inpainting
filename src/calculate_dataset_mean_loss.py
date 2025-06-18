import torch as th
from pathlib import Path
from losses import PerPixelMSE
from utils import parse_params, change_dataset_idx
import matplotlib.pyplot as plt

def main():
    """Main function to calculate the mean loss of a dataset."""
    params, paths = parse_params()
    
    plot = False
    
    dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
    dataset_specs_path = Path(paths["data"]["current_dataset_specs_path"])
    dataset_idx = int(params["training"]["dataset_idx"])
    if dataset_idx >= 0:
        dataset_path, dataset_specs_path = change_dataset_idx(dataset_path, dataset_specs_path, dataset_idx)
    
    for path in [dataset_path, dataset_specs_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file {path} not found")
        
    loss_func = PerPixelMSE()
        
    dataset = th.load(dataset_path)
    
    dataset_keys = list(dataset.keys())
    
    images = dataset[dataset_keys[0]]
    masks = dataset[dataset_keys[1]]
    nan_mask = dataset[dataset_keys[2]]
    
    del dataset
    
    c = 4
    known_channels = [0, 1, 2, 3, 5, 6, 7, 8]
    
    # known_images * (~nan_mask[i:i+1, :, :, :]).float() * th.nan
    
    known_images = images[:, known_channels, :, :]
    
    # print(f"nans in known_images: {th.isnan(known_images).sum()}")
    
    # Replace nan placeholder values with NaN
    # known_images *= (~nan_mask[:, known_channels, :, :]).float() * th.nan
    # print(f"nans in known_images after replacement: {th.isnan(known_images).sum()}")
    known_images = th.where(nan_mask[:, known_channels, :, :].bool(), known_images, th.nan)
    
    # Calculate the mean image for the known channels only on non-NaN values
    mean_image = th.nanmean(known_images, dim=1, keepdim=True)
    mean_image = th.nan_to_num(mean_image, nan=-300.0)
    mean_image = th.where(masks[:, c:c+1, :, :], images[:, c:c+1, :, :], mean_image)
    
    
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
    
    loss = loss_func(mean_image, images[:, c:c+1, :, :], val_mask[:, c:c+1, :, :]).item()
    n_valid_pixels = (~val_mask[:, c, :, :]).float().sum().item()
    
    loss /= n_valid_pixels
    
    print("loss:", loss)
    
if __name__ == "__main__":
    main()
    
    
        
        