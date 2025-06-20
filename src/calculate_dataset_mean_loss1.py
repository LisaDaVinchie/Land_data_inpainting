import torch as th
from pathlib import Path
from losses import PerPixelMSE, PerPixelL1
from utils import parse_params, change_dataset_idx
import matplotlib.pyplot as plt

def main():
    """Main function to calculate the mean loss of a dataset."""
    params, paths = parse_params()
    
    dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
    dataset_specs_path = Path(paths["data"]["current_dataset_specs_path"])
    dataset_idx = int(params["training"]["dataset_idx"])
    
    if dataset_idx >= 0:
        dataset_path, dataset_specs_path = change_dataset_idx(dataset_path, dataset_specs_path, dataset_idx)
    
    for path in [dataset_path, dataset_specs_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file {path} not found")
        
    loss_func = PerPixelL1()
    
    # Memory mapping for large datasets
    dataset = th.load(dataset_path, map_location='cpu')
    print(f"Loaded dataset from {dataset_path}", flush=True)
    
    dataset_keys = list(dataset.keys())
    n_images = dataset[dataset_keys[0]].shape[0]
    
    # Process in batches to reduce memory usage
    c = 4
    known_channels = [0, 1, 2, 3, 5, 6, 7, 8]
    
    n_samples = 100
    
    losses = []
    
    for i in range(0, 1000):
        
        if i % 100 == 0:
            print(f"Iteration {i}", flush=True)
        # random_indexes = th.randperm(n_images)[:n_samples]
        
        # images = dataset[dataset_keys[0]][random_indexes]
        # masks = dataset[dataset_keys[1]][random_indexes]
        # nan_mask = dataset[dataset_keys[2]][random_indexes]
        
        images = th.rand(n_samples, 13, 64, 64)  # Simulated images
        masks = th.ones(n_samples, 13, 64, 64, dtype=th.bool)  # Simulated masks
        masks[:, c, 0:32, 0:32] = False  # Simulated mask for channel c
        nan_mask = th.rand(n_samples, 13, 64, 64) > 0.3  # Simulated NaN mask

        known_images = images[:, known_channels, :, :]
        
        known_images = th.where(nan_mask[:, known_channels, :, :].bool(), known_images, th.nan)
        
        # Calculate the mean image for the known channels only on non-NaN values
        mean_image = th.nanmean(known_images, dim=1, keepdim=True)
        mean_image = th.nan_to_num(mean_image, nan=-300.0)
        mean_image = th.where(masks[:, c:c+1, :, :], images[:, c:c+1, :, :], mean_image)

            
        val_mask = ~(~masks & nan_mask)
        
        total_loss = 0.0
        for i in range(mean_image.shape[0]):
            loss = loss_func(
                mean_image[i:i+1, 0], 
                images[i:i+1, c:c+1, :, :], 
                val_mask[i:i+1, c:c+1, :, :]
            ).item()
            
            n_pixels = (~val_mask[i:i+1, c, :, :]).float().sum().item()
            
            total_loss += loss / (n_pixels + 1e-8)  # Avoid division by zero
            
        
        losses.append(total_loss / mean_image.shape[0])
        print("losses", losses[-1], flush=True)
            
        # loss = loss_func(mean_image, images[:, c:c+1, :, :], val_mask[:, c:c+1, :, :]).item()
        # n_valid_pixels = (~val_mask[:, c, :, :]).float().sum().item()
        # losses.append(loss / (n_valid_pixels + 1e-8))
        # print("losses", losses[-1], flush=True)
        
        if i % 100 == 0:
            print(f"Loss for iteration {i}: {losses[-1]}", flush=True)
    
    # th.save(th.tensor(total_loss), dataset_path.parent / f"mean_loss_{dataset_idx}.pt")
    
    plt.hist(losses, bins=10)
    plt.show()

if __name__ == "__main__":
    main()