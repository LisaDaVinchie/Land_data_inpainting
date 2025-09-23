import torch as th

def calculate_mean_image(images, masks, nan_masks, reconstruct: bool = False, c = 4, known_channels = [0, 1, 2, 3, 5, 6, 7, 8]):
    known_images = images[:, known_channels, :, :]
        
    # Set nan pixels to nan
    known_images = th.where(nan_masks[:, known_channels, :, :], known_images, th.nan)
        
    # Calculate batch mean
    mean_images = th.nanmean(known_images, dim=1, keepdim=True)
    
    if reconstruct:
        mean_images = th.where((~masks & nan_masks)[:, c:c+1, :, :], mean_images, images[:, c:c+1, :, :])
    else:
        mean_images = th.where((~masks & nan_masks)[:, c:c+1, :, :], mean_images, th.zeros_like(mean_images))
    mean_images = th.nan_to_num(mean_images, nan=0.0, posinf=0.0, neginf=0.0)
    return mean_images