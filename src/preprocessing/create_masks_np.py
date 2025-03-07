import numpy as np
from time import time
import matplotlib.pyplot as plt
import torch as th

start_time = time()

class SquareMask():
    def __init__(self, image_width: int, image_height: int, mask_percentage: float):
        """Create a square mask of n_pixels in the image"""
        self.image_width = image_width
        self.image_height = image_height
        self.mask_percentage = mask_percentage
        
        n_pixels = int(self.mask_percentage * self.image_width * self.image_height)
        self.square_width = int(n_pixels ** 0.5)
        self.half_square_width = self.square_width // 2
        
    def generate_masks(self, n_masks: int) -> np.ndarray:
        """Generate `n_masks` unique square masks"""
        masks = np.ones((n_masks, self.image_width, self.image_height), dtype=np.int32)
        
        # Randomly generate centers for each mask
        center_rows = np.random.randint(
            self.half_square_width, self.image_height - self.half_square_width, size=n_masks
        )
        center_cols = np.random.randint(
            self.half_square_width, self.image_width - self.half_square_width, size=n_masks
        )
        
        # Apply the square mask to each mask
        for i in range(n_masks):
            row_start = center_rows[i] - self.half_square_width
            row_end = center_rows[i] + self.half_square_width
            col_start = center_cols[i] - self.half_square_width
            col_end = center_cols[i] + self.half_square_width
            
            masks[i, row_start:row_end, col_start:col_end] = 0
        
        return masks

# Parameters
n_images = 100
n_channels = 13
n_lats = 3600
n_lons = 1600
mask_percentage = 0.1

# Initialize the mask generator
mask_generator = SquareMask(image_width=n_lons, image_height=n_lats, mask_percentage=mask_percentage)

# Generate all masks at once
n_masks = n_images * n_channels
print("Generating masks...")
all_masks = mask_generator.generate_masks(n_masks=n_masks)
print("Masks generated\n")

print("Reshape")
# Reshape the masks to match the tensor shape (n_images, n_channels, n_lons, n_lats)
mask_array = all_masks.reshape(n_images, n_channels, n_lons, n_lats)
print("Reshaped\n")

t_start = time()
mask_tensor = th.tensor(mask_array, dtype=th.uint8)
print("Elapsed time to create tensor: ", time() - t_start)

print("Elapsed time: ", time() - start_time)

random_fig = np.random.randint(0, n_images)
random_channel = np.random.randint(0, n_channels)

plt.imshow(mask_array[random_fig, random_channel, :, :])
plt.show()