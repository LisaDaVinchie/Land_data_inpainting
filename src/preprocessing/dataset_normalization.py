import torch as th

class MinMaxNormalization:
    def __init__(self, batch_size: int = 1000):
        self.min_val = None
        self.max_val = None
        self.batch_size = batch_size  # Default batch size for normalization
    
    def normalize(self, images: th.Tensor, masks: th.Tensor) -> tuple[th.Tensor, list]:
        """Normalize the dataset using min-max normalization.
        Exlcude from the normalization the masked pixels, leaving them out of the normalization and min and max calculation.
        Does not handle images with NaNs outside the masked pixels.

        Args:
            dataset (th.Tensor): dataset to normalize
            masks (th.Tensor): mask to use for normalization. 0 where the values are masked, 1 where the values are not masked

        Returns:
            th.Tensor: normalized dataset
            list: min and max values used for normalization
        """
        
        masks = masks.to(th.bool)

        # For small datasets, normalize the entire dataset at once
        # Faster but requires more memory
        if images.shape[0] < self.batch_size:
            # Get the min and max values of the dataset, excluding the masked pixels
            valid_pixels = images[masks]
            
            self.min_val = th.min(valid_pixels)
            self.max_val = th.max(valid_pixels)
            
            # Normalize where the mask is 1, keep the original values where the mask is 0
            norm_images = th.where(masks, (images - self.min_val) / (self.max_val - self.min_val), images)
        
        # For larger datasets, normalize in batches
        # More memory efficient but slower
        else:
            self.min_val = th.inf
            self.max_val = -th.inf
            
            # Find global min/max in batches
            for i in range(0, images.shape[0], self.batch_size):
                batch_images = images[i:i+self.batch_size]
                batch_masks = masks[i:i+self.batch_size]
                batch_valid = batch_images[batch_masks]  # Smaller temporary tensor

                if len(batch_valid) > 0:  # Avoid empty tensors
                    self.min_val = min(self.min_val, th.min(batch_valid).item())
                    self.max_val = max(self.max_val, th.max(batch_valid).item())
            
            norm_images = th.empty_like(images)
            
            scale_factor = 1 / (self.max_val - self.min_val)
            for i in range(0, images.shape[0], self.batch_size):
                if i + self.batch_size > images.shape[0]:
                    self.batch_size = images.shape[0] - i
                    
                batch_images = images[i:i+self.batch_size]
                batch_masks = masks[i:i+self.batch_size]
                norm_images[i:i+self.batch_size] = th.where(batch_masks, (batch_images - self.min_val) * scale_factor, batch_images)
        
        return norm_images, [self.min_val, self.max_val]
    
    def denormalize(self, images: th.Tensor, minmax: list = None) -> th.Tensor:
        """Denormalize the dataset using min-max denormalization.

        Args:s
            images (th.Tensor): dataset to denormalize
            minmax (list): min and max values used for normalization. If None, use the min and max values from the normalization

        Returns:
            th.Tensor: denormalized dataset
        """
        
        if minmax is None: # Use the min and max values from the normalization
            min_val = self.min_val
            max_val = self.max_val
        else: # Use the min and max values from the provided list
            min_val = minmax[0]
            max_val = minmax[1]
            
        if min_val is None or max_val is None:
            raise ValueError("Min and max values must be provided for denormalization.")
        
        return images * (max_val - min_val) + min_val