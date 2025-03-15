from torch.utils.data import Dataset
import torch as th

class CustomDataset(Dataset):
    def __init__(self, masked_images: th.tensor, inverse_masked_images: th.tensor, masks: th.tensor):
        """Custom dataset for inpainting

        Args:
            dataset (dict): A dictionary with the following keys: 'masked_image', 'target'
        """
        
        self.masked_images = masked_images
        self.inverse_masked_images = inverse_masked_images
        self.masks = masks
    
    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.masked_images)
    
    def __getitem__(self, idx: int) -> tuple:
        """Returns the name, image and target at the given index

        Args:
            idx (int): The index of the item to return

        Returns:
            tuple: A tuple containing the image and masked image
        """
        return self.masked_images[idx], self.inverse_masked_images[idx], self.masks[idx]