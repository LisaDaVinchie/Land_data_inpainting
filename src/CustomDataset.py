from torch.utils.data import Dataset
import torch as th

class CustomDataset(Dataset):
    def __init__(self, images: th.tensor, masks: th.tensor):
        """Custom dataset for inpainting

        Args:
            dataset (dict): A dictionary with the following keys: 'masked_image', 'target'
        """
        
        self.image = images
        self.mask = masks
    
    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.image)
    
    def __getitem__(self, idx: int) -> tuple:
        """Returns the name, image and target at the given index

        Args:
            idx (int): The index of the item to return

        Returns:
            tuple: A tuple containing the image and masked image
        """
        return self.image[idx], self.mask[idx]