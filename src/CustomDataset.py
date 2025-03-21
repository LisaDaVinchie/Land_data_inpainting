from torch.utils.data import Dataset, DataLoader, Subset
import torch as th

def create_dataloaders(dataset: dict, train_perc: float, batch_size: int) -> tuple:
    if train_perc <=0 or train_perc >= 1:
        raise ValueError(f"train_perc must be between 0 and 1, got {train_perc}")
    
    dataset_keys = list(dataset.keys())
    
    if len(dataset_keys) ==3:
        dataset_class = ExtendedDataset(dataset)
    elif len(dataset_keys) == 2:
        dataset_class = MinimalDataset(dataset)
    else:
        raise ValueError(f"Dataset keys must be 2 or 3, got {len(dataset_keys)}")
    
    len_dataset = len(dataset[dataset_keys[0]])
    for key in dataset_keys:
        assert len(dataset[key]) == len_dataset, f"Dataset keys have different lengths"
    
    train_size = int(train_perc * len_dataset)
    indices = th.randperm(len_dataset).tolist()
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    
    train_set = Subset(dataset_class, train_indices)
    test_set = Subset(dataset_class, test_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class ExtendedDataset(Dataset):
    def __init__(self, dataset: dict):
        """Custom dataset for inpainting

        Args:
            dataset (dict): A dictionary with the following keys: 'masked_images', 'inverse_masked_images', 'masks'
        """
        
        dataset_keys = list(dataset.keys())
        
        self.masked_images = dataset[dataset_keys[0]]
        self.inverse_masked_images = dataset[dataset_keys[1]]
        self.masks = dataset[dataset_keys[2]]
    
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
    

class MinimalDataset(Dataset):
    def __init__(self, dataset: dict):
        """Custom dataset for inpainting

        Args:
            dataset (dict): A dictionary with the following keys: 'images', 'masks'
        """
        
        dataset_keys = list(dataset.keys())
        
        self.image = dataset[dataset_keys[0]]
        self.mask = dataset[dataset_keys[1]]
    
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