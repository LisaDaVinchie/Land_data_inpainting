from torch.utils.data import Dataset, DataLoader, Subset
import torch as th
from pathlib import Path

class CreateDataloaders:
    def __init__(self, train_perc: float, batch_size: int = None):
        """Load the dataset and create the dataloaders

        Args:
            train_perc (float): Percentage of the dataset to use for training
            batch_size (int): Batch size for the dataloaders

        Raises:
            ValueError: If the train_perc is not between 0 and 1
            ValueError: If the batch_size is smaller than or equal to 0
        """
        self.train_perc = train_perc
        self.batch_size = batch_size
        
        if self.train_perc <=0 or self.train_perc >= 1:
            raise ValueError(f"train_perc must be between 0 and 1, got {self.train_perc}")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {self.batch_size}")

    def create(self, dataset: dict, batch_size: int = None) -> tuple:
        """Create the dataloaders

        Returns:
            tuple: A tuple containing the training and testing dataloaders
        """
        

        self.batch_size = batch_size if self.batch_size is None else self.batch_size
        
        if self.batch_size is None or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")
        
        train_indices, test_indices = self._get_train_test_indices(dataset)
        
        dataset_class = CustomDatasetClass(dataset)
        
        train_set = Subset(dataset_class, train_indices)
        test_set = Subset(dataset_class, test_indices)
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def load_dataset(self, dataset_path: Path) -> dict:
        """Loads the dataset from the given path.
        
        Args:
            dataset_path (Path): Path to the dataset.

        Returns:
            dict: A dictionary with the following keys: 'images', 'masks'
        """
        
        return th.load(dataset_path)
    
    def _get_train_test_indices(self, dataset):
        len_dataset = self._validate_dataset_length(dataset)
            
        train_size = int(self.train_perc * len_dataset)
        indices = th.randperm(len_dataset).tolist()
        train_indices, test_indices = indices[:train_size], indices[train_size:]
        return train_indices,test_indices

    def _validate_dataset_length(self, dataset):
        dataset_keys = list(dataset.keys())
        
        if len(dataset_keys) < 3:
            raise ValueError(f"Dataset keys must be at least 3, got {len(dataset_keys)}")
        
        len_dataset = len(dataset[dataset_keys[0]])
        for key in dataset_keys[:3]:
            if len(dataset[key]) != len_dataset:
                raise(ValueError("Datset lengths must be equal\n"))
        return len_dataset
    

class CustomDatasetClass(Dataset):
    def __init__(self, dataset: dict):
        """Custom dataset for inpainting

        Args:
            dataset (dict): A dictionary with the following keys: 'images', 'masks'
        """
        
        dataset_keys = list(dataset.keys())
        
        self.image = dataset[dataset_keys[0]]
        self.mask = dataset[dataset_keys[1]]
        self.nanmask = dataset[dataset_keys[2]]
    
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
        return self.image[idx], self.mask[idx], self.nanmask[idx]