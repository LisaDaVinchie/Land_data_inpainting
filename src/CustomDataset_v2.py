from torch.utils.data import Dataset, DataLoader, Subset
import torch as th
from pathlib import Path

class CreateDataloaders:
    def __init__(self, dataset_path: Path, train_perc: float, batch_size: int):
        """Load the dataset and create the dataloaders

        Args:
            dataset_path (Path): Path to the dataset
            train_perc (float): Percentage of the dataset to use for training
            batch_size (int): Batch size for the dataloaders

        Raises:
            ValueError: If the train_perc is not between 0 and 1
            ValueError: If the batch_size is smaller than or equal to 0
        """
        self.dataset_path = dataset_path
        self.train_perc = train_perc
        self.batch_size = batch_size
        
        if self.train_perc <=0 or self.train_perc >= 1:
            raise ValueError(f"train_perc must be between 0 and 1, got {train_perc}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")

    def create(self) -> tuple:
        """Create the dataloaders

        Returns:
            tuple: A tuple containing the training and testing dataloaders
        """
        dataset = self._load_dataset()
        
        train_indices, test_indices = self._get_train_test_indices(dataset)
        
        dataset_class = CustomDatasetClass(dataset)
        
        train_set = Subset(dataset_class, train_indices)
        test_set = Subset(dataset_class, test_indices)
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def _load_dataset(self) -> dict:
        """Loads the dataset from the given path

        Returns:
            dict: A dictionary with the following keys: 'images', 'masks'
        """
        return th.load(self.dataset_path)
    
    def _get_train_test_indices(self, dataset):
        len_dataset = self._validate_dataset_length(dataset)
            
        train_size = int(self.train_perc * len_dataset)
        indices = th.randperm(len_dataset).tolist()
        train_indices, test_indices = indices[:train_size], indices[train_size:]
        return train_indices,test_indices

    def _validate_dataset_length(self, dataset):
        dataset_keys = list(dataset.keys())
        
        if len(dataset_keys) != 2:
            raise ValueError(f"Dataset keys must be 2, got {len(dataset_keys)}")
        
        len_dataset = len(dataset[dataset_keys[0]])
        for key in dataset_keys:
            assert len(dataset[key]) == len_dataset, f"Dataset keys have different lengths"
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