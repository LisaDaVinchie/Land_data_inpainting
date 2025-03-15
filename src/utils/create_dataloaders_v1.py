from torch.utils.data import DataLoader
from CustomDataset_v1 import CustomDataset
import torch as th

def create_dataloaders(dataset, masks, train_perc, batch_size):
    assert len(dataset) == len(masks), "The dataset and the masks must have the same length"
    len_dataset = len(dataset)
    train_size = int(train_perc * len_dataset)
    indices = th.randperm(len_dataset).tolist()
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    train_set = CustomDataset(dataset[train_indices], masks[train_indices])
    test_set = CustomDataset(dataset[test_indices], masks[test_indices])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader