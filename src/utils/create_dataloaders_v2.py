from torch.utils.data import DataLoader
from CustomDataset_v2 import CustomDataset
import torch as th

def create_dataloaders(masked_images, inverse_masked_images, masks, train_perc, batch_size):
    assert len(masked_images) == len(masks), "The dataset and the masks must have the same length"
    assert len(masked_images) == len(inverse_masked_images), "The dataset and the inverse masked images must have the same length"
    len_dataset = len(masked_images)
    train_size = int(train_perc * len_dataset)
    indices = th.randperm(len_dataset).tolist()
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    train_set = CustomDataset(masked_images[train_indices], inverse_masked_images[train_indices], masks[train_indices])
    test_set = CustomDataset(masked_images[test_indices], inverse_masked_images[test_indices], masks[test_indices])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader