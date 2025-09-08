###############################################################################
##  
##  Dataset is non normalized, will be normalized later
##  Masks are inserted in the process
##  
###############################################################################

import torch as th
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from pathlib import Path
from time import time
# import wandb

from models import DINCAE_pconvs
from losses import PerPixelMSE

def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    __builtins__.print(*args, **kwargs)

def save_results(path, train_losses, test_losses):
    with open(path, "w") as f:
        f.write("Train losses:")
        f.write(", ".join([f"{loss:.6f}" for loss in train_losses]) + "\n")
        f.write("Test losses:")
        f.write(", ".join([f"{loss:.6f}" for loss in test_losses]) + "\n")
    
class NetCDFDataset(Dataset):
    def __init__(self, dataset, sst_var = 'sst', nanmask_var = 'nan_mask'):
        self.sst = dataset[sst_var].values   # e.g. shape [N, C, H, W]
        self.nanmask = dataset[nanmask_var].values   # e.g. shape [N]
        self.length = self.sst.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load only the required slice
        sst = th.from_numpy(self.sst[idx]).float()
        nanmask = th.from_numpy(self.nanmask[idx]).bool()
        return sst, nanmask
    

if __name__ == "__main__":
    start_time = time()
    epochs = 10
    batch_size = 32
    learning_rate = 0.00058
    ntime_win = 3
    l2_lambda = 0.0001
    model = DINCAE_pconvs(ntime_win + 4, interp_mode='nearest')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2_lambda)
    loss_fn = PerPixelMSE()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    train_dataset_path = Path('./data/minimal_datasets/dataset_proc_1.nc')
    test_dataset_path = Path('./data/minimal_datasets/dataset_proc_1_test.nc')
    if not train_dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {train_dataset_path}")
    if not test_dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {test_dataset_path}")

    result_dir = Path('./data/results')
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path('./data/weights')
    if not weights_dir.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
    
    result_list = result_dir.glob('result_*.txt')
    # Find the next available results file name
    
    
    max_idx = 0
    if len(list(result_list)) > 0:
        max_idx = max([int(f.stem.split('_')[1]) for f in result_list])
    i = max_idx + 1
    
    results_path = result_dir / f'result_{i}.txt' 
    weights_path = Path(f'./data/weights/weights_{i}.pt')

    ds = xr.load_dataset(train_dataset_path)
    ds_test = xr.load_dataset(test_dataset_path)
    print(f"Dataset loaded from {train_dataset_path} with shape {ds['sst'].shape}.\n")
    cloud_mask = th.from_numpy(ds['mask'].values).bool()
    cloud_mask_test = th.from_numpy(ds_test['mask'].values).bool()
    N_masks = cloud_mask.shape[0]
    N_masks_test = cloud_mask_test.shape[0]
    print(f"Number of masks in dataset: {N_masks}, Number of masks in test dataset: {N_masks_test}")

    train_set = NetCDFDataset(ds)
    test_set = NetCDFDataset(ds_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # wandb.init(project="SST_Inpainting", name="Model_Training", config={
    #     "epochs": epochs,
    #     "batch_size": batch_size,
    #     "learning_rate": learning_rate,
    #     "model": model.__class__.__name__,
    #     "loss_function": loss_fn.__class__.__name__,
    #     "dataset": train_dataset_path.name,
    #     "results_path": results_path.name,
    #     "weights_path": weights_path.name
    # })
    
    model.to(device)
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, nanmasks) in enumerate(train_loader):
            images = images.to(device)
            nanmasks = nanmasks.to(device)
            optimizer.zero_grad()
            mask_idx = th.randint(0, N_masks, (1,), device=device).item()
            input_mask = (nanmasks & cloud_mask[mask_idx]).to(device)
            outputs = model(images * input_mask.float(), input_mask.float())
            loss_mask = nanmasks & ~input_mask.to(device)
            loss = loss_fn(outputs, images, loss_mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        
        with th.no_grad():
            model.eval()
            epoch_test_loss = 0.0
            for batch_idx, (images, nanmasks) in enumerate(test_loader):
                images = images.to(device)
                nanmasks = nanmasks.to(device)
                mask_idx = th.randint(0, N_masks_test, (1,), device=device).item()
                input_mask = nanmasks & cloud_mask_test[mask_idx].to(device)
                outputs = model(images * input_mask.float(), input_mask.float())
                loss_mask = nanmasks & ~input_mask.to(device)
                loss = loss_fn(outputs, images, loss_mask)
                epoch_test_loss += loss.item()
            test_losses.append(epoch_test_loss / len(test_loader))
        # wandb.log({"epoch": epoch + 1, "loss": train_losses[-1], "test_loss": test_losses[-1]})
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            th.save(model.state_dict(), weights_path)
            save_results(results_path, train_losses, test_losses)
            print(f"Model weights saved to {weights_path}")
            
    save_results(results_path, train_losses, test_losses)
    th.save(model.state_dict(), weights_path)
    print(f"Training complete in {time() - start_time:.2f} seconds.")
    print(f"Final model weights saved to {weights_path}")
    print(f"Results saved to {results_path}")
