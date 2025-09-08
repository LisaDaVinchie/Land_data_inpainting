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
import wandb

from models import DINCAE_pconvs
from losses import PerPixelMSE
import data

def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    __builtins__.print(*args, **kwargs)

def main():

    # train_loader = create_dataloader(dataset_path, batch_size=batch_size, split='train', shuffle=True)
    # test_loader = create_dataloader(dataset_path, batch_size=batch_size, split='test', shuffle=False)
    # print("DataLoader created for single-GPU or CPU training.")
    
    
    # lr_lambda = lambda step: 2 ** -(step // step_size)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    wandb.init(project="SST_Inpainting", name="Model_Training", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model": model.__class__.__name__,
        "loss_function": loss_fn.__class__.__name__,
        "dataset": dataset_path.name,
        "results_path": results_path.name,
        "weights_path": weights_path.name
    })
    
    # train = TrainModel(
    #     model=model,
    #     loss_fn=loss_fn,
    #     optimizer=optimizer,
    #     model_save_path=weights_path,
    #     results_path=results_path
    # )
    
    # print("Starting training...")
    # train.train(train_loader, test_loader, epochs=epochs, scheduler=scheduler)
    # train.save_results()
    # print(f"Results saved to {results_path}")
    # th.save(model.state_dict(), 'model.pth')
    # print(f"Training completed and model saved in {time() - start_time:.2f} seconds.\n")

# def create_dataloader(dataset_path, batch_size, split='train', shuffle=True):
#     dataset = SSTDataset(dataset_path, split=split)
#     print(f"Dataset loaded with {len(dataset)} samples.")
    
#     # For single-GPU or CPU training:
#     loader = DataLoader(
#         dataset,
#         batch_size= batch_size,
#         shuffle= shuffle,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     return loader

# class TrainModel:
#     def __init__(self, model, loss_fn, optimizer, model_save_path, results_path):
#         self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
#         self.model = model.to(self.device)
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer
            
#         self.clip_value = 5.0  # Gradient clipping value
#         self.n_days = 9  # Number of days in the dataset
#         self.current_day_channel = self.n_days // 2
#         self.save_every = 1  # Save model every 10 epochs
        
#         self.train_losses = []
#         self.test_losses = []
#         self.lr = []
        
#         self.model_save_path = Path(model_save_path)
#         self.results_path = Path(results_path)

#     def train(self, train_loader, test_loader, epochs, scheduler=None):

#         for epoch in range(epochs):
#             print(f"Epoch {epoch+1}/{epochs}")
            
#             train_loss = self.step(train_loader, backprop=True)
#             self.train_losses.append(train_loss)
            
#             if scheduler:
#                 scheduler.step()
#                 print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
#             with th.no_grad():
#                 test_loss = self.step(test_loader, backprop=False)
#             self.test_losses.append(test_loss)
            
#             self.lr.append(self.optimizer.param_groups[0]['lr'])
            
#             min_epoch = min(5, epoch + 1)
#             test_loss_avg = sum(self.test_losses[-min_epoch:]) / min_epoch
            
#             wandb.log({
#                 "epoch": epoch + 1,
#                 "train_loss": train_loss,
#                 "test_loss": test_loss,
#                 "learning_rate": self.optimizer.param_groups[0]['lr'],
#                 "test_loss_avg": test_loss_avg
#             })
            
#             wandb.watch(self.model, log="all")
            
#             if (epoch + 1) % self.save_every == 0:
#                 th.save(self.model.state_dict(), self.model_save_path)
#                 self.save_results()

#             print()
    
#     def step(self, loader, backprop=True):
#         self.model.train() if backprop else self.model.eval()
        
#         total_loss = 0.0
#         for (img, mask, nanmask) in loader:
#             img, mask, nanmask = img.to(self.device), mask.to(self.device), nanmask.to(self.device)

#             output = self.model(img * mask.float(), (mask & nanmask).float())
            
#             loss = self.loss_fn(output[:, 0:1],
#                                 img[:, self.current_day_channel:self.current_day_channel + 1],
#                                 self.validation_mask(
#                                     mask[:, self.current_day_channel: self.current_day_channel + 1],
#                                     nanmask[:, self.current_day_channel: self.current_day_channel + 1]
#                                     )
#                                 )
#             total_loss += loss.item()
            
#             if backprop:
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 th.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
#                 self.optimizer.step()
                
#         return total_loss / len(loader.dataset)
    
#     def validation_mask(self, masks: th.Tensor, nan_masks: th.Tensor, loss: bool = True):
#         """Calculate the mask used to calculate the loss, i.e. where the pixel is masked but not nan.

#         Args:
#             masks (th.Tensor): masks tensor
#             nan_masks (th.Tensor): nan masks tensor

#         Returns:
#             th.Tensor: validation mask
#         """
#         return ~(~masks & nan_masks) if loss else (~masks & nan_masks)
    
    
#     def save_results(self, elapsed_time: float = None):
#         """Save the training results to a file.

#         Args:
#             elapsed_time (float): elapsed time of the training
#         """
        
#         # Save the train losses to a txt file
#         with open(self.results_path, 'w') as f:
#             if elapsed_time is not None:
#                 f.write("Elapsed time [s]:\n")
#                 f.write(f"{elapsed_time}\n\n")
#             f.write("Train losses\n")
#             for loss in self.train_losses:
#                 f.write(f"{loss}\t")
#             f.write("\n\n")
#             f.write("Test losses\n")
#             for loss in self.test_losses:
#                 f.write(f"{loss}\t")
#             f.write("\n\n")
#             f.write("Learning rate\n")
#             for lr in self.lr:
#                 f.write(f"{lr}\t")

    
# class SSTDataset(Dataset):
#     """
#     Custom PyTorch Dataset for NetCDF SST data with time, lat, lon dimensions.
#     """

#     def __init__(self, dataset):
#         self.sst_data = dataset['sst'].values  # Shape: (time, lat, lon)
#         # Handle NaN values
#         self.sst_data = np.nan_to_num(self.sst_data, nan=-2)
#         self.n_samples = self.sst_data.shape[0]

#     def __len__(self) -> int:
#         return self.n_samples
    
#     def __getitem__(self, idx: int) -> th.Tensor:
#         """
#         Get a sample from the dataset.
        
#         Returns:
#             input_sequence: Tensor of shape (sequence_length, lat, lon)
#         """
#         start_idx = idx * self.stride
#         end_idx = start_idx + self.sequence_length
        
#         # Get input sequence
#         input_seq = self.sst_data[start_idx:end_idx]
        
#         # Get target (next time step after sequence)
#         if end_idx < len(self.sst_data):
#             target = self.sst_data[end_idx]
#         else:
#             # If no next step available, use last step as target
#             target = self.sst_data[end_idx - 1]
            
#         # Convert to tensors
#         input_tensor = th.FloatTensor(input_seq)
        
#         return input_tensor

if __name__ == "__main__":
    start_time = time()
    step_size = 1
    epochs = 10
    batch_size = 32
    learning_rate = 0.00058
    ntime_win = 3
    model = DINCAE_pconvs(ntime_win + 4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    
    
    dataset_path = Path('./data/minimal_datasets/dataset.nc')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    result_dir = Path('./data/results')
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path('./data/weights')
    if not weights_dir.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
    
    result_list = result_dir.glob('result_*.txt')
    # Find the next available results file name
    max_idx = 0
    if result_list:
        max_idx = max([int(f.stem.split('_')[1]) for f in result_list])
    i = max_idx + 1
    
    results_path = result_dir / f'result_{i}.txt' 
    weights_path = Path(f'./data/weights/weights_{i}.pt')

    ds = xr.load_dataset(dataset_path)
    print(f"Dataset loaded from {dataset_path} with shape {ds['sst'].shape}.\n")
    
    
    sst_varname = 'sst'
    ds[sst_varname] = ds[sst_varname].where(ds[sst_varname] > 0, np.nan)
    ds[sst_varname] = ds[sst_varname].where(ds[sst_varname] < 40, np.nan)


    data.z_score(ds, varname=sst_varname)
    data.minmax_scale(ds, varname='lat')
    data.minmax_scale(ds, varname='lon')
    print(f"Dataset loaded and normalized in {time() - start_time:.2f} seconds.\n")

    print(f"The number of nans is: {np.sum(np.isnan(ds[sst_varname].values))}")

    data.add_mask(ds)
    data.add_cv_points(ds)
    data.add_encoded_time(ds)
    new_path = dataset_path.parent / "dataset_w_clouds.nc"
    ds.to_netcdf(new_path)
    print(f"Dataset saved to {new_path}")
    print(f"The number of nans is: {np.sum(np.isnan(ds[sst_varname].values))}")

    train_set = data.XarrayDataset(ds, time_w=3)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    loss_fn = PerPixelMSE()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Training loop here

        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            mask = 
            loss = loss_fn(output, mask)
            loss.backward()
            optimizer.step()
