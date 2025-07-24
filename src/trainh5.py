import torch as th
import h5py
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from pathlib import Path
from time import time

from models import DINCAE_pconvs
from losses import PerPixelMSE

def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    __builtins__.print(*args, **kwargs)

def main():
    start_time = time()
    step_size = 1
    epochs = 10
    batch_size = 32
    learning_rate = 0.0001
    model = DINCAE_pconvs(13)
    loss_fn = PerPixelMSE()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    
    
    dataset_path = Path('data/minimal_datasets/dataset_1.h5')
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    result_dir = Path('data/results')
    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path('data/weights')
    if not weights_dir.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
    
    result_list = result_dir.glob('result_*.txt')
    # Find the next available results file name
    max_idx = 0
    if result_list:
        max_idx = max([int(f.stem.split('_')[1]) for f in result_list])
    i = max_idx + 1
    
    results_path = result_dir / f'result_{i}.txt' 
    weights_path = Path(f'data/weights/weights_{i}.pt')

    train_loader = create_dataloader(dataset_path, batch_size=batch_size, split='train', shuffle=True)
    test_loader = create_dataloader(dataset_path, batch_size=batch_size, split='test', shuffle=False)
    print("DataLoader created for single-GPU or CPU training.")
    
    
    lr_lambda = lambda step: 2 ** -(step // step_size)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    train = TrainModel(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        model_save_path=weights_path,
        results_path=results_path
    )
    
    print("Starting training...")
    train.train(train_loader, test_loader, epochs=epochs)
    train.save_results()
    print(f"Results saved to {results_path}")
    th.save(model.state_dict(), 'model.pth')
    print(f"Training completed and model saved in {time() - start_time:.2f} seconds.\n")


def create_dataloader(dataset_path, batch_size, split='train', shuffle=True):
    dataset = MultiH5Dataset(dataset_path, split=split)
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # For single-GPU or CPU training:
    loader = DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle= shuffle,
        num_workers=4,
        pin_memory=True
    )
    
    return loader

def get_metadata(dataset_path, split='train'):
    dataset = MultiH5Dataset(dataset_path, split=split)
    
    return dataset.get_metadata()

    # # For DistributedDataParallel (multi-GPU):
    # sampler = DistributedSampler(dataset)
    # loader = DataLoader(
    #     dataset,
    #     batch_size=32,
    #     sampler=sampler,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
class TrainModel:
    def __init__(self, model, loss_fn, optimizer, model_save_path, results_path):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
            
        self.clip_value = 5.0  # Gradient clipping value
        self.n_days = 9  # Number of days in the dataset
        self.current_day_channel = self.n_days // 2
        self.save_every = 1  # Save model every 10 epochs
        
        self.train_losses = []
        self.test_losses = []
        self.lr = []
        
        self.model_save_path = Path(model_save_path)
        self.results_path = Path(results_path)

    def train(self, train_loader, test_loader, epochs, scheduler=None):

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            train_loss = self.step(train_loader, backprop=True)
            self.train_losses.append(train_loss)
            
            if scheduler:
                scheduler.step()
                print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            with th.no_grad():
                test_loss = self.step(test_loader, backprop=False)
            self.test_losses.append(test_loss)
            
            self.lr.append(self.optimizer.param_groups[0]['lr'])
            
            if (epoch + 1) % self.save_every == 0:
                th.save(self.model.state_dict(), self.model_save_path)
                self.save_results()

            print()
    
    def step(self, loader, backprop=True):
        self.model.train() if backprop else self.model.eval()
        
        total_loss = 0.0
        for (img, mask, nanmask) in loader:
            img, mask, nanmask = img.to(self.device), mask.to(self.device), nanmask.to(self.device)

            output = self.model(img * mask.float(), (mask & nanmask).float())
            
            loss = self.loss_fn(output[:, 0:1],
                                img[:, self.current_day_channel:self.current_day_channel + 1],
                                self.validation_mask(
                                    mask[:, self.current_day_channel: self.current_day_channel + 1],
                                    nanmask[:, self.current_day_channel: self.current_day_channel + 1]
                                    )
                                )
            total_loss += loss.item()
            
            if backprop:
                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
                self.optimizer.step()
                
        return total_loss / len(loader.dataset)
    
    def validation_mask(self, masks: th.Tensor, nan_masks: th.Tensor, loss: bool = True):
        """Calculate the mask used to calculate the loss, i.e. where the pixel is masked but not nan.

        Args:
            masks (th.Tensor): masks tensor
            nan_masks (th.Tensor): nan masks tensor

        Returns:
            th.Tensor: validation mask
        """
        return ~(~masks & nan_masks) if loss else (~masks & nan_masks)
    
    
    def save_results(self, elapsed_time: float = None):
        """Save the training results to a file.

        Args:
            elapsed_time (float): elapsed time of the training
        """
        
        # Save the train losses to a txt file
        with open(self.results_path, 'w') as f:
            if elapsed_time is not None:
                f.write("Elapsed time [s]:\n")
                f.write(f"{elapsed_time}\n\n")
            f.write("Train losses\n")
            for loss in self.train_losses:
                f.write(f"{loss}\t")
            f.write("\n\n")
            f.write("Test losses\n")
            for loss in self.test_losses:
                f.write(f"{loss}\t")
            f.write("\n\n")
            f.write("Learning rate\n")
            for lr in self.lr:
                f.write(f"{lr}\t")

    
class MultiH5Dataset(th.utils.data.Dataset):
    def __init__(self, h5_path, split):
        self.h5 = None
        with h5py.File(h5_path, 'r') as f:
            self.indices = list(f[split].keys())
        
        self.indices = [k for k in self.indices if k != "metadata"]
        self.path = h5_path
        self.split = split

    def __getitem__(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, 'r')
        grp = self.h5[self.split][self.indices[idx]]
        img = th.from_numpy(grp['image'][...]).float()
        mask = th.from_numpy(grp['mask'][...]).bool()
        nan = th.from_numpy(grp['nanmask'][...]).bool()
        return img, mask, nan
    
    def get_metadata(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, 'r')
        return self.h5[self.split]['metadata'][...]

    def __len__(self):
        return len(self.indices)
    
if __name__ == '__main__':
    main()
