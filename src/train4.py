###############################################################################
##  
##  As the original training, but now the input is normalized to the range
##  [-1, 1] while the output is assumed to be normalized in the same range.
##  
###############################################################################


import torch as th
import torch.optim as optim
import torch.nn.utils as utils
from pathlib import Path
from time import time
import json

from models import initialize_model_and_dataset_kind
from losses import get_loss_function, calculate_valid_pixels
from select_lr_scheduler import select_lr_scheduler
from utils import change_dataset_idx, parse_params, find_min_max_values
from CustomDataset_v2 import CreateDataloaders

def main():
    """Main function to train a model on a dataset."""
    start_time = time()
    
    params, paths = parse_params()
    
    training_params = params["training"]
    train_perc = float(training_params["train_perc"])
    batch_size = int(training_params["batch_size"])
    epochs = int(training_params["epochs"])
    model_kind = str(training_params["model_kind"])
    learning_rate = float(training_params["learning_rate"])
    loss_kind = str(training_params["loss_kind"])
    scheduler_kind = str(training_params["lr_scheduler"])
    nan_placeholder = float(training_params["placeholder"])
    
    weights_path, results_path = configure_file_paths(paths)
    
    # Ensure the results file exists and is a txt file
    results_path = results_path.with_suffix('.txt')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.touch(exist_ok=True)
    
    dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
    dataset_specs_path = Path(paths["data"]["current_dataset_specs_path"])
    minmax_path = Path(paths["data"]["current_minmax_path"])
    dataset_idx = int(params["training"]["dataset_idx"])
    if dataset_idx >= 0:
        dataset_path, dataset_specs_path = change_dataset_idx(dataset_path, dataset_specs_path, dataset_idx)
    
    for path in [dataset_path, dataset_specs_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file {path} not found")
        
    if not minmax_path.parent.exists():
        raise FileNotFoundError(f"Minmax file directory {minmax_path.parent} does not exist")
    minmax_base_path = str(minmax_path)[:-4]  # Remove the last two characters (dataset index)
    minmax_path = Path(minmax_base_path + f"{dataset_idx}.pt")
    
    with open(dataset_specs_path, 'r') as f:
        dataset_specs = json.load(f)
        
    dl = CreateDataloaders(train_perc, batch_size)
    dataset = dl.load_dataset(dataset_path)
    train_loader, test_loader = dl.create(dataset)

    min_val, max_val = find_min_max_values(minmax_path, dataset)
    
    images = dataset["images"]
    masks = dataset["masks"]
    images *= masks.float()  # Apply masks to images to exclude masked pixels
    
    min_val = images[:, 0:9][images[:, 0:9] > 0].min()
    max_val = images[:, 0:9][images[:, 0:9] > 0].max()
    print(f"Minimum value in the dataset: {min_val}", flush=True)
    
    model, _ = initialize_model_and_dataset_kind(params, model_kind)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    loss_function = get_loss_function(loss_kind, nan_placeholder)
    lr_scheduler = select_lr_scheduler(params, scheduler_kind, optimizer)
        
    train = TrainModel(model = model, 
                      loss_function = loss_function, 
                      optimizer = optimizer, 
                      min_val = min_val, 
                      max_val = max_val, 
                      lr_scheduler = lr_scheduler,
                      nan_placeholder = nan_placeholder)
    
    train.results_path = results_path
    train.weights_path = weights_path
    train.params = params
    train.dataset_specs = dataset_specs
    
    train.train(train_loader, test_loader, epochs)
    
    elapsed_time = time() - start_time
    print(flush=True)
    train.save_weights()
    print(f"Model weights saved at {weights_path}", flush=True)
    
    print(flush=True)
    train.save_results(elapsed_time)
    print(f"Results saved at {results_path}", flush=True)
    
    print(f"\nTraining completed in {elapsed_time:.2f} seconds", flush=True)
    
def configure_file_paths(paths):
    """Get and check the paths for the weights and results files.

    Args:
        paths (_type_): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        _type_: _description_
    """
    weights_path = Path(paths["results"]["weights_path"])
    results_path = Path(paths["results"]["results_path"])
    
    for path in [weights_path, results_path]:
        if not path.parent.exists():
            raise FileNotFoundError(f"Directory {path.parent} does not exist")
    
    return weights_path, results_path
    
class TrainModel:
    def __init__(self, model, loss_function, optimizer, min_val, max_val, nan_placeholder, clip_value = 5.0, lr_scheduler = None, save_every = 1):
        """Initialize the training class.

        Args:
            params (_type_): json
            weights_path (_type_): _description_
            results_path (_type_): _description_
            dataset_specs (_type_, optional): _description_. Defaults to None.
        """

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.clip_value = clip_value
        self.save_every = save_every
        self.nan_placeholder = nan_placeholder
        
        self.results_path = None
        self.weights_path = None
        self.params = None
        self.dataset_specs = None
        
        self.train_losses = []
        self.test_losses = []
        self.training_lr = []
        
        # boundaries = th.load(Path("data/minmax_vals/min_max.pt"))
        self.min_val = min_val
        self.max_val = max_val
        self.diff_inv = 1 / (self.max_val - self.min_val)
        
        print(f"Using min value: {self.min_val}, max value: {self.max_val}", flush=True)
        
    def normalize(self, images):
        norm_images = 2 * (images - self.min_val) * self.diff_inv - 1
        
        # Make sure that the NaN placeholder value is preserved
        norm_images = th.where(images == self.nan_placeholder, self.nan_placeholder, norm_images)
        
        return norm_images
        
    def train(self, train_loader: th.utils.data.DataLoader, test_loader: th.utils.data.DataLoader, epochs: int = None):
        """Train the model on the dataset.

        Args:
            train_loader (th.utils.data.DataLoader): training dataloader
            test_loader (th.utils.data.DataLoader): testing dataloader
        """
        
        print(flush=True)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}\n", flush=True)
            self.model.train()
            epoch_loss = 0.0
            n_valid_pixels = 0
            for (images, masks) in train_loader:
                loss = self._compute_loss(images, masks)
                epoch_loss += loss.item()
                
                loss.backward()
                utils.clip_grad_value_(self.model.parameters(), self.clip_value)  # Clip gradients to avoid exploding gradients
                self.optimizer.step()
                self.optimizer.zero_grad()
                n_valid_pixels += calculate_valid_pixels(masks[:, 4], images[:, 4], self.nan_placeholder)

            self.training_lr.append(self.optimizer.param_groups[0]['lr'])
            self.scheduler.step() if self.scheduler is not None else None
            
            self.train_losses.append(epoch_loss / (n_valid_pixels + 1e-8))
            
            with th.no_grad():
                self.model.eval()
                epoch_loss = 0.0
                n_valid_pixels = 0
                for (images, masks) in test_loader:
                    loss = self._compute_loss(images, masks)
                    epoch_loss += loss.item()
                    n_valid_pixels += calculate_valid_pixels(masks[:, 4], images[:, 4], self.nan_placeholder)
            self.test_losses.append(epoch_loss / (n_valid_pixels + 1e-8))
            
            if (epoch + 1) % self.save_every == 0:
                self.save_weights()
                self.save_results()
                print(f"\nModel weights and results saved at epoch {epoch + 1}\n", flush=True)
                
        print(flush=True)
                
    def _compute_loss(self, images, masks):
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        norm_images = images.clone()
        
        norm_images[0:9] = self.normalize(images[0:9])
        
        
        # Multiply images by masks to exxlude information from masked pixels
        output = self.model(norm_images * masks.float(), masks.float())
        loss = self.loss_function(output[:, 0], norm_images[:, 4], masks[:, 4])
        return loss
    
    def save_weights(self):
        """Save the model weights to a file."""
        th.save(self.model.state_dict(), self.weights_path)
        
    def save_results(self, elapsed_time: float = None):
        """Save the training results to a file.

        Args:
            elapsed_time (float): elapsed time of the training
        """
        
        json_str = json.dumps(self.params, indent=4)[1: -1]
        
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
            for lr in self.training_lr:
                f.write(f"{lr}\t")
            f.write("\n\n")
            f.write("Parameters\n")
            f.write(json_str)
            f.write("\n\n")
            f.write("\nDataset specifications from original file:\n\n")
            f.write(json.dumps(self.dataset_specs, indent=4)[1: -1])

if __name__ == "__main__":
    main()