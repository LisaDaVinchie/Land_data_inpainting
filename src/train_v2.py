import torch as th
import torch.optim as optim
from pathlib import Path
import argparse
from time import time
import json

from models import initialize_model_and_dataset_kind
from losses import get_loss_function
from CustomDataset_v2 import CreateDataloaders

def main():
    """Main function to train a model on a dataset."""
    start_time = time()
    
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')
    
    args = parser.parse_args()
    
    params_file_path = args.params
    paths_file_path = args.paths
    if not params_file_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file_path}")
    if not paths_file_path.exists():
        raise FileNotFoundError(f"Paths file not found: {paths_file_path}")
    with open(params_file_path, 'r') as f:
        params = json.load(f)
    with open(paths_file_path, 'r') as f:
        paths = json.load(f)
    
    train_perc = float(params["dataset"]["train_perc"])
    batch_size = int(params["dataset"]["batch_size"])
    
    weights_path, results_path, dataset_specs_path, dataset_path = configure_file_paths(paths)
        
    dl = CreateDataloaders(dataset_path, train_perc, batch_size)
    train_loader, test_loader = dl.create()
        
    train = TrainModel(params, weights_path, results_path, dataset_specs_path)
    
    train.train(train_loader, test_loader)
    
    elapsed_time = time() - start_time
    train.save_weights()
    train.save_results(elapsed_time)
    
    print(f"Training completed in {elapsed_time:.2f} seconds", flush=True)
    
def configure_file_paths(paths):
    weights_path = Path(paths["results"]["weights_path"])
    dataset_specs_path = Path(paths["data"]["current_dataset_specs_path"])
    results_path = Path(paths["results"]["results_path"])
    dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
    
    if not [dataset_specs_path.exists(), dataset_path.exists()]:
        raise FileNotFoundError(f"File not found at path {dataset_specs_path}")
    
    for path in [weights_path, results_path]:
        if not path.parent.exists():
            raise FileNotFoundError(f"Directory {path.parent} does not exist")
    
    return weights_path, results_path, dataset_specs_path, dataset_path
    
class TrainModel:
    def __init__(self, params, weights_path, results_path, dataset_specs_path):
        self.params = params
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        self._configure_training_parameters(params)
        
        self.train_losses = []
        self.test_losses = []
        self.lr = []
        
        self.weights_path = weights_path
        self.results_path = results_path
        self.dataset_specs_path = dataset_specs_path

    def _configure_training_parameters(self, params):
        training_params = params["training"]
        
        model_kind = training_params["model_kind"]
        self.model, self.dataset_kind = initialize_model_and_dataset_kind(self.params, model_kind)
        
        loss_kind = str(training_params["loss_kind"])
        nan_placeholder = float(params["dataset"]["nan_placeholder"])
        self.loss_function = get_loss_function(loss_kind, nan_placeholder)
        
        lr = training_params['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        lr_scheduler = training_params["lr_scheduler"]
        self.scheduler = None
        if lr_scheduler == "step":
            step_size = int(params["lr_schedulers"][lr_scheduler]["step_size"])
            gamma = float(params["lr_schedulers"][lr_scheduler]["gamma"])
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif lr_scheduler != "none":
            raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")
        
        self.epochs = training_params['epochs']
        
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive.")
        
        self.save_every = training_params['save_every']
        if self.save_every <= 0:
            raise ValueError("Save interval must be positive.")
        
    def train(self, train_loader: th.utils.data.DataLoader, test_loader: th.utils.data.DataLoader):
        """Train the model on the dataset.

        Args:
            train_loader (th.utils.data.DataLoader): training dataloader
            test_loader (th.utils.data.DataLoader): testing dataloader
        """
        len_train_inv = 1 / len(train_loader)
        len_test_inv = len(test_loader)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}", flush=True)
            self.model.train()
            epoch_loss = 0.0
            for (images, masks) in train_loader:
                loss = self._compute_loss(images, masks)
                epoch_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.lr.append(self.optimizer.param_groups[0]['lr'])
            self.scheduler.step() if self.scheduler is not None else None
            
            self.train_losses.append(epoch_loss * len_train_inv)
            
            with th.no_grad():
                self.model.eval()
                epoch_loss = 0.0
                for (images, masks) in test_loader:
                    loss = self._compute_loss(images, masks)
                    epoch_loss += loss.item()
                self.test_losses.append(epoch_loss * len_test_inv)
            
            if (epoch + 1) % self.save_every == 0:
                self.save_weights()
                self.save_results()
                print(f"\nModel weights and results saved at epoch {epoch + 1}\n", flush=True)
                
    def _compute_loss(self, images, masks):
        images = images.to(self.device)
        masks = masks.to(self.device)
                
        output = self.model(images, masks.float())
        loss = self.loss_function(output, images, masks)
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
            for lr in self.lr:
                f.write(f"{lr}\t")
            f.write("\n\n")
            f.write("Parameters\n")
            f.write(json_str)
            f.write("\n\n")
            f.write("\nDataset specifications from original file:\n\n")
            with open(self.dataset_specs_path, "r") as dataset_file:
                f.write(dataset_file.read())

if __name__ == "__main__":
    main()