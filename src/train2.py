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
    
    params_file_path = Path(args.params)
    paths_file_path = Path(args.paths)
    if not params_file_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_file_path}")
    if not paths_file_path.exists():
        raise FileNotFoundError(f"Paths file not found: {paths_file_path}")
    with open(params_file_path, 'r') as f:
        params = json.load(f)
    with open(paths_file_path, 'r') as f:
        paths = json.load(f)
    
    train_perc = float(params["training"]["train_perc"])
    batch_size = int(params["training"]["batch_size"])
    
    weights_path, results_path = configure_file_paths(paths)
    
    # Ensure the results file exists and is a txt file
    results_path = results_path.with_suffix('.txt')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.touch(exist_ok=True)
    
    dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
    dataset_specs_path = Path(paths["data"]["current_dataset_specs_path"])
    dataset_idx = int(params["training"]["dataset_idx"])
    if dataset_idx >= 0:
        dataset_path, dataset_specs_path = change_dataset_idx(dataset_path, dataset_specs_path, dataset_idx)
    
    for path in [dataset_path, dataset_specs_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file {path} not found")
    
    with open(dataset_specs_path, 'r') as f:
        dataset_specs = json.load(f)
        
    dl = CreateDataloaders(train_perc, batch_size)
    dataset = dl.load_dataset(dataset_path)
    train_loader, test_loader = dl.create(dataset)
        
    train = TrainModel(params, weights_path, results_path, dataset_specs)
    
    train.train(train_loader, test_loader)
    
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

def change_dataset_idx(dataset_path: Path, dataset_specs_path: Path, new_idx: int) -> tuple:
    """Change the dataset index in the file names.

    Args:
        dataset_path (Path): latest dataset path
        dataset_specs_path (Path): latest dataset specs path
        new_idx (int): new dataset index

    Raises:
        FileNotFoundError: dataset file not found
        FileNotFoundError: dataset specs file not found

    Returns:
        tuple: new dataset path, new dataset specs path
    """
    dataset_ext = dataset_path.suffix
    dataset_name = dataset_path.stem.split("_")[0]
    new_dataset_path = dataset_path.parent / f"{dataset_name}_{new_idx}{dataset_ext}"
    
    dataset_specs_ext = dataset_specs_path.suffix
    dataset_specs_name = "dataset_specs"
    
    new_dataset_specs_path = dataset_specs_path.parent / f"{dataset_specs_name}_{new_idx}{dataset_specs_ext}"
    
    return new_dataset_path, new_dataset_specs_path
    
class TrainModel:
    def __init__(self, training_params, weights_path, results_path, dataset_specs = None):
        """Initialize the training class.

        Args:
            params (_type_): json
            weights_path (_type_): _description_
            results_path (_type_): _description_
            dataset_specs (_type_, optional): _description_. Defaults to None.
        """
        self.params = training_params
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        self.weights_path = weights_path
        self.results_path = results_path
        self.dataset_specs = dataset_specs
        
        self._configure_training_parameters()
        self.nan_placeholder = float(dataset_specs["dataset"]["nan_placeholder"])
        
        
        self._initialize_training_components()
        
        self.train_losses = []
        self.test_losses = []
        self.training_lr = []

    def _configure_training_parameters(self):
        training_params = self.params["training"]
        
        self.model_kind = training_params["model_kind"]
        
        self.loss_kind = str(training_params["loss_kind"])
        self.lr = training_params['learning_rate']
        self.lr_scheduler = training_params["lr_scheduler"]
        self.epochs = training_params['epochs']
        
        self.save_every = training_params['save_every']
        if self.save_every <= 0:
            raise ValueError("Save interval must be positive.")

    def _initialize_training_components(self, lr = None, lr_scheduler = None, loss_kind = None, nan_placeholder = None, model_kind = None):
        
        lr = lr if lr is not None else self.lr
        lr_scheduler = lr_scheduler if lr_scheduler is not None else self.lr_scheduler
        loss_kind = loss_kind if loss_kind is not None else self.loss_kind
        nan_placeholder = nan_placeholder if nan_placeholder is not None else self.nan_placeholder
        model_kind = model_kind if model_kind is not None else self.model_kind
        
        self.model, self.dataset_kind = initialize_model_and_dataset_kind(self.params, model_kind, self.dataset_specs)
        
        self.loss_function = get_loss_function(loss_kind, nan_placeholder)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        
        self.scheduler = None
        if lr_scheduler == "step":
            step_size = int(self.params["lr_schedulers"][lr_scheduler]["step_size"])
            gamma = float(self.params["lr_schedulers"][lr_scheduler]["gamma"])
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif lr_scheduler != "none":
            raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")
        
    def train(self, train_loader: th.utils.data.DataLoader, test_loader: th.utils.data.DataLoader, epochs: int = None):
        """Train the model on the dataset.

        Args:
            train_loader (th.utils.data.DataLoader): training dataloader
            test_loader (th.utils.data.DataLoader): testing dataloader
        """
        
        epochs = epochs if epochs is not None else self.epochs
        
        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive.")
        
        len_train_inv = 1 / len(train_loader)
        len_test_inv = len(test_loader)
        print(flush=True)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}\n", flush=True)
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for (images, masks) in train_loader:
                loss = self._compute_loss(images, masks)
                epoch_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                n_batches += 1

            self.training_lr.append(self.optimizer.param_groups[0]['lr'])
            self.scheduler.step() if self.scheduler is not None else None
            
            self.train_losses.append(epoch_loss / n_batches)
            
            with th.no_grad():
                self.model.eval()
                epoch_loss = 0.0
                n_batches = 0
                for (images, masks) in test_loader:
                    loss = self._compute_loss(images, masks)
                    epoch_loss += loss.item()
                    n_batches += 1
                self.test_losses.append(epoch_loss / n_batches)
            
            if (epoch + 1) % self.save_every == 0:
                self.save_weights()
                self.save_results()
                print(f"\nModel weights and results saved at epoch {epoch + 1}\n", flush=True)
                
        print(flush=True)
                
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