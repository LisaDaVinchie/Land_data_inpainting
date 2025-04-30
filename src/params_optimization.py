import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from train import TrainModel
import torch as th
import torch.optim as optim
from pathlib import Path
import argparse
from time import time
import json

from CustomDataset import create_dataloaders
from models import initialize_model_and_dataset_kind
from losses import get_loss_function

def main():
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')
    parser.add_argument('--dataset_idx', type=int, help='Index of the dataset to use', required=False)
    
    args = parser.parse_args()
    
    start_time = time()

    params_path = args.params
    paths_path = args.paths
    # Create a study (you can also specify direction="maximize" if appropriate)
    study = optuna.create_study(direction="minimize")  # e.g., minimize test loss

    obj = Objective(params_path, paths_path)
    # Optimize the objective function
    study.optimize(obj.objective, n_trials=obj.n_trials)

    # Save the best hyperparameters
    obj.save_optim_specs(study.best_trial)
    
    print("Elapsed time:", time() - start_time, "seconds", flush = True)

class Objective():
    def __init__(self, params_path: Path, paths_path: Path):
        # Load default config
        with open(paths_path, "r") as f:
            paths = json.load(f)
            
        with open(params_path, "r") as f:
            self.params = json.load(f)
            
        self.train_perc = self.params["training"]["train_perc"]
        loss_kind = self.params["training"]["loss_kind"]
        model_kind = self.params["training"]["model_kind"]
        self.placeholder = self.params["training"]["placeholder"]
        nan_placeholder = self.params["dataset"]["nan_placeholder"]
        print("Parameters imported\n", flush = True)
        
        current_minimal_dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
        self.current_dataset_specs_path = Path(paths["data"]["current_dataset_specs_path"])
        self.optim_next_path = Path(paths["results"]["optim_next_path"])
        
        if not current_minimal_dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {current_minimal_dataset_path} does not exist.")
        if not self.current_dataset_specs_path.exists():
            raise FileNotFoundError(f"Dataset specs path {self.current_dataset_specs_path} does not exist.")
        if not self.optim_next_path.parent.exists():
            raise FileNotFoundError(f"Optimization results dir {self.optim_next_path.parent} does not exist.")

        self.model, self.dataset_kind = initialize_model_and_dataset_kind(params_path, model_kind)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        dataset_path = current_minimal_dataset_path
        print("Using dataset path:", dataset_path, flush = True)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        
        self.loss_function = get_loss_function(loss_kind, nan_placeholder)
        self.dataset = th.load(dataset_path)
        
        self.n_trials = self.params["optimization"]["n_trials"]
        self.batch_size_values = self.params["optimization"]["batch_size_values"]
        self.learning_rate_range = self.params["optimization"]["learning_rate_range"]
        self.epochs_range = self.params["optimization"]["epochs_range"]
        
    def objective(self, trial):
        # Suggest hyperparameters
        batch_size = trial.suggest_categorical("batch_size", self.batch_size_values)
        learning_rate = trial.suggest_float("learning_rate", self.learning_rate_range[0], self.learning_rate_range[1], log=True)
        epochs = trial.suggest_int("epochs", self.epochs_range[0], self.epochs_range[1])
        
        train_loader, test_loader = create_dataloaders(self.dataset, self.train_perc, batch_size)
        
    
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        training_class = TrainModel(self.model, epochs, self.device, train_loader, test_loader, self.loss_function, optimizer, None)
        if self.dataset_kind == "extended":
            train_losses, test_losses = training_class.train_loop_extended(self.placeholder)
        elif self.dataset_kind == "minimal":
            train_losses, test_losses = training_class.train_loop_minimal()
        else:
            raise ValueError(f"Dataset kind {self.dataset_kind} not recognized")

        return test_losses[-1]  # Optuna minimizes this
    
    def save_optim_specs(self, trial):
        # Save the best hyperparameters
        
        json_str = json.dumps(self.params, indent=4)[1: -1]
        
        with open(self.optim_next_path, "w") as f:
            f.write("Search range:\n")
            f.write("\n")
            f.write("Batch size Values:\n")
            for val in self.batch_size_values:
                f.write(f"{val}\t")
            f.write("\n\n")
            f.write("Learning rate range:\n")
            for lr in self.learning_rate_range:
                f.write(f"{lr}\t")
            f.write("\n\n")
            f.write("Epochs range:\n")
            for epoch in self.epochs_range:
                f.write(f"{epoch}\t")
            f.write("\n\n")
            f.write("Best trial:\n")
            f.write(f"{json.dumps(trial.params, indent=4)}\n")
            f.write("\n")
            f.write("Best trial value:\n")
            f.write(f"{trial.value}\n")
            f.write("\n")
            f.write("Training parameters:\n")
            f.write(f"{json_str}\n")
            f.write("\nDataset specifications from original file:\n\n")
            with open(self.current_dataset_specs_path, "r") as dataset_file:
                f.write(dataset_file.read())
        
        print(f"Best hyperparameters saved to {self.optim_next_path}", flush = True)

if __name__ == "__main__":
    main()