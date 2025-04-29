import optuna
from train import TrainModel
import torch as th
import torch.optim as optim
from pathlib import Path
import argparse
from time import time
import json

from CustomDataset import create_dataloaders
from models import initialize_model_and_dataset_kind
from preprocessing.mask_data import mask_inversemask_image
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
    study.optimize(obj.objective, n_trials=5)  # Run 50 trials

    # Save the best hyperparameters
    obj.save_optim_specs(study.best_trial)
    
    print("Elapsed time:", time() - start_time, "seconds", flush = True)

class Objective():
    def __init__(self, params_path: Path, paths_path: Path):
        # Load default config
        with open(params_path, "r") as f:
            self.train_params = json.load(f)
        with open(paths_path, "r") as f:
            paths = json.load(f)
            
        self.train_perc = self.train_params["training"]["train_perc"]
        loss_kind = self.train_params["training"]["loss_kind"]
        model_kind = self.train_params["training"]["model_kind"]
        self.placeholder = self.train_params["training"]["placeholder"]
        nan_placeholder = self.train_params["dataset"]["nan_placeholder"]
        print("Parameters imported\n", flush = True)
        
        current_minimal_dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
        self.optim_next_path = Path(paths["results"]["optim_next_path"])

        self.model, self.dataset_kind = initialize_model_and_dataset_kind(params_path, model_kind)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        dataset_path = current_minimal_dataset_path
        print("Using dataset path:", dataset_path, flush = True)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        
        self.loss_function = get_loss_function(loss_kind, nan_placeholder)
        self.dataset = th.load(dataset_path)
        
        self.batch_size_values = [32, 64, 128, 256]
        self.learning_rate_range = [1e-5, 1e-2]
        self.epochs_range = [5, 20]
        
    def objective(self, trial):
        # Suggest hyperparameters
        batch_size = trial.suggest_categorical("batch_size", self.batch_size_values)
        learning_rate = trial.suggest_loguniform("learning_rate", self.learning_rate_range[0], self.learning_rate_range[1])
        epochs = trial.suggest_int("epochs", self.epochs_range[0], self.epochs_range[1])
        
        self.params = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs
        }
        
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
        
        json_str = json.dumps(self.train_params, indent=4)[1: -1]
        
        best_params = {
            "batch_size": trial.params["batch_size"],
            "learning_rate": trial.params["learning_rate"],
            "epochs": trial.params["epochs"]
        }
        
        with open(self.optim_next_path, "w") as f:
            f.write("Search range:\n")
            f.write("\n")
            f.write("Batch size Values:\n")
            for val in self.batch_size_values:
                f.write(f"{val}")
            f.write("\n\n")
            f.write("Learning rate range:\n")
            f.write(f"{self.learning_rate_range[0]}\t{self.learning_rate_range[1]}\n")
            f.write("\n")
            f.write("Epochs range:\n")
            f.write(f"{self.epochs_range[0]}\t{self.epochs_range[1]}\n")
            f.write("\n")
            f.write("Best hyperparameters:\n")
            f.write(f"{json.dumps(best_params, indent=4)}\n")
            f.write("\n")
            f.write("Best trial:\n")
            f.write(f"{json.dumps(trial.params, indent=4)}\n")
            f.write("\n")
            f.write("Best trial value:\n")
            f.write(f"{trial.value}\n")
            f.write("\n")
            f.write("Training parameters:\n")
            f.write(f"{json_str}\n")
        
        print("Best hyperparameters saved to best_hyperparameters.json")

if __name__ == "__main__":
    main()

