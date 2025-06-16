###########################################################################
#
#   Fix the number of epochs, initial lr with lambda scheduler and
#   batch size
#   Optimize step size
#   Use train5, that does not require the params dictionary
# 
###########################################################################


import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from pathlib import Path
from time import time
from torch import optim
import json
import signal
import os
import sys

from train5 import TrainModel
from models import initialize_model_and_dataset_kind
from losses import get_loss_function
from utils import change_dataset_idx, parse_params
from CustomDataset_v2 import CreateDataloaders

from CustomDataset_v2 import CreateDataloaders

terminate_early = False
study = None
obj = None
terminate_early = False

def signal_handler(signum, frame):
    global terminate_early, study, obj
    print("Signal received, stopping optimization...", flush=True)
    terminate_early = True
    if study is not None:
        try:
            obj.save_optim_specs(study.best_trial, study)
            print("Best hyperparameters saved before exit.", flush=True)
        except Exception as e:
            print(f"Failed to save best trial: {e}", flush=True)
    sys.exit(0)


def main():
    start_time = time()
    params, paths = parse_params()
    
    global study, obj
        
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
        
    training_params = params["training"]
    train_perc = float(training_params["train_perc"])
    batch_size = int(training_params["batch_size"])
    epochs = int(training_params["epochs"])
    model_kind = str(training_params["model_kind"])
    learning_rate = float(training_params["learning_rate"])
    loss_kind = str(training_params["loss_kind"])
    nan_placeholder = float(training_params["placeholder"])
    step_size_range = list(params["optimization"]["step_size_range"])
    n_trials = int(params["optimization"]["n_trials"])
    
    print(f"Starting optimization with parameters:\n"
          f"Dataset path: {dataset_path}\n"
          f"Dataset specs path: {dataset_specs_path}\n"
          f"Dataset index: {dataset_idx}\n"
          f"Train percentage: {train_perc}\n"
          f"Batch size: {batch_size}\n"
          f"Epochs: {epochs}\n"
          f"Model kind: {model_kind}\n"
          f"Learning rate: {learning_rate}\n"
          f"Loss kind: {loss_kind}\n"
          f"Nan placeholder: {nan_placeholder}\n"
          f"Step size range: {step_size_range}\n"
          f"Number of trials: {n_trials}\n", flush=True)
        
    dl = CreateDataloaders(train_perc, batch_size)
    dataset = dl.load_dataset(dataset_path)
    train_loader, test_loader = dl.create(dataset)
    
    model, _ = initialize_model_and_dataset_kind(params, model_kind)
    loss_function = get_loss_function(loss_kind, nan_placeholder)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    
        
    obj = Objective(
        model = model,
        epochs = epochs,
        loss_function = loss_function,
        optimizer = optimizer,
        nan_placeholder= nan_placeholder,
        train_loader=train_loader,
        test_loader=test_loader,
        step_size_range=step_size_range)
    # Import parameters and paths
    obj.import_and_check_paths(paths)
    
    obj.train.results_path = obj.results_path
    obj.train.weights_path = obj.weights_path
    obj.train.params = params
    obj.params = params
    obj.train.dataset_specs = dataset_specs
    
    storage = obj.create_storage()
    
    # Create a study to minimize the objective function
    study = optuna.create_study(direction="minimize",
                                storage=storage,
                                study_name=Path(obj.storage_path).stem,
                                load_if_exists=False)    

    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) # For SLURM jobs
    
    try:
        # Optimize in small chunks so we can handle early termination
        completed_trials = 0
        while not terminate_early and completed_trials < n_trials:
            study.optimize(obj.objective, n_trials=1, gc_after_trial=True, catch=(Exception,))
            completed_trials += 1

            # Checkpoint: Save intermediate best trial
            print("Checkpoint: Saving intermediate best trial...", flush=True)
            obj.save_optim_specs(study.best_trial, study)

    except Exception as e:
        print(f"An error occurred during optimization: {e}", flush=True)
    finally:
        if study.best_trial is not None:
            obj.save_optim_specs(study.best_trial, study)
        print(f"\nElapsed time: {time() - start_time:.2f} seconds\n", flush=True)

class Objective():
    def __init__(self, model, epochs, loss_function, optimizer, nan_placeholder, train_loader, test_loader, step_size_range, lr_scheduler=None):
        # Load default config
        
        self.dataloader_cache = {}
        self.dataset = None
        
        self.epochs = epochs
        self.step_size_range = step_size_range
        
        self.optim_next_path = None
        self.dataset_path = None
        self.dataset_specs = None
        self.train_loader = None
        self.test_loader = None
        
        self.train = TrainModel(
            model = model,
            loss_function = loss_function, 
            lr_scheduler = lr_scheduler,
            nan_placeholder = nan_placeholder,
            optimizer = optimizer)
        
        self.train_loader = train_loader
        self.test_loader = test_loader

    def import_and_check_paths(self, paths: Path):
        self.optim_next_path = Path(paths["results"]["optim_next_path"])
        self.storage_path = Path(paths["results"]["study_next_path"])
        self.dataset = None
        
        if not self.optim_next_path.parent.exists():
            raise FileNotFoundError(f"Optimization results dir {self.optim_next_path.parent} does not exist.")
        if not self.storage_path.parent.exists():
            raise FileNotFoundError(f"Storage path {self.storage_path} does not exist.")

        self.weights_path = Path(paths["results"]["weights_path"])
        self.results_path = Path(paths["results"]["results_path"])
        
        for path in [self.weights_path, self.results_path]:
            if not path.parent.exists():
                raise FileNotFoundError(f"Directory {path.parent} does not exist")
    
    def create_storage(self, storage_path: Path = None):
        """Create a storage for Optuna."""
        if storage_path is None:
            storage_path = self.storage_path
        if not storage_path.parent.exists():
            raise FileNotFoundError(f"Storage path dir {storage_path.parent} does not exist.")
        return JournalStorage(JournalFileBackend(str(storage_path)))
        
    def objective(self, trial: optuna.Trial):
        # Suggest hyperparameters
        step_size = trial.suggest_int("step_size", self.step_size_range[0], self.step_size_range[1])
        
        for layer in self.train.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # Reset optimizer state
        for param_group in self.train.optimizer.param_groups:
            for param in param_group['params']:
                param.grad = None
                
        lr_lambda = lambda step: 2 ** -(step // step_size)
        self.train.scheduler = optim.lr_scheduler.LambdaLR(self.train.optimizer, lr_lambda=lr_lambda)

        self.train.train(self.train_loader, self.test_loader, self.epochs)
        
        trial.set_user_attr("train_losses", self.train.train_losses)
        trial.set_user_attr("test_losses", self.train.test_losses)
        
        n_vals = min(5, len(self.train.train_losses))

        return sum(self.train.test_losses[-n_vals:]) / n_vals  # Optuna minimizes this (average of last n test losses)
    
    def save_optim_specs(self, trial, study: optuna.Study, optim_path: Path = None):
        # Save the best hyperparameters
        
        if optim_path is None:
            optim_path = self.optim_next_path
        
        if optim_path is None or not optim_path.parent.exists():
            raise FileNotFoundError(f"Optimization results path {optim_path} is not available.")
        
        json_str = json.dumps(self.params, indent=4)[1: -1]
        
        all_trials = []
        for t in study.trials:
            trial_info = {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
                "start_time": str(t.datetime_start) if t.datetime_start else None,
                "end_time": str(t.datetime_complete) if t.datetime_complete else None,
                "user_attrs": t.user_attrs
            }
            all_trials.append(trial_info)
        
        with open(self.optim_next_path, "w") as f:
            f.write("Best trial parameters:\n")
            f.write(f"{json.dumps(trial.params, indent=4)}\n")
            f.write("\n")
            f.write("Best trial value:\n")
            f.write(f"{trial.value}\n")
            f.write("\n")
            f.write("Best trial train losses:\n")
            for i, loss in enumerate(trial.user_attrs["train_losses"]):
                f.write(f"{loss}\t")
            f.write("\n\n")
            f.write("Best trial test losses:\n")
            for i, loss in enumerate(trial.user_attrs["test_losses"]):
                f.write(f"{loss}\t")
            f.write("\n\n")
            f.write("All trials: \n")
            f.write(json.dumps(all_trials, indent=4))
            f.write("\n")
            f.write("Training parameters:\n")
            f.write(f"{json_str}\n")
            f.write("\nDataset specifications from original file:\n\n")
            f.write(json.dumps(self.dataset_specs, indent=4))
                
            # Flush and sync to disk
            f.flush()
            os.fsync(f.fileno())
        
        print(f"Best hyperparameters saved to {optim_path}", flush = True)

if __name__ == "__main__":
    main()