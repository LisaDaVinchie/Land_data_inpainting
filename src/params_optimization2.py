###########################################################################
#
#   Fix the number of epochs, initial lr with lambda scheduler and
#   batch size
#   Optimize step size
#   Use train3
# 
###########################################################################


import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from train3 import TrainModel, change_dataset_idx
from pathlib import Path
import argparse
from time import time
import torch as th
from torch import optim
import json
import signal
import os
import sys

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
    parser = argparse.ArgumentParser(description='Train a CNN model on a dataset')
    parser.add_argument('--paths', type=Path, help='Path to the JSON file with data paths')
    parser.add_argument('--params', type=Path, help='Path to the JSON file with model parameters')
    
    args = parser.parse_args()
    
    start_time = time()

    params_path = Path(args.params)
    paths_path = Path(args.paths)
    
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters path {params_path} does not exist.")
    if not paths_path.exists():
        raise FileNotFoundError(f"Paths path {paths_path} does not exist.")
    
    global study, obj
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    with open(paths_path, 'r') as f:
        paths = json.load(f)
        
    dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
    minmax_path = Path(paths["data"]["current_minmax_path"])
    dataset_specs_path = Path(paths["data"]["current_dataset_specs_path"])
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
    
    minmax = th.load(minmax_path)
    
    with open(dataset_specs_path, 'r') as f:
        dataset_specs = json.load(f)
        
    obj = Objective(dataset_specs, minmax[0], minmax[1])
    # Import parameters and paths
    obj.import_params(params)
    obj.import_and_check_paths(paths)
    
    obj.dataloader_init(dataset_specs, dataset_path)
    
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
        while not terminate_early and completed_trials < obj.n_trials:
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
    def __init__(self, dataset_specs: dict, min_val, max_val):
        # Load default config
        
        self.dataloader_cache = {}
        self.dataset = None
        
        self.optim_next_path = None
        self.dataset_path = None
        self.dataset_specs = dataset_specs
        self.train_loader = None
        self.test_loader = None
        self.min_val = min_val
        self.max_val = max_val

    def import_params(self, params: dict):
        self.params = params
        optim_params = params["optimization"]
        self.n_trials = optim_params["n_trials"]
        self.step_size_range = optim_params["step_size_range"]
        
        fixed_params = params["training"]
        self.train_perc = fixed_params["train_perc"]
        self.epochs = fixed_params["epochs"]
        self.learning_rate = fixed_params["learning_rate"]
        self.batch_size = fixed_params["batch_size"]

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
    
    def dataloader_init(self, dataset_specs: dict, dataset_path: Path = None):
        dl = CreateDataloaders(self.train_perc)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        
        self.dataset = dl.load_dataset(dataset_path)
            
        self.dataset_specs = dataset_specs
        
        self.train_loader, self.test_loader = dl.create(self.dataset, self.batch_size)
    
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
        
        train = TrainModel(training_params=self.params,
                           weights_path=self.weights_path,
                           results_path=self.results_path,
                           dataset_specs=self.dataset_specs,
                           min_val=self.min_val,
                           max_val=self.max_val)
        train._initialize_training_components()
        
        lr_lambda = lambda step: 2 ** -(step // step_size)
        train.scheduler = optim.lr_scheduler.LambdaLR(train.optimizer, lr_lambda=lr_lambda)
        

        train.train(self.train_loader, self.test_loader, self.epochs)
        
        trial.set_user_attr("train_losses", train.train_losses)
        trial.set_user_attr("test_losses", train.test_losses)

        return train.test_losses[-1]  # Optuna minimizes this
    
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