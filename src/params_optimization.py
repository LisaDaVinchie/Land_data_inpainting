import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from train import TrainModel
from pathlib import Path
import argparse
from time import time
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

    params_path = args.params
    paths_path = args.paths
    
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters path {params_path} does not exist.")
    if not paths_path.exists():
        raise FileNotFoundError(f"Paths path {paths_path} does not exist.")
    
    global study, obj
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    with open(paths_path, 'r') as f:
        paths = json.load(f)
        
        
    obj = Objective(params, paths)
    
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
    def __init__(self, params, paths):
        # Load default config
        
        self.params = params
        self._import_paths(paths)
        self._import_params()
        
        self.dataloader_cache = {}

    def _import_params(self):
        self.n_trials = self.params["optimization"]["n_trials"]
        self.batch_size_values = self.params["optimization"]["batch_size_values"]
        self.learning_rate_range = self.params["optimization"]["learning_rate_range"]
        self.epochs_range = self.params["optimization"]["epochs_range"]
        self.train_perc = self.params["training"]["train_perc"]
        print("Parameters imported\n", flush = True)

    def _import_paths(self, paths: Path):
        self.dataset_path = Path(paths["data"]["current_minimal_dataset_path"])
        self.current_dataset_specs_path = Path(paths["data"]["current_dataset_specs_path"])
        self.optim_next_path = Path(paths["results"]["optim_next_path"])
        self.storage_path = str(paths["results"]["study_next_path"])
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")
        if not self.current_dataset_specs_path.exists():
            raise FileNotFoundError(f"Dataset specs path {self.current_dataset_specs_path} does not exist.")
        if not self.optim_next_path.parent.exists():
            raise FileNotFoundError(f"Optimization results dir {self.optim_next_path.parent} does not exist.")
        if not Path(self.storage_path).parent.exists():
            raise FileNotFoundError(f"Storage path {self.storage_path} does not exist.")
    
    def create_storage(self):
        return JournalStorage(JournalFileBackend(self.storage_path))
    
    def _get_dataloaders(self, batch_size: int):
        """Get dataloaders with caching."""
        if batch_size not in self.dataloader_cache:
            dl = CreateDataloaders(self.dataset_path, self.train_perc, batch_size)
            train_loader, test_loader = dl.create()
            self.dataloader_cache[batch_size] = (train_loader, test_loader)
        return self.dataloader_cache[batch_size]
        
    def objective(self, trial):
        # Suggest hyperparameters
        batch_size = trial.suggest_categorical("batch_size", self.batch_size_values)
        learning_rate = trial.suggest_float("learning_rate", self.learning_rate_range[0], self.learning_rate_range[1], log=True)
        epochs = trial.suggest_int("epochs", self.epochs_range[0], self.epochs_range[1])
        
        train_loader, test_loader = self._get_dataloaders(batch_size)
        
        train = TrainModel(self.params, Path("weights.pt"), Path("results.txt"), self.current_dataset_specs_path)
        train.lr = learning_rate
        train._initialize_training_components(lr = learning_rate)
        
        train.train(train_loader, test_loader, epochs)
        
        trial.set_user_attr("train_losses", train.train_losses)
        trial.set_user_attr("test_losses", train.test_losses)

        return train.test_losses[-1]  # Optuna minimizes this
    
    def save_optim_specs(self, trial, study: optuna.Study):
        # Save the best hyperparameters
        
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
            with open(self.current_dataset_specs_path, "r") as dataset_file:
                f.write(dataset_file.read())
                
            # Flush and sync to disk
            f.flush()
            os.fsync(f.fileno())
        
        print(f"Best hyperparameters saved to {self.optim_next_path}", flush = True)

if __name__ == "__main__":
    main()