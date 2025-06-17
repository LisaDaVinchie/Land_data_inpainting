import unittest
import json
import torch as th
from pathlib import Path
import tempfile
import shutil
import optuna
import os
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from params_optimization import Objective

class TestOptunaOptimization(unittest.TestCase):
    def setUp(self):
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.params_path = Path(self.temp_dir) / "params.json"
        self.paths_path = Path(self.temp_dir) / "paths.json"
        
        self.n_trials = 3
        self.batch_size_values = [5, 10]
        self.learning_rate_range = [0.000001, 0.001]
        self.epochs_range = [2, 3]
        self.dataset_shape = (20, 3, 30, 15)
        
        # Sample test data
        self.sample_params = {
            "optimization": {
                "n_trials": self.n_trials,
                "batch_size_values": self.batch_size_values,
                "learning_rate_range": self.learning_rate_range,
                "epochs_range": self.epochs_range,
            },
            "training":{
                "train_perc": 0.8,
                "loss_kind": "custom1",
                "_possible_losses": ["per_pixel", "per_pixel_mse", "tv_loss", "custom1"],
                "model_kind": "DINCAE_pconvs",
                "_possible_model_kinds": ["simple_conv", "DINCAE_like", "DINCAE_pconvs"],
                "placeholder": 0.0,
                "save_every": 100,
                "dataset_idx": 0,

                "batch_size": 0,
                "epochs": 0,
                "learning_rate": 0,
                "lr_scheduler": "none",
                "_possible_schedulers": ["none", "step"],
                "optimizer_kind": "adam"
            },
            "dataset":{
                "cutted_nrows": 128,
                "cutted_ncols": 128,
                "nan_placeholder": -2.0,
                "dataset_kind": "biochemistry",
                "_possible_dataset_kinds": ["biochemistry", "temperature"],
                "biochemistry": {
                    "n_channels": self.dataset_shape[1]
                },
                "temperature": {
                    "n_channels": self.dataset_shape[1]
                }
            },
            "models":{
                "DINCAE_pconvs": {
                    "middle_channels": [16, 30, 58, 110, 209],
                    "kernel_sizes": [3, 3, 3, 3, 3],
                    "pooling_sizes": [2, 2, 2, 2, 2],
                    "interp_mode": "bilinear",
                    "_possible_interpolation_modes": ["nearest", "linear", "bilinear", "bicubic", "trilinear"]
                },
                "DINCAE_like": {
                    "middle_channels": [16, 30, 58, 110, 209],
                    "kernel_sizes": [3, 3, 3, 3, 3],
                    "pooling_sizes": [2, 2, 2, 2, 2],
                    "output_size": 15,
                    "interp_mode": "bilinear",
                    "_possible_interpolation_modes": ["nearest", "linear", "bilinear", "bicubic", "trilinear"]
                },
                "simple_conv":{
                    "middle_channels": [32, 64, 128],
                    "kernel_size": [3, 3, 3],
                    "stride": [1, 1, 1],
                    "padding": [1, 1, 1],
                    "output_padding": [0, 0, 0]
                }
            }
        }

        self.dataset_specs = {
            "dataset":{
                "cutted_nrows": 128,
                "cutted_ncols": 128,
                "nan_placeholder": -2.0,
                "dataset_kind": "biochemistry",
                "_possible_dataset_kinds": ["biochemistry", "temperature"],
                "biochemistry": {
                    "n_channels": self.dataset_shape[1]
                },
                "temperature": {
                    "n_channels": self.dataset_shape[1]
                }
            }
        }
        self.temp_dataset_path = str(Path(self.temp_dir) / "data.pt")
        self.temp_dataset_specs_path = str(Path(self.temp_dir) / "specs.json")
            
        # Create required directories
        os.makedirs(Path(self.temp_dir) / "data")
        os.makedirs(Path(self.temp_dir) / "results")
        with open(Path(self.temp_dir) / "specs.json", 'w') as f:
            json.dump({"test": "specs"}, f)
    
        # Create sample dataset:
        self.dataset = {
            "images": th.randn(self.dataset_shape),
            "masks": th.randint(0, 2, self.dataset_shape)
        }
        
        # Create a temporary study file
        self.temp_study_path = Path(self.temp_dir) / "temp_study.db"
        
            
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
        if self.temp_study_path.exists():
            os.remove(self.temp_study_path)

    # @patch('optuna.create_study')
    # @patch('train.TrainModel')
    # @patch('CustomDataset_v2.CreateDataloaders')
    # def test_main_optimization_flow(self, mock_dataloaders, mock_train, mock_study):
        
    #     # Setup mocks
    #     mock_study.return_value = MagicMock()
    #     mock_train_instance = MagicMock()
    #     mock_train_instance.train_losses = [0.5, 0.4]
    #     mock_train_instance.test_losses = [0.6, 0.5]
    #     mock_train.return_value = mock_train_instance
        
    #     mock_dl_instance = MagicMock()
    #     mock_dl_instance.create.return_value = (MagicMock(), MagicMock())
    #     mock_dataloaders.return_value = mock_dl_instance
        
    #     # Simulate command line arguments
    #     with patch('argparse.ArgumentParser.parse_args', 
    #               return_value=argparse.Namespace(
    #                   paths=self.paths_path,
    #                   params=self.params_path
    #               )):
    #         main()
            
    #     # Verify study was created
    #     mock_study.assert_called_once()
        
    #     # Verify optimization ran
    #     mock_study.return_value.optimize.assert_called()

    def test_objective_initialization(self):
        
        obj = Objective()
        obj.import_params(self.sample_params)
        
        self.assertEqual(obj.n_trials, self.n_trials)
        self.assertEqual(obj.batch_size_values, self.batch_size_values)
        self.assertEqual(obj.learning_rate_range, self.learning_rate_range)
        self.assertEqual(obj.train_perc, 0.8)

    # def test_objective_function(self):
        
    #     obj = Objective()
    #     obj.import_params(self.sample_params)
    #     obj.dataset = self.dataset
    #     obj.storage_path = self.temp_study_path
    #     obj.dataloader_init(dataset_specs=self.dataset_specs, dataset=self.dataset)
        
    #     storage = obj.create_storage(self.temp_study_path)
    #         # Create a study to minimize the objective function
    #     study = optuna.create_study(direction="minimize",
    #                             storage=storage,
    #                             study_name=Path(obj.storage_path).stem,
    #                             load_if_exists=False)
        
        
        
    #     study.optimize(obj.objective, n_trials=1)
        
    #     # Verify hyperparameters were suggested
    #     study.trials[0].suggest_categorical.assert_called_with("batch_size", [16, 32])
    #     study.trials[0].suggest_float.assert_called_with("learning_rate", 0.001, 0.1, log=True)
    #     study.trials[0].suggest_int.assert_called_with("epochs", 1, 3)
        
    #     # Verify attributes were set
    #     study.trials[0].set_user_attr.assert_any_call("train_losses", [0.5, 0.4])
    #     study.trials[0].set_user_attr.assert_any_call("test_losses", [0.6, 0.5])
        
    #     # Verify return value
    #     self.assertEqual(study.best_value, 0.5)
        
        

    # @patch('sys.exit')
    # def test_signal_handler(self, mock_exit):
        
    #     # Setup test objects
    #     test_study = MagicMock()
    #     test_obj = MagicMock()
    #     study = test_study
    #     obj = test_obj
        
    #     # Call handler
    #     signal_handler(None, None)
        
    #     # Verify behavior
    #     test_obj.save_optim_specs.assert_called_with(test_study.best_trial, test_study)
    #     mock_exit.assert_called_with(0)

if __name__ == '__main__':
    unittest.main()