import unittest
import torch as th
from pathlib import Path
import json
import tempfile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from train_v2 import TrainModel
from CustomDataset_v2 import CreateDataloaders

class TestTrainingScript(unittest.TestCase):
    def setUp(self):
        # Create temporary directories and files for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.train_perc = 0.8
        self.batch_size = 32
        self.nan_placeholder = -1.0
        self.lr = 0.001
        self.epochs = 3
        self.save_every = 5
        self.n_channels = 3
        self.nrows = 128
        self.ncols = 128
        
        # Create dummy JSON files
        self.params_content = {
            "dataset": {
                "nan_placeholder": self.nan_placeholder,
                "cutted_nrows": self.nrows,
                "cutted_ncols": self.ncols,
                "dataset_kind": "test",
                "test": {
                    "n_channels": self.n_channels,
                }
            },
            "training": {
                "train_perc": self.train_perc,
                "batch_size": self.batch_size,
                "model_kind": "DINCAE_pconvs",
                "loss_kind": "per_pixel_mse",
                "learning_rate": self.lr,
                "lr_scheduler": "step",
                "epochs": self.epochs,
                "save_every": self.save_every,
            },
            "lr_schedulers": {
                "step": {
                    "step_size": 5,
                    "gamma": 0.1
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
                }
            }
        }
        
        self.paths_content = {
            "data": {
                "current_minimal_dataset_path": "dataset.pt",
                "current_dataset_specs_path": "specs.txt"
            },
            "results": {
                "weights_path": "weights.pth",
                "results_path": "results.txt"
            }
        }
        
        # Write files to temp dir
        self.params_path = Path(self.temp_dir.name) / "params.json"
        self.paths_path = Path(self.temp_dir.name) / "paths.json"
        self.dataset_path = Path(self.temp_dir.name) / "dataset.pt"
        self.specs_path = Path(self.temp_dir.name) / "specs.txt"
        
        with open(self.params_path, 'w') as f:
            json.dump(self.params_content, f)
            
        with open(self.paths_path, 'w') as f:
            json.dump(self.paths_content, f)
            
        # Create dummy dataset
        dummy_dataset = {
            'images': th.randn(100, self.n_channels, self.nrows, self.ncols),
            'masks': th.ones((100, self.n_channels, self.nrows, self.ncols), dtype=th.bool)
        }
        th.save(dummy_dataset, self.dataset_path)
        
        # Create dummy specs file
        with open(self.specs_path, 'w') as f:
            f.write("Dataset specifications\n")
        
        self.weights_path = "weights.pt"
        self.results_path = "results.txt"
        self.dataset_specs_path = "specs.txt"

    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_initialize_train_model(self):
        # Test initialization of TrainModel
        train_model = TrainModel(self.params_content, self.weights_path, self.results_path, self.dataset_specs_path)
        
        # Check if model and dataset are initialized correctly
        self.assertIsNotNone(train_model.model)
        self.assertIsNotNone(train_model.dataset_kind)
        
        # Check if loss function is initialized correctly
        self.assertIsNotNone(train_model.loss_function)
        
        # Check if optimizer is initialized correctly
        self.assertIsNotNone(train_model.optimizer)
        
        # Check if scheduler is initialized correctly
        self.assertIsNotNone(train_model.scheduler)
        
        # Check if training parameters are set correctly
        self.assertEqual(train_model.epochs, self.epochs)
        self.assertEqual(train_model.save_every, self.save_every)

    def test_train_model_invalid_epochs(self):
        # Test invalid epochs parameter
        invalid_params = self.params_content.copy()
        invalid_params["training"]["epochs"] = 0
        
        with self.assertRaises(ValueError):
            TrainModel(invalid_params, self.weights_path, self.results_path, self.dataset_specs_path)

    def test_train_model_invalid_save_every(self):
        # Test invalid save_every parameter
        invalid_params = self.params_content.copy()
        invalid_params["training"]["save_every"] = 0
        
        with self.assertRaises(ValueError):
            TrainModel(invalid_params, self.weights_path, self.results_path, self.dataset_specs_path)

    def test_train_model_invalid_lr_scheduler(self):
        # Test invalid lr_scheduler parameter
        invalid_params = self.params_content.copy()
        invalid_params["training"]["lr_scheduler"] = "invalid"
        
        with self.assertRaises(ValueError):
            TrainModel(invalid_params, self.weights_path, self.results_path, self.dataset_specs_path)

    def test_train_model_no_scheduler(self):
        # Test no scheduler case
        no_scheduler_params = self.params_content.copy()
        no_scheduler_params["training"]["lr_scheduler"] = "none"
        
        train_model = TrainModel(no_scheduler_params, self.weights_path, self.results_path, self.dataset_specs_path)
        self.assertIsNone(train_model.scheduler)

    def test_train_method(self):
        # Test the train method
        
        dl = CreateDataloaders(self.dataset_path, self.train_perc, self.batch_size)
        mock_train_loader, mock_test_loader = dl.create()
        
        # Run training
        train_model = TrainModel(self.params_content, self.weights_path, self.results_path, self.dataset_specs_path)
        train_model.train(mock_train_loader, mock_test_loader)
        
        # Check results
        self.assertEqual(len(train_model.train_losses), self.epochs)
        self.assertEqual(len(train_model.test_losses), self.epochs)
        self.assertEqual(len(train_model.lr), self.epochs)

if __name__ == '__main__':
    unittest.main()