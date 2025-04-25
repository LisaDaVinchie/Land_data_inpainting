import unittest
from unittest.mock import patch
import torch as th
from pathlib import Path
import json
from tempfile import NamedTemporaryFile

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from train import track_memory, change_dataset_idx, validate_paths, train_loop_extended, train_loop_minimal
from models import simple_conv, DINCAE_pconvs
from CustomDataset import create_dataloaders
from losses import get_loss_function

class TestTrainingFunctions(unittest.TestCase):
    def setUp(self):
        """Create a temporary JSON file with parameters."""
        self.batch_size = 64
        self.dataset_len = 100
        self.n_channels = 3
        self.ncols = 64
        self.nrows = 64
        self.epochs = 3
        self.nan_placeholder = -2.0
        
        self.middle_channels_e = [12, 24, 36]
        self.kernel_sizes_e = [3, 3, 3]
        self.stride_e = [2, 2, 2]
        self.padding_e = [1, 1, 1]
        self.output_padding_e = [1, 1, 1]
        
        
        self.extended_model_params = {
            "simple_conv": {
                "middle_channels": self.middle_channels_e,
                "kernel_size": self.kernel_sizes_e,
                "stride": self.stride_e,
                "padding": self.padding_e,
                "output_padding": self.output_padding_e
                },  
            "dataset": {
                "dataset_kind": "test",
                "test": {
                    "n_channels": self.n_channels,
                }
            }
        }
        
        
        # Create a temporary file
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.extended_model_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.extended_params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        self.extended_model = simple_conv(params_path=self.extended_params_path)
        
        self.reduced_model_params = {
            "dataset": {
                "cutted_nrows": self.nrows,
                "cutted_ncols": self.ncols,
                "dataset_kind": "test",
                "test": {
                    "n_channels": self.n_channels,
                }
            },
            "DINCAE_pconvs": {
                "middle_channels": [16, 16, 16, 16, 16],
                "kernel_sizes": [5, 3, 5, 3, 5],
                "pooling_sizes": [2, 2, 2, 2, 2],
                "interp_mode": "nearest"
            }
        }
        
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.reduced_model_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.reduced_params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        self.reduced_model = DINCAE_pconvs(params_path=self.reduced_params_path)
        
        
        images = th.randn(self.dataset_len, self.n_channels, self.nrows, self.ncols)
        masks = th.ones(self.dataset_len, self.n_channels, self.nrows, self.ncols)
        masks[th.randint(0, 2, (self.dataset_len, self.n_channels, self.nrows, self.ncols)) == 0] = 0
        test_dataset = {
            "images": images,
            "masks": masks
            }
        
        self.train_loader, self.test_loader = create_dataloaders(test_dataset, 0.8, batch_size=self.batch_size)

        # Use a real loss function and optimizer
        self.loss_function = get_loss_function("per_pixel_mse", self.nan_placeholder)
        
    def test_track_memory(self):
        """Test that track_memory logs memory usage."""
        with patch("psutil.Process") as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 100 * 1e6  # 100 MB
            with patch("builtins.print") as mock_print:
                track_memory("Test Stage")
                mock_print.assert_called_with("[Memory] Test Stage: 100.00 MB\n", flush=True)

    def test_change_dataset_idx(self):
        """Test that change_dataset_idx updates the dataset path correctly."""
        dataset_path = Path("/data/dataset_0.pt")
        
        # Test with a specified index
        new_path = change_dataset_idx(1, dataset_path)
        self.assertEqual(new_path, Path("/data/dataset_1.pt"))
        
        # Test with no index (should return the original path)
        new_path = change_dataset_idx(None, dataset_path)
        self.assertEqual(new_path, dataset_path)

    def test_validate_paths(self):
        """Test that validate_paths raises FileNotFoundError for missing paths."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            with self.assertRaises(FileNotFoundError):
                validate_paths([Path("/fake/path")])
            
            mock_exists.return_value = True
            try:
                validate_paths([Path("/real/path")])
            except FileNotFoundError:
                self.fail("validate_paths raised FileNotFoundError unexpectedly")

    def test_train_loop_extended(self):
        """Test that train_loop_extended trains the model and returns losses."""
        
        optimizer = th.optim.Adam(self.extended_model.parameters())
        
        initial_weights = {k: v.clone() for k, v in self.extended_model.state_dict().items()}

        # Call the function
        model, train_losses, test_losses = train_loop_extended(
            epochs=self.epochs,
            placeholder=0.0,
            model=self.extended_model,
            device=th.device("cpu"),
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            loss_function=self.loss_function,
            optimizer=optimizer,
        )
        
        model.eval()
        final_weights = {k: v.clone() for k, v in model.state_dict().items()}

        # Check the outputs
        for i in range(len(train_losses)):
            self.assertIsInstance(train_losses[i], float)
            self.assertIsInstance(test_losses[i], float)
        self.assertEqual(len(train_losses), self.epochs)  # 2 epochs
        self.assertEqual(len(test_losses), self.epochs)  # 2 epochs
        self.assertIsInstance(model, th.nn.Module)
        
        
        # Check that the model has been trained
        updated = any(not th.equal(initial_weights[k], final_weights[k]) for k in initial_weights)
        self.assertTrue(updated, "Weights did not update after training!")

        
    def test_train_loop_minimal(self):
        """Test that train_loop_extended trains the model and returns losses."""
        
        optimizer = th.optim.Adam(self.reduced_model.parameters())
        
        initial_weights = {k: v.clone() for k, v in self.reduced_model.state_dict().items()}

        # Call the function
        model, train_losses, test_losses = train_loop_minimal(
            epochs=self.epochs,
            model=self.reduced_model,
            device = th.device("cpu"),
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            loss_function=self.loss_function,
            optimizer=optimizer,
        )

        # Check the outputs
        for i in range(len(train_losses)):
            self.assertIsInstance(train_losses[i], float)
            self.assertIsInstance(test_losses[i], float)
        self.assertEqual(len(train_losses), self.epochs)  # 2 epochs
        self.assertEqual(len(test_losses), self.epochs)  # 2 epochs
        self.assertIsInstance(model, th.nn.Module)
        
        final_weights = {k: v.clone() for k, v in model.state_dict().items()}
        # Check that the model has been trained
        updated = any(not th.equal(initial_weights[k], final_weights[k]) for k in initial_weights)
        self.assertTrue(updated, "Weights did not update after training!")

if __name__ == "__main__":
    unittest.main()