import unittest
from unittest.mock import patch
import torch as th
from pathlib import Path
from models import DINCAE_pconvs, simple_conv
import sys
import os
import json
from tempfile import NamedTemporaryFile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from train import track_memory, change_dataset_idx, validate_paths, train_loop_extended, train_loop_minimal
from CustomDataset import create_dataloaders

class TestTrainingFunctions(unittest.TestCase):
    def setUp(self):
        """Create a temporary JSON file with parameters."""
        self.batch_size = 64
        self.dataset_len = 100
        self.n_channels = 3
        self.height = 64
        self.width = 64
        self.epochs = 3
        
        self.extended_model_params = {
            "simple_conv": {
                "middle_channels": [64, 128, 256],
                "kernel_size": [3, 3, 3],
                "stride": [2, 2, 2],
                "padding": [1, 1, 1],
                "output_padding": [1, 1, 1]
                },  
            "dataset":{
                "n_channels": self.n_channels
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
                "n_channels": self.n_channels,
                "cutted_width": self.width,
                "cutted_height": self.height
            },
            "DINCAE_pconvs": {
                "middle_channels": [16, 16, 16, 16, 16],
                "kernel_sizes": [5, 3, 5, 3, 5],
                "pooling_sizes": [2, 2, 2, 2, 2],
                "interp_mode": "nearest"
            }
        }
        
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.extended_model_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.reduced_params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        self.reduced_model = DINCAE_pconvs(params_path=self.reduced_params_path)
        
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
                validate_paths(Path("/fake/path"))
            
            mock_exists.return_value = True
            try:
                validate_paths(Path("/real/path"))
            except FileNotFoundError:
                self.fail("validate_paths raised FileNotFoundError unexpectedly")

    def test_train_loop_extended(self):
        """Test that train_loop_extended trains the model and returns losses."""
        # Mock the model and dataloader
        test_dataset = {
            "masked_images": th.randn(self.dataset_len, self.n_channels, self.width, self.height),
            "inverse_masked_images": th.randn(self.dataset_len, self.n_channels, self.width, self.height),
            "masks": th.ones(self.dataset_len, self.n_channels, self.width, self.height)}
        
        train_loader, test_loader = create_dataloaders(test_dataset, 0.8, batch_size=self.batch_size)

        # Use a real loss function and optimizer
        loss_function = th.nn.MSELoss()
        optimizer = th.optim.Adam(self.extended_model.parameters())
        
        initial_weights = self.extended_model.state_dict().copy()

        # Call the function
        model, train_losses, test_losses = train_loop_extended(
            epochs=self.epochs,
            placeholder=0.0,
            model=self.extended_model,
            device=th.device("cpu"),
            train_loader=train_loader,
            test_loader=test_loader,
            loss_function=loss_function,
            optimizer=optimizer,
        )
        
        model.eval()
        final_weights = model.state_dict().copy()

        # Check the outputs
        for i in range(len(train_losses)):
            self.assertIsInstance(train_losses[i], float)
            self.assertIsInstance(test_losses[i], float)
        self.assertEqual(len(train_losses), self.epochs)  # 2 epochs
        self.assertEqual(len(test_losses), self.epochs)  # 2 epochs
        self.assertIsInstance(model, th.nn.Module)
        
        
        # Check that the model has been trained
        differences = 0
        for key in initial_weights:
            if not th.equal(initial_weights[key], final_weights[key]):
                differences += 1
        self.assertGreaterEqual(differences, 1, "Less than two weight tensors were updated during training")
        
    def test_train_loop_minimal(self):
        """Test that train_loop_extended trains the model and returns losses."""
        # Mock the model and dataloader
        test_dataset = {
            "images": th.randn(self.dataset_len, self.n_channels, self.width, self.height),
            "masks": th.ones(self.dataset_len, self.n_channels, self.width, self.height)
            }
        
        train_loader, test_loader = create_dataloaders(test_dataset, 0.8, batch_size=self.batch_size)

        # Use a real loss function and optimizer
        loss_function = th.nn.MSELoss()
        optimizer = th.optim.Adam(self.reduced_model.parameters())

        # Call the function
        model, train_losses, test_losses = train_loop_minimal(
            epochs=self.epochs,
            placeholder=0.0,
            model=self.reduced_model,
            device = th.device("cpu"),
            train_loader=train_loader,
            test_loader=test_loader,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        # Check the outputs
        for i in range(len(train_losses)):
            self.assertIsInstance(train_losses[i], float)
            self.assertIsInstance(test_losses[i], float)
        self.assertEqual(len(train_losses), self.epochs)  # 2 epochs
        self.assertEqual(len(test_losses), self.epochs)  # 2 epochs
        self.assertIsInstance(model, th.nn.Module)

if __name__ == "__main__":
    unittest.main()