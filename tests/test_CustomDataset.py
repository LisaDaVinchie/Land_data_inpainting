import unittest
from pathlib import Path
import torch as th
import tempfile
import os
import sys
# Add ../src to sys.path to import CreateDataloaders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from CustomDataset_v2 import CreateDataloaders, CustomDatasetClass

class TestCreateDataloaders(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and dummy dataset for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "test_dataset.pt"
        
        # Create a dummy dataset
        self.dataset = {
            'images': th.randn(100, 3, 64, 64),  # 100 RGB images of 64x64
            'masks': th.randn(100, 1, 64, 64)    # 100 single-channel masks
        }
        th.save(self.dataset, self.dataset_path)
        
        # Also create an invalid dataset with mismatched lengths
        self.invalid_dataset_path = Path(self.temp_dir.name) / "invalid_dataset.pt"
        invalid_dataset = {
            'images': th.randn(50, 3, 64, 64),
            'masks': th.randn(60, 1, 64, 64)
        }
        th.save(invalid_dataset, self.invalid_dataset_path)
        
        # And a dataset with wrong number of keys
        self.wrong_keys_dataset_path = Path(self.temp_dir.name) / "wrong_keys_dataset.pt"
        wrong_keys_dataset = {
            'images': th.randn(50, 3, 64, 64),
            'masks': th.randn(50, 1, 64, 64),
            'extra': th.randn(50, 1, 64, 64)
        }
        th.save(wrong_keys_dataset, self.wrong_keys_dataset_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_valid_parameters(self):
        # Test that initialization works with valid parameters
        dataloader_creator = CreateDataloaders(
            dataset_path=self.dataset_path,
            train_perc=0.8,
            batch_size=32
        )
        self.assertEqual(dataloader_creator.dataset_path, self.dataset_path)
        self.assertEqual(dataloader_creator.train_perc, 0.8)
        self.assertEqual(dataloader_creator.batch_size, 32)

    def test_init_invalid_train_perc(self):
        # Test that invalid train_perc raises ValueError
        with self.assertRaises(ValueError):
            CreateDataloaders(self.dataset_path, -0.1, 32)
        with self.assertRaises(ValueError):
            CreateDataloaders(self.dataset_path, 1.1, 32)
        with self.assertRaises(ValueError):
            CreateDataloaders(self.dataset_path, 1.0, 32)
        with self.assertRaises(ValueError):
            CreateDataloaders(self.dataset_path, 0.0, 32)

    def test_init_invalid_batch_size(self):
        # Test that invalid batch_size raises ValueError
        with self.assertRaises(ValueError):
            CreateDataloaders(self.dataset_path, 0.8, 0)
        with self.assertRaises(ValueError):
            CreateDataloaders(self.dataset_path, 0.8, -10)

    def test_load_dataset(self):
        # Test that _load_dataset correctly loads the dataset
        dataloader_creator = CreateDataloaders(self.dataset_path, 0.8, 32)
        loaded_dataset = dataloader_creator._load_dataset()
        
        self.assertIsInstance(loaded_dataset, dict)
        self.assertEqual(set(loaded_dataset.keys()), {'images', 'masks'})
        self.assertEqual(loaded_dataset['images'].shape, (100, 3, 64, 64))
        self.assertEqual(loaded_dataset['masks'].shape, (100, 1, 64, 64))

    def test_validate_dataset_length_valid(self):
        # Test _validate_dataset_length with valid dataset
        dataloader_creator = CreateDataloaders(self.dataset_path, 0.8, 32)
        loaded_dataset = dataloader_creator._load_dataset()
        length = dataloader_creator._validate_dataset_length(loaded_dataset)
        self.assertEqual(length, 100)

    def test_validate_dataset_length_invalid(self):
        # Test _validate_dataset_length with invalid dataset
        dataloader_creator = CreateDataloaders(self.invalid_dataset_path, 0.8, 32)
        loaded_dataset = dataloader_creator._load_dataset()
        
        with self.assertRaises(AssertionError):
            dataloader_creator._validate_dataset_length(loaded_dataset)

    def test_validate_dataset_length_wrong_keys(self):
        # Test _validate_dataset_length with wrong number of keys
        dataloader_creator = CreateDataloaders(self.wrong_keys_dataset_path, 0.8, 32)
        loaded_dataset = dataloader_creator._load_dataset()
        
        with self.assertRaises(ValueError):
            dataloader_creator._validate_dataset_length(loaded_dataset)

    def test_get_train_test_indices(self):
        # Test that _get_train_test_indices returns correct splits
        dataloader_creator = CreateDataloaders(self.dataset_path, 0.8, 32)
        loaded_dataset = dataloader_creator._load_dataset()
        
        train_indices, test_indices = dataloader_creator._get_train_test_indices(loaded_dataset)
        
        # Check lengths
        self.assertEqual(len(train_indices), 80)  # 80% of 100
        self.assertEqual(len(test_indices), 20)  # 20% of 100
        
        # Check no overlap
        self.assertEqual(len(set(train_indices) & set(test_indices)), 0)
        
        # Check all indices are covered
        all_indices = set(train_indices + test_indices)
        self.assertEqual(len(all_indices), 100)
        self.assertEqual(all_indices, set(range(100)))

    def test_create_dataloaders(self):
        # Test that create() returns correct dataloaders
        dataloader_creator = CreateDataloaders(self.dataset_path, 0.8, 32)
        train_loader, test_loader = dataloader_creator.create()
        
        # Check types
        self.assertIsInstance(train_loader, th.utils.data.DataLoader)
        self.assertIsInstance(test_loader, th.utils.data.DataLoader)
        
        # Check batch sizes
        self.assertEqual(train_loader.batch_size, 32)
        self.assertEqual(test_loader.batch_size, 32)
        
        # Check dataset sizes
        self.assertEqual(len(train_loader.dataset), 80)
        self.assertEqual(len(test_loader.dataset), 20)
        
        # Check one batch
        train_batch = next(iter(train_loader))
        self.assertEqual(len(train_batch), 2)  # image and mask
        self.assertEqual(train_batch[0].shape, (32, 3, 64, 64))
        self.assertEqual(train_batch[1].shape, (32, 1, 64, 64))

    def test_custom_dataset_class(self):
        # Test the CustomDatasetClass
        dataset = {
            'images': th.randn(10, 3, 64, 64),
            'masks': th.randn(10, 1, 64, 64)
        }
        custom_dataset = CustomDatasetClass(dataset)
        
        # Test length
        self.assertEqual(len(custom_dataset), 10)
        
        # Test getitem
        img, mask = custom_dataset[0]
        self.assertEqual(img.shape, (3, 64, 64))
        self.assertEqual(mask.shape, (1, 64, 64))
        
        # Test with different key names
        dataset = {
            'data': th.randn(5, 3, 32, 32),
            'labels': th.randn(5, 1, 32, 32)
        }
        custom_dataset = CustomDatasetClass(dataset)
        self.assertEqual(len(custom_dataset), 5)
        img, mask = custom_dataset[0]
        self.assertEqual(img.shape, (3, 32, 32))
        self.assertEqual(mask.shape, (1, 32, 32))

if __name__ == '__main__':
    unittest.main()