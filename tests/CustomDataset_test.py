import unittest
import torch as th
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.CustomDataset import create_dataloaders

class TestCreateDataloaders(unittest.TestCase):
    def test_minimal_dataset(self):
        # Create a minimal dataset
        dataset = {
            'images': th.randn(100, 3, 64, 64),
            'masks': th.randn(100, 3, 64, 64)
        }
        
        # Create dataloaders
        train_loader, test_loader = create_dataloaders(dataset, train_perc=0.8, batch_size=10)
        
        # Check if the dataloaders are created correctly
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)
        
        # Check the length of the dataloaders
        self.assertEqual(len(train_loader.dataset), 80)
        self.assertEqual(len(test_loader.dataset), 20)
        
        # Check the batch size
        self.assertEqual(train_loader.batch_size, 10)
        self.assertEqual(test_loader.batch_size, 10)
        
         # Check shuffling
        first_epoch_train_data = [batch for batch in train_loader][0]
        second_epoch_train_data = [batch for batch in train_loader][0]
        
        for i in range(len(first_epoch_train_data)):
            self.assertFalse(th.equal(first_epoch_train_data[i], second_epoch_train_data[i]))
            
        
        first_epoch_test_data = [batch for batch in test_loader][0]
        second_epoch_test_data = [batch for batch in test_loader][0]
        
        for i in range(len(first_epoch_test_data)):
            self.assertTrue(th.equal(first_epoch_test_data[i], second_epoch_test_data[i]))
        

    def test_extended_dataset(self):
        # Create an extended dataset
        dataset = {
            'masked_images': th.randn(100, 3, 64, 64),
            'inverse_masked_images': th.randn(100, 3, 64, 64),
            'masks': th.randn(100, 1, 64, 64)
        }
        
        # Create dataloaders
        train_loader, test_loader = create_dataloaders(dataset, train_perc=0.7, batch_size=20)
        
        # Check if the dataloaders are created correctly
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)
        
        # Check the length of the dataloaders
        self.assertEqual(len(train_loader.dataset), 70)
        self.assertEqual(len(test_loader.dataset), 30)
        
        # Check the batch size
        self.assertEqual(train_loader.batch_size, 20)
        self.assertEqual(test_loader.batch_size, 20)
        
        # Check shuffling
        first_epoch_train_data = [batch for batch in train_loader][0]
        second_epoch_train_data = [batch for batch in train_loader][0]
        
        for i in range(len(first_epoch_train_data)):
            self.assertFalse(th.equal(first_epoch_train_data[i], second_epoch_train_data[i]))
            
        
        first_epoch_test_data = [batch for batch in test_loader][0]
        second_epoch_test_data = [batch for batch in test_loader][0]
        
        for i in range(len(first_epoch_test_data)):
            self.assertTrue(th.equal(first_epoch_test_data[i], second_epoch_test_data[i]))

    def test_invalid_dataset_keys(self):
        # Create a dataset with invalid number of keys
        dataset = {
            'masked_images': th.randn(100, 3, 64, 64),
            'inverse_masked_images': th.randn(100, 3, 64, 64),
            'masks': th.randn(100, 1, 64, 64),
            'extra_key': th.randn(100, 1, 64, 64)
        }
        
        # Check if the function raises a ValueError
        with self.assertRaises(ValueError):
            create_dataloaders(dataset, train_perc=0.8, batch_size=10)

    def test_mismatched_dataset_lengths(self):
        # Create a dataset with mismatched lengths
        dataset = {
            'masked_images': th.randn(100, 3, 64, 64),
            'inverse_masked_images': th.randn(90, 3, 64, 64),
            'masks': th.randn(100, 1, 64, 64)
        }
        
        # Check if the function raises an AssertionError
        with self.assertRaises(AssertionError):
            create_dataloaders(dataset, train_perc=0.8, batch_size=10)

    def test_edge_cases(self):
        # Create a minimal dataset
        dataset = {
            'images': th.randn(100, 3, 64, 64),
            'masks': th.randn(100, 3, 64, 64)
        }
        
        # Test with train_perc = 0
        with self.assertRaises(ValueError):
            create_dataloaders(dataset, train_perc=0.0, batch_size=10)
        
        
        # Test with train_perc = 1
        with self.assertRaises(ValueError):
            create_dataloaders(dataset, train_perc=1.0, batch_size=10)

if __name__ == '__main__':
    unittest.main()