import unittest
import torch as th
from pathlib import Path
import tempfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from preprocessing.cut_images import CutAndMaskImage
from preprocessing.mask_data import SquareMask


class TestGenerateMaskedImageDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.processed_data_dir = Path(self.temp_dir.name)

        # Create dummy processed images
        self.n_images = 5
        self.x_shape_raw = 30
        self.y_shape_raw = 30
        self.n_channels = 4
        # Test parameters
        self.n_cutted_images = 8
        self.cutted_nrows = 10
        self.cutted_ncols = 10
        mask_percentage = 0.5
        self.masked_channels = [0, 2]
        self.placeholder = -1.0
        
        # Create dummy input with nans
        for i in range(self.n_images):
            # Create a random image
            dummy_image = th.rand(self.n_channels - 1, self.x_shape_raw, self.y_shape_raw)
            
            # Select random points to mark as nan
            random_x = th.randint(0, self.x_shape_raw, (1,))
            random_y = th.randint(0, self.y_shape_raw, (1,))
            
            # Create a mask with 0s in the selected points
            nan_mask = th.ones(self.n_channels - 1, self.x_shape_raw, self.y_shape_raw)
            nan_mask[self.masked_channels, random_x, random_y] = 0
            
            # Set the selected points to nan
            dummy_image[nan_mask == 0] = th.nan
            
            # Save the image with the correct
            years = th.randint(2000, 2020, (1,))
            months = th.randint(1, 13, (1,))
            days = th.randint(1, 29, (1,))
            dummy_date = f"{years.item()}_{months.item()}_{days.item()}"
            th.save(dummy_image, self.processed_data_dir / f"{dummy_date}.pt")
            
        self.cut_class = CutAndMaskImage(
            original_nrows=self.x_shape_raw,
            original_ncols=self.y_shape_raw,
            final_nrows=self.cutted_nrows,
            final_ncols=self.cutted_ncols,
            nans_threshold=0.5,
            n_cutted_images=self.n_cutted_images)
        
        # Select random points and map them to images
        random_points = self.cut_class.select_random_points(n_points=self.n_cutted_images)
        
        processed_images_paths = list(self.processed_data_dir.glob("*.pt"))
        self.path_to_indices = self.cut_class.map_random_points_to_images(processed_images_paths, random_points)
        
        self.mask_function = SquareMask(image_nrows=self.cutted_nrows,
                                   image_ncols=self.cutted_ncols,
                                   mask_percentage=mask_percentage)

        # Generate the dataset
        self.dataset_ext, self.dataset_min, self.nans_mask = self.cut_class.generate_image_dataset(
                        n_channels=self.n_channels,
                        mask_function=self.mask_function,
                        masked_channels_list=self.masked_channels,
                        path_to_indices_map=self.path_to_indices,
                        minimal_data=True, extended_data=True,
                        placeholder=self.placeholder)

    def tearDown(self):
        """Clean up temporary directory after tests."""
        self.temp_dir.cleanup()
        
    def test_generate_masked_datasets_keys(self):
        """Test that generate_masked_image_dataset returns a dictionary with the correct keys."""
        self.assertIsInstance(self.dataset_ext, dict)
        self.assertIsInstance(self.dataset_min, dict)
        # Check that the output is a dictionary with the correct keys
        self.assertSetEqual(set(self.dataset_ext.keys()), {"masked_images", "inverse_masked_images", "masks"})
        self.assertSetEqual(set(self.dataset_min.keys()), {"images", "masks"})

    def test_generate_masked_datasets_shapes_and_dtypes(self):
        """Test that generate_masked_image_dataset returns tensors with the correct shapes and dtypes."""
        
        # Check that each tensor has the correct shape and dtype
        expected_shape = (self.n_cutted_images, self.n_channels, self.cutted_nrows, self.cutted_ncols)
        expected_dtype = th.float32

        for key in self.dataset_ext.keys():
            self.assertIsInstance(self.dataset_ext[key], th.Tensor)
            self.assertEqual(self.dataset_ext[key].shape, expected_shape)
            self.assertEqual(self.dataset_ext[key].dtype, expected_dtype)
            
        for key in self.dataset_min.keys():
            self.assertIsInstance(self.dataset_min[key], th.Tensor)
            self.assertEqual(self.dataset_min[key].shape, expected_shape)
            self.assertEqual(self.dataset_min[key].dtype, expected_dtype)
            
        self.assertIsInstance(self.nans_mask, th.Tensor)
        self.assertEqual(self.nans_mask.shape, expected_shape)
        
    def test_generate_masked_datasets_values(self):
        """Test that generate_masked_image_dataset returns tensors with the correct masked channels."""
        
        # Check that the masked channels have the placeholder value and the other channels do not
        keys = list(self.dataset_ext.keys())
        for i in range(self.n_cutted_images):
            for j in self.masked_channels:
                self.assertTrue((self.dataset_ext[keys[0]][:, j, :, :] == self.placeholder).any())
        
        # Check that there are no NaNs in dataset_ext
        for key in self.dataset_ext.keys():
            self.assertFalse(th.isnan(self.dataset_ext[key]).any(), f"NaNs found in dataset_ext[{key}]")
        
        # Check that the masked channels have 0's and the other channels have all 1's
        keys = list(self.dataset_min.keys())
        for i in range(self.n_cutted_images):
            for j in self.masked_channels:
                self.assertTrue((self.dataset_min[keys[1]][:, j, :, :] == 0).any())
        
        # Check that the mask has no nan
        mask_key = list(self.dataset_min.keys())[1]
        self.assertFalse(th.isnan(self.dataset_min[mask_key]).any(), f"NaNs found in dataset_min[{mask_key}]")
        
        # Check that the nan values are replaced with the placeholder value
        for i in range(self.n_cutted_images):
            for j in range(self.n_channels):
                nan_mask = self.nans_mask[i, j, :, :]
                image = self.dataset_min[keys[0]][i, j, :, :]
                self.assertTrue(th.all(image[nan_mask == 0] == self.placeholder), "Not all values under the nan mask are placeholder")
                
    def test_nans_coverage(self):
        """Test that the masks cover all the nans in the images."""
        # Check that the 0s in the masks of dataset_min[keys[1]] cover all the nans in dataset_min[keys[0]]
        
        keys = list(self.dataset_min.keys())
        for i in range(self.n_cutted_images):
            nan_mask = self.nans_mask[i]
            mask = self.dataset_min[keys[1]][i]
            
            # Check that the mask covers all the nans
            self.assertTrue(th.all(mask[nan_mask == 0] == 0), "Mask does not cover all nans")
             
    def test_dataset_kind_switch_extended_data_false(self):
        """Test that the dataset kind switch works correctly. In this case, the extended dataset should be None."""
        
        dataset_ext, dataset_min, nans_mask = self.cut_class.generate_image_dataset(
                        n_channels=self.n_channels,
                        mask_function=self.mask_function,
                        masked_channels_list=self.masked_channels,
                        path_to_indices_map=self.path_to_indices,
                        minimal_data=True, extended_data=False,
                        placeholder=self.placeholder)
        
        # Check that dataset_ext is None and dataset_min is not None
        self.assertIsNone(dataset_ext)
        self.assertIsNotNone(dataset_min)
    
    def test_dataset_kind_switch_minimal_data_false(self):
        """Test that the dataset kind switch works correctly. In this case, the minimal dataset should be None."""
        
        dataset_ext, dataset_min, nans_mask = self.cut_class.generate_image_dataset(
                        n_channels=self.n_channels,
                        mask_function=self.mask_function,
                        masked_channels_list=self.masked_channels,
                        path_to_indices_map=self.path_to_indices,
                        minimal_data=False, extended_data=True,
                        placeholder=self.placeholder)
        
        self.assertIsNotNone(dataset_ext)
        self.assertIsNone(dataset_min)
    
    def test_dataset_kind_switch_both_false(self):
        """Test that the dataset kind switch works correctly. In this case, both datasets should be None."""
        
        dataset_ext, dataset_min, nans_mask = self.cut_class.generate_image_dataset(
                        n_channels=self.n_channels,
                        mask_function=self.mask_function,
                        masked_channels_list=self.masked_channels,
                        path_to_indices_map=self.path_to_indices,
                        minimal_data=False, extended_data=False,
                        placeholder=self.placeholder)
        
        self.assertIsNone(dataset_ext)
        self.assertIsNone(dataset_min)
        
if __name__ == "__main__":
    unittest.main()