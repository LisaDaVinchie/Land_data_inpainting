import unittest
import torch as th
from pathlib import Path
import tempfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from preprocessing.cut_images import generate_image_dataset, select_random_points, map_random_points_to_images


class TestGenerateMaskedImageDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.processed_data_dir = Path(self.temp_dir.name)

        # Create dummy processed images
        self.n_images = 5
        self.x_shape_raw = 100
        self.y_shape_raw = 100
        self.n_channels = 4
        for i in range(self.n_images):
            dummy_image = th.rand(self.n_channels - 1, self.x_shape_raw, self.y_shape_raw)
            years = th.randint(2000, 2020, (1,))
            months = th.randint(1, 13, (1,))
            days = th.randint(1, 29, (1,))
            dummy_date = f"{years.item()}_{months.item()}_{days.item()}"
            th.save(dummy_image, self.processed_data_dir / f"{dummy_date}.pt")

        # Test parameters
        self.n_cutted_images = 8
        self.cutted_width = 32
        self.cutted_height = 32
        self.mask_percentage = 0.5
        self.non_masked_channels = [0]
        self.placeholder = -1.0

    def tearDown(self):
        """Clean up temporary directory after tests."""
        self.temp_dir.cleanup()

    def test_generate_masked_datasets_shapes_and_dtypes(self):
        """Test that generate_masked_image_dataset returns tensors with the correct shapes and dtypes."""
        # Select random points and map them to images
        random_points = select_random_points(
            original_width=self.x_shape_raw,
            original_height=self.y_shape_raw,
            n_points=self.n_cutted_images,
            final_width=self.cutted_width,
            final_height=self.cutted_height,
        )
        processed_images_paths = list(self.processed_data_dir.glob("*.pt"))
        path_to_indices = map_random_points_to_images(processed_images_paths, random_points)

        # Generate the dataset
        dataset_ext, dataset_min = generate_image_dataset(
            original_width=self.x_shape_raw,
            original_height=self.y_shape_raw,
            n_images=self.n_cutted_images,
            final_width=self.cutted_width,
            final_height=self.cutted_height,
            n_channels=self.n_channels,
            masked_fraction=self.mask_percentage,
            non_masked_channels_list=self.non_masked_channels,
            path_to_indices_map=path_to_indices,
            minimal_data=True, extended_data=True,
            placeholder=self.placeholder,
            nans_threshold=0.5
        )

        # Check that the output is a dictionary with the correct keys
        self.assertIsInstance(dataset_ext, dict)
        self.assertIsInstance(dataset_min, dict)
        self.assertSetEqual(set(dataset_ext.keys()), {"masked_images", "inverse_masked_images", "masks"})
        self.assertSetEqual(set(dataset_min.keys()), {"images", "masks"})

        # Check that each tensor has the correct shape and dtype
        expected_shape = (self.n_cutted_images, self.n_channels, self.cutted_width, self.cutted_height)
        expected_dtype = th.float32

        for key in dataset_ext.keys():
            self.assertIsInstance(dataset_ext[key], th.Tensor)
            self.assertEqual(dataset_ext[key].shape, expected_shape)
            self.assertEqual(dataset_ext[key].dtype, expected_dtype)
            
        for key in dataset_min.keys():
            self.assertIsInstance(dataset_min[key], th.Tensor)
            self.assertEqual(dataset_min[key].shape, expected_shape)
            self.assertEqual(dataset_min[key].dtype, expected_dtype)
            
    def test_dataset_kind_switch(self):
        # Generate the dataset
        random_points = select_random_points(
            original_width=self.x_shape_raw,
            original_height=self.y_shape_raw,
            n_points=self.n_cutted_images,
            final_width=self.cutted_width,
            final_height=self.cutted_height,
        )
        processed_images_paths = list(self.processed_data_dir.glob("*.pt"))
        path_to_indices = map_random_points_to_images(processed_images_paths, random_points)
        
        dataset_ext, dataset_min = generate_image_dataset(
            original_width=self.x_shape_raw,
            original_height=self.y_shape_raw,
            n_images=self.n_cutted_images,
            final_width=self.cutted_width,
            final_height=self.cutted_height,
            n_channels=self.n_channels,
            masked_fraction=self.mask_percentage,
            non_masked_channels_list=self.non_masked_channels,
            path_to_indices_map=path_to_indices,
            minimal_data=True, extended_data=False,
            placeholder=self.placeholder,
            nans_threshold=0.5
        )
        
        # Check that dataset_ext is None and dataset_min is not None
        self.assertIsNone(dataset_ext)
        self.assertIsNotNone(dataset_min)
        
        dataset_ext, dataset_min = generate_image_dataset(
            original_width=self.x_shape_raw,
            original_height=self.y_shape_raw,
            n_images=self.n_cutted_images,
            final_width=self.cutted_width,
            final_height=self.cutted_height,
            n_channels=self.n_channels,
            masked_fraction=self.mask_percentage,
            non_masked_channels_list=self.non_masked_channels,
            path_to_indices_map=path_to_indices,
            minimal_data=False, extended_data=True,
            placeholder=self.placeholder,
            nans_threshold=0.5
        )
        
        self.assertIsNotNone(dataset_ext)
        self.assertIsNone(dataset_min)
        
        dataset_ext, dataset_min = generate_image_dataset(
            original_width=self.x_shape_raw,
            original_height=self.y_shape_raw,
            n_images=self.n_cutted_images,
            final_width=self.cutted_width,
            final_height=self.cutted_height,
            n_channels=self.n_channels,
            masked_fraction=self.mask_percentage,
            non_masked_channels_list=self.non_masked_channels,
            path_to_indices_map=path_to_indices,
            minimal_data=False, extended_data=False,
            placeholder=self.placeholder,
            nans_threshold=0.5
        )
        
        self.assertIsNone(dataset_ext)
        self.assertIsNone(dataset_min)


if __name__ == "__main__":
    unittest.main()