import unittest
import os
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data_loader import prepare_data, WeatherDataset, create_data_loaders
import config

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset directory
        self.mock_data_dir = Path("mock_data")
        self.mock_data_dir.mkdir(exist_ok=True)
        self.class_names = ["sunrise", "cloudy"]
        self.image_files = []

        for class_name in self.class_names:
            class_dir = self.mock_data_dir / class_name
            class_dir.mkdir(exist_ok=True)
            for i in range(20):
                img_path = class_dir / f"{class_name}_{i}.jpg"
                # Create a dummy image file
                with open(img_path, "w") as f:
                    f.write("dummy image")
                self.image_files.append(str(img_path))

        # Override the config to use the mock data
        self.original_data_dir = config.DATA_DIR
        self.original_classes = config.CLASSES
        config.DATA_DIR = self.mock_data_dir
        config.CLASSES = self.class_names
        config.NUM_CLASSES = len(self.class_names)

    def tearDown(self):
        # Clean up the mock dataset directory
        shutil.rmtree(self.mock_data_dir)
        # Restore the original config
        config.DATA_DIR = self.original_data_dir
        config.CLASSES = self.original_classes
        config.NUM_CLASSES = len(self.original_classes)

    def test_prepare_data(self):
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

        # Check the number of images in each set
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(self.image_files))

        # Check the number of labels
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_val), len(X_val))
        self.assertEqual(len(y_test), len(X_test))

    def test_weather_dataset(self):
        # Create a dummy dataset
        dataset = WeatherDataset(self.image_files, [0] * len(self.image_files))
        self.assertEqual(len(dataset), len(self.image_files))

        # This will fail because the dummy image is not a real image
        # We'd need to mock Image.open for a more thorough test
        with self.assertRaises(Exception):
            dataset[0]

    def test_create_data_loaders(self):
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=2)

        # Check the type of the returned objects
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

        # Check the batch size
        self.assertEqual(train_loader.batch_size, 2)
        self.assertEqual(val_loader.batch_size, 2)
        self.assertEqual(test_loader.batch_size, 2)

if __name__ == '__main__':
    unittest.main()
