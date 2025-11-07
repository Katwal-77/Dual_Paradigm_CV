import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
from pathlib import Path
from vehicle_detector import VehicleDetector
import vehicle_detection_config as config

class MockYOLOResult:
    def __init__(self, boxes):
        self.boxes = boxes

    def plot(self, **kwargs):
        return MagicMock()

class MockBox:
    def __init__(self, cls):
        self.cls = [cls]

class TestVehicleDetector(unittest.TestCase):

    def setUp(self):
        # Create a mock YOLO model
        self.mock_yolo = MagicMock()
        self.detector = VehicleDetector()
        self.detector.model = self.mock_yolo

        # Create a mock image file
        self.mock_image_dir = Path("mock_images")
        self.mock_image_dir.mkdir(exist_ok=True)
        self.mock_image_path = self.mock_image_dir / "test_image.jpg"
        with open(self.mock_image_path, "w") as f:
            f.write("dummy image")

    def tearDown(self):
        # Clean up the mock image directory
        shutil.rmtree(self.mock_image_dir)

    def test_detect_image(self):
        # Configure the mock to return some detections
        self.mock_yolo.predict.return_value = [
            MockYOLOResult(boxes=[MockBox(cls=2), MockBox(cls=3)])
        ]

        results, vehicle_count = self.detector.detect_image(str(self.mock_image_path))

        # Check that the predict method was called
        self.mock_yolo.predict.assert_called_once()

        # Check the vehicle count
        self.assertEqual(vehicle_count["car"], 1)
        self.assertEqual(vehicle_count["motorcycle"], 1)
        self.assertEqual(vehicle_count["bus"], 0)

    def test_batch_detect(self):
        # Configure the mock to return different detections for each call
        self.mock_yolo.predict.side_effect = [
            [MockYOLOResult(boxes=[MockBox(cls=2)])],
            [MockYOLOResult(boxes=[MockBox(cls=7)])]
        ]

        # Create a second mock image
        mock_image_path_2 = self.mock_image_dir / "test_image_2.jpg"
        with open(mock_image_path_2, "w") as f:
            f.write("dummy image")

        all_results = self.detector.batch_detect(str(self.mock_image_dir))

        # Check that the predict method was called twice
        self.assertEqual(self.mock_yolo.predict.call_count, 2)

        # Check the results
        self.assertEqual(len(all_results), 2)
        self.assertEqual(all_results[0]['total'], 1)
        self.assertEqual(all_results[0]['vehicle_count']['car'], 1)
        self.assertEqual(all_results[1]['total'], 1)
        self.assertEqual(all_results[1]['vehicle_count']['truck'], 1)

if __name__ == '__main__':
    unittest.main()
