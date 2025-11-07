import unittest
import torch
from models import get_model, CustomCNN, ResNetClassifier, EfficientNetClassifier

class TestModels(unittest.TestCase):

    def test_get_model(self):
        # Test getting a ResNet model
        model = get_model('resnet50', num_classes=10, pretrained=False)
        self.assertIsInstance(model, ResNetClassifier)

        # Test getting an EfficientNet model
        model = get_model('efficientnet_b0', num_classes=10, pretrained=False)
        self.assertIsInstance(model, EfficientNetClassifier)

        # Test getting a CustomCNN model
        model = get_model('custom_cnn', num_classes=10)
        self.assertIsInstance(model, CustomCNN)

        # Test for an unknown model
        with self.assertRaises(ValueError):
            get_model('unknown_model')

if __name__ == '__main__':
    unittest.main()
