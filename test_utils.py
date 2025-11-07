import unittest
import torch
import torch.nn as nn
import os
from pathlib import Path
from utils import EarlyStopping, save_checkpoint, load_checkpoint, count_parameters

class TestUtils(unittest.TestCase):

    def test_early_stopping(self):
        early_stopping = EarlyStopping(patience=3, min_delta=0.1)
        self.assertIsNone(early_stopping.best_loss)
        self.assertEqual(early_stopping.counter, 0)
        self.assertFalse(early_stopping.early_stop)

        # Test with improving loss
        early_stopping(1.0)
        self.assertEqual(early_stopping.best_loss, 1.0)
        self.assertEqual(early_stopping.counter, 0)

        early_stopping(0.8)
        self.assertEqual(early_stopping.best_loss, 0.8)
        self.assertEqual(early_stopping.counter, 0)

        # Test with non-improving loss
        early_stopping(0.85)
        self.assertEqual(early_stopping.counter, 1)

        early_stopping(0.9)
        self.assertEqual(early_stopping.counter, 2)

        # Test early stop
        early_stopping(0.95)
        self.assertTrue(early_stopping.early_stop)

    def test_checkpointing(self):
        # Create a dummy model and optimizer
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        epoch = 10
        accuracy = 0.95
        filepath = "test_checkpoint.pth"

        # Test save_checkpoint
        save_checkpoint(model, optimizer, epoch, accuracy, filepath)
        self.assertTrue(Path(filepath).exists())

        # Test load_checkpoint
        loaded_model = nn.Linear(10, 2)
        loaded_optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.01)
        l_model, l_optimizer, l_epoch, l_accuracy = load_checkpoint(loaded_model, loaded_optimizer, filepath)

        self.assertEqual(l_epoch, epoch)
        self.assertEqual(l_accuracy, accuracy)
        self.assertTrue(torch.equal(model.state_dict()['weight'], l_model.state_dict()['weight']))

        # Clean up the created file
        os.remove(filepath)

    def test_count_parameters(self):
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        # Total params = (10*20 + 20) + (20*5 + 5) = 220 + 105 = 325
        # All params are trainable
        total_params, trainable_params = count_parameters(model)
        self.assertEqual(total_params, 325)
        self.assertEqual(trainable_params, 325)

        # Freeze a layer
        for param in model[0].parameters():
            param.requires_grad = False

        # Trainable params = 325 - (10*20 + 20) = 105
        total_params, trainable_params = count_parameters(model)
        self.assertEqual(total_params, 325)
        self.assertEqual(trainable_params, 105)

if __name__ == '__main__':
    unittest.main()
