"""
Configuration file for Weather Classification Project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data_classification"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset configuration
CLASSES = [
    'cloudy', 'day', 'dust', 'fall', 'fog', 'hurricane', 
    'lightning', 'night', 'rain', 'snow', 'spring', 
    'summer', 'sun', 'tornado', 'windy', 'winter'
]
NUM_CLASSES = len(CLASSES)

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 224
NUM_WORKERS = 4

# Model configuration
MODEL_NAME = 'resnet50'  # Options: 'resnet50', 'efficientnet_b0', 'custom_cnn'
PRETRAINED = True

# Training settings
EARLY_STOPPING_PATIENCE = 10
SAVE_BEST_MODEL = True
RANDOM_SEED = 42

# Device configuration
DEVICE = 'cuda'  # Will auto-detect in code

