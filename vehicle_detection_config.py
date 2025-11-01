"""
Configuration for YOLOv11 Vehicle Detection Project
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
VEHICLE_DATA_DIR = PROJECT_ROOT / "vehicle_data"
VEHICLE_MODELS_DIR = PROJECT_ROOT / "vehicle_models"
VEHICLE_RESULTS_DIR = PROJECT_ROOT / "vehicle_results"
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "detection_outputs"

# Create directories
VEHICLE_DATA_DIR.mkdir(exist_ok=True)
VEHICLE_MODELS_DIR.mkdir(exist_ok=True)
VEHICLE_RESULTS_DIR.mkdir(exist_ok=True)
DETECTION_OUTPUT_DIR.mkdir(exist_ok=True)

# YOLOv11 Model Configuration
YOLO_MODEL = 'yolo11n.pt'  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
# n=nano (fastest), s=small, m=medium, l=large, x=extra large (most accurate)

# Vehicle Classes (from COCO dataset)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle', 
    5: 'bus',
    7: 'truck',
    1: 'bicycle',
    6: 'train'
}

# Detection Configuration
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detection
IOU_THRESHOLD = 0.45  # IoU threshold for NMS (Non-Maximum Suppression)
IMG_SIZE = 640  # Image size for detection (640 is standard for YOLO)

# Training Configuration (if fine-tuning)
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.01
DEVICE = 'cpu'  # Will auto-detect GPU if available

# Visualization
SHOW_LABELS = True
SHOW_CONFIDENCE = True
BOX_THICKNESS = 2
FONT_SIZE = 0.5

