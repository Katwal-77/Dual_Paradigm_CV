# Dual_Paradigm_CV — Weather Classification + Vehicle Detection

End‑to‑end computer vision project combining:
- Weather image classification (training, evaluation, prediction) using PyTorch + transfer learning.
- Vehicle detection on images, video, and webcam using YOLOv11 (Ultralytics).

This guide walks you from download → setup → how to run both pipelines and how they work.

## Project Overview

- Weather classification: trains a model on 16 weather classes, saves checkpoints, evaluates, and produces presentation‑ready plots.
- Vehicle detection: runs YOLOv11 to detect cars, buses, trucks, motorcycles, bicycles, trains on images/videos/webcam.
- Reproducibility: centralized config files, saved artifacts, and consistent folder layout for results.

## Directory Layout

- `config.py` — Classification config (paths, classes, hyperparams, model name).
- `models.py` — Model factory (`resnet50`, `efficientnet_b0`, `custom_cnn`).
- `data_loader.py` — Dataset split + PyTorch DataLoaders.
- `train.py` — Train loop with early stopping and scheduler.
- `evaluate.py` — Metrics + plots (confusion matrix, per‑class, history).
- `predict.py` — Inference and visualization for a single image.
- `vehicle_detector.py` — Programmatic YOLOv11 detector (image/video/webcam/batch).
- `demo_vehicle_detection.py` — CLI wrapper for running detection.
- `vehicle_detection_config.py` — Detection config (thresholds, classes, dirs, model weight).
- `results/` — Classification outputs (plots, JSON metrics, history).
- `models/` — Saved classification checkpoints.
- `detection_outputs/` — Saved images/videos with bounding boxes.
- `outputs/` — Presentation images (e.g., architecture, predictions).
- `data_classification/` — Your dataset folder (not included in the repo; see below).

## Prerequisites

- Python 3.9+ recommended (PyTorch 2.x).
- pip, venv (or Conda).
- Optional GPU with CUDA for faster training/inference.

## Get the Code

- Option A — Clone:
  - `git clone https://github.com/Katwal-77/Dual_Paradigm_CV.git`
  - `cd Dual_Paradigm_CV`
- Option B — Download ZIP: extract and open the folder in your IDE/terminal.

## Setup

1) Create and activate a virtual environment
- Windows: `python -m venv venv && venv\Scripts\activate`
- macOS/Linux: `python -m venv venv && source venv/bin/activate`

2) Install dependencies
- Classification core: `pip install -r requirements.txt`
- YOLOv11 detection: `pip install -r requirements_yolo.txt`

3) Verify install (optional quick checks)
- `python -c "import torch, ultralytics, cv2; print('OK')"`

## Dataset (Weather Classification)

- Create `data_classification/` with 16 subfolders (one per class):
  `cloudy, day, dust, fall, fog, hurricane, lightning, night, rain, snow, spring, summer, sun, tornado, windy, winter`.
- Place images inside each class folder (JPG/PNG). The code splits into train/val/test automatically using `config.py` ratios.
- A reference setup uses ~3,360 images in total (~210 per class).

## Quick Start — Weather Classification

1) Train
- `python train.py`
- Saves: `models/best_model.pth`, `models/latest_model.pth`, and `results/training_history.json`.

2) Evaluate
- `python evaluate.py`
- Saves to `results/`: `confusion_matrix.png`, `per_class_accuracy.png`, `training_history.png`, `test_metrics.json`.

3) Predict on a single image
- Example: `python predict.py --image data_classification/cloudy/cloudy_0.jpg`
- Save visualization: `python predict.py --image path/to.jpg --output outputs/prediction.png`
- Force CPU (if no GPU): `--device cpu`

4) Change model/params
- Edit `config.py` (e.g., `MODEL_NAME`, `IMG_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`).

## Quick Start — Vehicle Detection (YOLOv11)

Weights
- Repo includes `yolo11n.pt` (nano). You can swap in other sizes by setting `YOLO_MODEL` in `vehicle_detection_config.py`.

CLI usage (recommended)
- Image: `python demo_vehicle_detection.py --mode image --source test_car.jpg --show`
- Video: `python demo_vehicle_detection.py --mode video --source path/to/video.mp4`
- Webcam: `python demo_vehicle_detection.py --mode webcam`
- Batch folder: `python demo_vehicle_detection.py --mode batch --source path/to/images/`
- Optional: `--output <path>` to save, `--conf 0.30` to tweak threshold.

Programmatic usage
- See `vehicle_detector.py` for `VehicleDetector.detect_image`, `detect_video`, `detect_webcam`, `batch_detect`.

Outputs
- Annotated media saved under `detection_outputs/` (default) and summaries under `vehicle_results/`.

## How It Works (Brief)

- Classification pipeline
  - `data_loader.py`: builds train/val/test splits and DataLoaders with augmentations.
  - `models.py`: returns a transfer‑learning model (`resnet50` default) with final layer sized to 16 classes.
  - `train.py`: trains with Adam, scheduler on val loss, early stopping; persists best/latest checkpoints and history.
  - `evaluate.py`: computes accuracy/precision/recall/F1, confusion matrix, per‑class plots, and exports JSON + PNGs.
  - `predict.py`: loads `models/best_model.pth`, produces a side‑by‑side visualization with top‑k scores.

- Detection pipeline
  - `ultralytics.YOLO` model loads `YOLO_MODEL` from `vehicle_detection_config.py`.
  - Runs on GPU if available, else CPU; filters to vehicle classes only and renders labeled boxes.
  - CLI in `demo_vehicle_detection.py` simplifies running across image/video/webcam/batch modes.

## Tips & Troubleshooting

- No GPU? Add `--device cpu` in `predict.py`; training and YOLO will auto‑fallback to CPU but run slower.
- Missing dataset? Create `data_classification/` with the 16 class folders and add images before training.
- Large artifacts: consider keeping generated items (`results/`, `detection_outputs/`, model weights) out of version control.
- Windows reserved names: avoid creating a file named `nul` in the repo root.

## License / Credits

- Built with PyTorch, TorchVision, and Ultralytics YOLO. Adjust and extend for your coursework or production needs.

