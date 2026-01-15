mkdir -p /mnt/workspace/xuliang/YOLO-SERIES/yolov8/datasets
nano /mnt/workspace/xuliang/YOLO-SERIES/yolov8/datasets/3d_dataset.yaml#!/bin/bash

# Validate workspace structure before execution
if [ ! -f "/mnt/workspace/xuliang/YOLO-SERIES/yolov8/ultralytics/models/yolo/pose/train3D.py" ]; then
    echo "ERROR: train3D.py not found. Please restore using Lingma." >&2
    exit 1
fi

# Check for virtual environment
VENV_PATH="/mnt/workspace/xuliang/YOLO-SERIES/yolov8/.venv"
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    echo "WARNING: Virtual environment not found at $VENV_PATH"
    echo "Consider creating one with: python -m venv $VENV_PATH"
fi

# Verify dataset configuration
DATASET_CONFIG="/mnt/workspace/xuliang/YOLO-SERIES/yolov8/datasets/3d_dataset.yaml"
if [ ! -f "$DATASET_CONFIG" ]; then
    echo "ERROR: Dataset configuration not found at $DATASET_CONFIG"
    echo "Please create a 3D dataset YAML with RGB-D paths"
    echo "Example content:"
    echo "path: /path/to/3d_dataset"
    echo "train: images/train (must contain 4-channel PNGs)"
    echo "val: images/val"
    echo "nc: 80"
    echo "names: [...]"
    exit 1
fi

# Execute 3D training with validation
echo "Starting 3D object detection training..."
yolo train \
    model=yolov8n.yaml \
    data=$DATASET_CONFIG \
    task=detect3d \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0

# Verify output structure
if [ -d "runs/detect3d/train" ]; then
    echo "\nTraining completed successfully!"
    echo "Results saved in: $(pwd)/runs/detect3d/train"
else
    echo "ERROR: Training output directory not created" >&2
    exit 1
fi