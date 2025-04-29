#!/bin/bash

# Usage: ./ismir25.sh [baseline|proposed]

set -e

MODEL_TYPE=$1

if [ -z "$MODEL_TYPE" ]; then
    echo "Usage: $0 [baseline|proposed]"
    exit 1
fi

mkdir -p checkpoints

# Download model based on selection
if [ "$MODEL_TYPE" = "baseline" ]; then
    MODEL_URL="<BASELINE_MODEL_URL>"
    MODEL_PATH="checkpoints/baseline_model.pth"
elif [ "$MODEL_TYPE" = "proposed" ]; then
    MODEL_URL="<PROPOSED_MODEL_URL>"
    MODEL_PATH="checkpoints/proposed_model.pth"
else
    echo "Invalid model type. Choose 'baseline' or 'proposed'."
    exit 1
fi

# Download model if not already present
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading $MODEL_TYPE model..."
    wget -O "$MODEL_PATH" "$MODEL_URL"
else
    echo "$MODEL_TYPE model already exists. Skipping download."
fi

# Download and unzip fingerprint directory
FP_DIR="<YOUR_FP_DIR_PATH>"
mkdir -p "$FP_DIR"
FP_URL="<FINGERPRINT_ZIP_URL>"
FP_ZIP_PATH="fp_dir/fingerprints.zip"

if [ ! -d "$FP_DIR/fingerprints" ]; then
    echo "Downloading fingerprints..."
    wget -O "$FP_ZIP_PATH" "$FP_URL"
    echo "Unzipping fingerprints..."
    unzip "$FP_ZIP_PATH" -d "$FP_DIR"
    rm "$FP_ZIP_PATH"
else
    echo "Fingerprints already exist. Skipping download."
fi

# Run test_fp.py
echo "Running fingerprint retrieval test..."
python test_fp.py --model "$MODEL_PATH" --fp_dir "$FP_DIR"
