#!/bin/bash

# Usage: ./ismir25.sh [baseline|proposed]

set -e

MODEL_TYPE=$1

if [ -z "$MODEL_TYPE" ]; then
    echo "Usage: $0 [baseline|proposed]"
    exit 1
fi

mkdir -p checkpoint

# Download model based on selection
if [ "$MODEL_TYPE" = "baseline" ]; then
    MODEL_URL="https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/model_tc_39_100.pth"
    MODEL_PATH="checkpoint/baseline_model.pth"
elif [ "$MODEL_TYPE" = "proposed" ]; then
    MODEL_URL="https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/model_tc_35_best.pth"
    MODEL_PATH="checkpoint/model_tc_35_best.pth"
    CLF_URL_URL="https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/clf_tc_35_4.pth"
    CLF_PATH="checkpoint/clf_tc_35_4.pth"
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

# Download classifier model if proposed model is selected
if [ "$MODEL_TYPE" = "proposed" ] && [ ! -f "$CLF_PATH" ]; then
    echo "Downloading classifier model..."
    wget -O "$CLF_PATH" "$CLF_URL"
else
    echo "Classifier model already exists. Skipping download."
fi

# Download and unzip fingerprint directory
FP_DIR="log/emb/test"
mkdir -p "$FP_DIR"
if [ "$MODEL_TYPE" = "proposed" ]; then
    FP_URL="https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/model_tc_35_best.zip"
    FP_ZIP_PATH="data/model_tc_35_best.zip"
    FP_OUT_PATH="$FP_DIR/model_tc_35_best"
else
    FP_URL="https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/model_tc_39_100.zip"
    FP_ZIP_PATH="data/model_tc_39_100.zip"
    FP_OUT_PATH="$FP_DIR/model_tc_39_100"
fi


if [ ! -d "$FP_OUT_PATH" ]; then
    echo "Downloading fingerprints..."
    wget -O "$FP_ZIP_PATH" "$FP_URL"
    echo "Unzipping fingerprints..."
    unzip "$FP_ZIP_PATH" -d "$FP_DIR"
    rm "$FP_ZIP_PATH"
else
    echo "Fingerprints already exist. Skipping download."
fi

# Run test_fp.py
echo "Running sample retrieval evaluation..."
# If model is proposed
if [ "$MODEL_TYPE" = "proposed" ]; then
    python test_fp.py --query_lens=5,7,10,15,20 \
                  --text=tc35_reprod \
                  --test_dir=data \
                  --map \
                  --clf_ckp=clf_tc_35_4.pth \
                  --test_config='{"tc_35:"best"}' \
                  --ismir25
else
    cd baseline
    python run_eval.py  --query_lens=5,7,10,15,20 \
                        --text=tc39_reprod \
                        --test_dir=../data \
                        --map \
                        --test_config='{"tc_39":100}' \
                        --ismir25
    
fi

