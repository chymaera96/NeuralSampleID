# NeuralSampleID

Official repository for "REFINING MUSIC SAMPLE IDENTIFICATION WITH A SELF-SUPERVISED GRAPH NEURAL NETWORK" currently under review at Internation Society for Music Information Retrieval (ISMIR), 2025.

---

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation] (#dataset-preparation)
- [Pretraining](#pretraining)
- [Classifier Training](#classifier-training)
- [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/chymaera96/NeuralSampleID.git
cd NeuralSampleID

# Install dependencies
pip install -r requirements.txt

# (Optional) Install FAISS GPU for faster evaluation
conda install faiss-gpu -c pytorch
```

---

## Dataset Preparation

The models are trained using the `fma_medium` subset of the Free Music Archive (FMA) dataset. The audio files are first preprocessed into source-separated stems; specifically `vocal`,`drum`,`bass`,`other` stems. For this, we use HTDemucs \[cite\]. For the training setup, the source separated audio files should follow the following directory structure.

htdemucs/ ├── 12345/ │ ├── vocals.mp3 │ ├── drums.mp3 │ ├── bass.mp3 │ └── other.mp3 ├── 12346/ │ ├── vocals.mp3 │ ├── drums.mp3 │ ├── bass.mp3 │ └── other.mp3 ├── ...


Each subfolder (e.g., `12345`) corresponds to a unique FMA track ID and contains the separated stem files in `.mp3` format.


## Pretraining

The pretraining step uses contrastive learning of the Graph Neural Network backbone. 

```bash
python train.py --config config/grafp.yaml --ckp CKP_NAME
```

Key arguments:
- `--config`: YAML config file path
- `--ckp`: Placeholder name for the training run

> **Note**:  Update the paths (particularly, `htdemucs_dir` and `fma_dir`) in the YAML file to point at the directory containing the source-separated audio data for training. If you want to resume from a checkpoint, use `--resume path/to/checkpoint.pth`.

---

## Classifier Training

After pretraining, you can fine-tune the MHCA classifier on the learned embeddings (fingerprints).

```bash
# (Coming soon) Fine-tuning script
```

For now, classification can be incorporated by evaluating extracted fingerprints.

---

## Evaluation

Evaluation compares query fingerprints against a database using FAISS. To reproduce evaluation:

```bash
# Usage: ./ismir25.sh [baseline|proposed]

# Example: Evaluate the proposed model
bash ismir25.sh proposed

# Example: Evaluate the baseline model
bash ismir25.sh baseline
```

The script `ismir25.sh` handles running evaluation with the appropriate model to reproduce published benchmarks.


## Pretrained Models

 [![HuggingFace](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)](https://huggingface.co/chymaera96/NeuralSampleID)

- [GrafPrint pretrained weights](https://huggingface.co/chymaera96/NeuralSampleID/tree/main/grafp-weights)
- [Baseline ResNet-IBN fingerprints](https://huggingface.co/chymaera96/NeuralSampleID/tree/main/baseline-fingerprints)
- [GrafPrint fingerprints](https://huggingface.co/chymaera96/NeuralSampleID/tree/main/grafp-fingerprints)

---

## Citation

TBD

---

For issues or questions, please open an [Issue](https://github.com/chymaera96/NeuralSampleID/issues).