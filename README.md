# NeuralSampleID

Official repository for "REFINING MUSIC SAMPLE IDENTIFICATION WITH A SELF-SUPERVISED GRAPH NEURAL NETWORK" currently under review at Internation Society for Music Information Retrieval (ISMIR), 2025.

---

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Pretraining](#pretraining)
- [Classifier Training](#classifier-training)
- [Evaluation](#evaluation)
- [Pretrained Models and Fingerprints](#pretrained-models)
- [Citation](#citation)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/automatic-sample-id-ismir25/asid-ismir25.git
cd asid-ismir25

# Install dependencies
pip install -r requirements.txt

# Install FAISS GPU for faster evaluation
conda install faiss-gpu -c pytorch

# Install DGL (cuda version specific)
conda install -c dglteam/label/th24_cu121 dgl
```

---

## Dataset Preparation

The models are trained using the `fma_medium` subset of the Free Music Archive (FMA) dataset. The audio files are first preprocessed into source-separated stems; specifically `vocal`,`drum`,`bass`,`other` stems. For this, we use HTDemucs \[cite\]. For the training setup, the source separated audio files should follow the following directory structure.

```
htdemucs/
├── 12345/
│   ├── vocals.mp3
│   ├── drums.mp3
│   ├── bass.mp3
│   └── other.mp3
├── 12346/
│   ├── vocals.mp3
│   ├── drums.mp3
│   ├── bass.mp3
│   └── other.mp3
├── ...
```

Each subfolder (e.g., `12345`) corresponds to a unique FMA track ID and contains the separated stem files in `.mp3` format.

We use the our extended annotations of the Sample100 dataset -- `sample100-ext` for retrieval evaluation. Details of the dataset can be found in the dataset [README](https://github.com/automatic-sample-id-ismir25/asid-ismir25/blob/main/sample100-ext/README.md). Evaluation audio files have not been shared as a part of this work. Instead, we provide the fingeprints computed using our setup for queries and reference database. 


## Pretraining

The pretraining step uses contrastive learning of the Graph Neural Network backbone. 

```bash
# Pre-training the proposed model
python train.py --config config/grafp.yaml --ckp CKP_NAME
# (Single-stage) training of the baseline mode
cd baseline
python train.py --config config/resnet_ibn.yaml --ckp CKP_NAME
```

Key arguments:
- `--config`: YAML config file path
- `--ckp`: Placeholder name for the training run

> **Note**:  Update the paths (particularly, `htdemucs_dir` and `fma_dir`) in the YAML file to point at the directory containing the source-separated and mixed audio data for training. If you want to resume from a checkpoint, use `--resume path/to/checkpoint.pth`.

---

## Classifier Training

After pretraining, you can fine-tune the MHCA classifier on the learned embeddings (fingerprints).

```bash
python downstream.py --enc_wts ENCODER_CHECKPOINT
```
---

## Evaluation

Given a query set, the evaluation process compares the retrieval rates and mean average precision (mAP). 

```bash
# Usage: ./ismir25.sh [baseline|proposed]

# Example: Evaluate the proposed model
bash ismir25.sh proposed

# Example: Evaluate the baseline model
bash ismir25.sh baseline
```

The script `ismir25.sh` handles running evaluation with the appropriate model to reproduce published benchmarks. A detailed demonstration of evaluation on custom datasets will be updated soon!


## Pretrained Models and Fingeprints

 [![HuggingFace](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)](https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25)
- [GNN pretrained weights](https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/model_tc_35_best.pth)
- [MHCA classifier weights](https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/clf_tc_35_4.pth)
- [Baseline ResNet-IBN weights](https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/model_tc_39_100.pth)
- [Evaluation database](https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/proposed_db.zip)

---

## Citation

TBD

---

For issues or questions, please open an [Issue](https://github.com/chymaera96/NeuralSampleID/issues).