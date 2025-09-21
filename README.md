# NeuralSampleID: A Framework for Automatic Sample Identification

**NeuralSampleID** is a lightweight and scalable framework for automatic sample identification (ASID), the task of detecting and retrieving music samples embedded within audio queries. This system:

- Uses a self-supervised Graph Neural Network (GNN) encoder trained with contrastive learning.
- Includes a cross-attention classifier that refines and ranks retrieval results.
- Benchmarks performance using fine-grained annotations from an extended version of the Sample100 dataset.

Our method achieves SOTA with only 9% of the parameters used by prior systems. For more details, please see the preprint and the documentation in this repository.

---

 **Our work has been accepted to ISMIR 2025!**  
Check out the preprint [here](https://arxiv.org/abs/placeholder-link).

This repository contains the official implementation for the paper:

**"Refining Music Sample Identification with a Self-Supervised Graph Neural Network"**  
_A. Bhattacharjee, I. Meresman Higgs, M. Sandler, and E. Benetos_  
_To appear in the Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR), 2025_

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
git clone https://github.com/chymaera96/NeuralSampleID.git
cd NeuralSampleID

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
- [Evaluation database](https://huggingface.co/automatic-sample-id-ismir25/asid-ismir25/blob/main/model_tc_35_best.zip)

---

## Citation

If you use this code or the dataset in your research, please cite our paper:

Bhattacharjee, A., Meresman Higgs, I., Sandler, M., & Benetos, E. (2025). Refining Music Sample Identification with a Self-Supervised Graph Neural Network. In _Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)_. Daejeon, South Korea.

```bibtex
@inproceedings{bhattacharjee2025refining,
  title={Refining Music Sample Identification with a Self-Supervised Graph Neural Network},
  author={Bhattacharjee, Aditya and Meresman Higgs, Ivan and Sandler, Mark and Benetos, Emmanouil},
  booktitle={Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2025},
  address={Daejeon, South Korea},
  publisher={ISMIR},
  note={Preprint available at \url{https://www.arxiv.org/abs/2506.14684}}
}
```

---

For issues or questions, please open an [Issue](https://github.com/chymaera96/NeuralSampleID/issues).
