# NeuralSampleID

NeuralSampleID is a framework for learning contrastive audio fingerprints for audio identification tasks. It uses a SimCLR-style self-supervised pretraining scheme, followed by classifier-based fine-tuning and evaluation.

Pretrained models and extracted fingerprints are hosted on HuggingFace.

---

## Table of Contents
- [Installation](#installation)
- [Pretraining](#pretraining)
- [Training a Classifier](#training-a-classifier)
- [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

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

## Pretraining

The pretraining step uses contrastive learning (NT-Xent Loss) with a Graph Neural Network or ResNet-IBN backbone.

```bash
# Example: Pretrain the model with Graph Encoder
python train.py --config config/grafp.yaml --train_dir path/to/train_data --val_dir path/to/val_data --ckp grafp_pretrain
```

Key arguments:
- `--config`: YAML config file path
- `--train_dir`, `--val_dir`: Paths to training and validation datasets
- `--ckp`: Checkpoint name (saved under `checkpoint/`)

> **Note**: If you want to resume from a checkpoint, use `--resume path/to/checkpoint.pth`.

---

## Training a Classifier

After pretraining, you can fine-tune a classifier on the learned embeddings.

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

The script `ismir25.sh` handles running evaluation with the appropriate model and fingerprint data.

### Example structure for `ismir25.sh`

```bash
#!/bin/bash

MODEL_TYPE=$1

if [ "$MODEL_TYPE" == "baseline" ]; then
    python eval.py --model baseline --fingerprints path/to/baseline/fingerprints
elif [ "$MODEL_TYPE" == "proposed" ]; then
    python eval.py --model proposed --fingerprints path/to/proposed/fingerprints
else
    echo "Unknown model type: $MODEL_TYPE"
fi
```

---

## Pretrained Models

You can download pretrained model weights and extracted fingerprints from [![HuggingFace](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)](https://huggingface.co/chymaera96/NeuralSampleID).

- [GrafPrint pretrained weights](https://huggingface.co/chymaera96/NeuralSampleID/tree/main/grafp-weights)
- [Baseline ResNet-IBN fingerprints](https://huggingface.co/chymaera96/NeuralSampleID/tree/main/baseline-fingerprints)
- [GrafPrint fingerprints](https://huggingface.co/chymaera96/NeuralSampleID/tree/main/grafp-fingerprints)

---

## Citation

If you use NeuralSampleID in your work, please cite:

```bibtex
@inproceedings{your_bibtex_here,
  title={Neural SampleID: Contrastive Audio Fingerprinting},
  author={Your Name},
  booktitle={Proceedings of the 2025 ISMIR Conference},
  year={2025}
}
```

---

## Acknowledgements

- Parts of the code structure and training loop were adapted from SimCLR and MixCo.
- FAISS is used for fast nearest neighbor search during evaluation.

---

For issues or questions, please open an [Issue](https://github.com/chymaera96/NeuralSampleID/issues).