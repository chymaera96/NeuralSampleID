import os
import json
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Prevent crashes on headless servers
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from modules.data import Sample100Dataset
from modules.transformations import GPUTransformSampleID
from simclr.simclr import SimCLR
from encoder.dgl.graph_encoder import GraphEncoderDGL
from downstream import CrossAttentionClassifier
from util import load_config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def collect_scores(model, ref_dir, classifier, dataloader, transform, n_samples):
    scores = []
    for i, (nm, audio) in enumerate(dataloader):
        nm = nm[0]
        if i >= n_samples:
            break
        
        audio = audio.to(device)
        x, _ = transform(audio, None)
        r = random.randint(0, x.size(0) - 1)
        x = x[r].unsqueeze(0)

        p = model.peak_extractor(x)
        nm_q, _ = model.encoder(p, return_pre_proj=True)

        ref_path = os.path.join(ref_dir, f"{nm.split('_')[0]}.npy")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(ref_dir, random.choice(os.listdir(ref_dir)))
        nm_r = torch.tensor(np.load(ref_path)).to(device)
        nm_q = nm_q.repeat(nm_r.size(0), 1, 1)

        logits = classifier(nm_q, nm_r).max().item()

        scores.append(logits)

        if i % 10 == 0:
            print(f"Processed {i}/{n_samples} samples...")
        
    return scores



def compute_rejection_stats(real_scores, dummy_scores, threshold=0.5, save_path=None):
    scores = np.array(real_scores + dummy_scores)
    labels = np.array([1] * len(real_scores) + [0] * len(dummy_scores))

    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    true_rejects = sum(s < threshold for s in dummy_scores)
    false_accepts = sum(s >= threshold for s in dummy_scores)
    total_dummies = len(dummy_scores)

    print(f"\nAUROC: {auroc:.4f}")
    print(f"True Positive Rejects: {true_rejects}/{total_dummies} ({true_rejects / total_dummies:.2%})")
    print(f"False Accept Rate: {false_accepts}/{total_dummies} ({false_accepts / total_dummies:.2%})")

    if save_path:
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Classifier ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")

    return auroc, true_rejects, false_accepts


def main():
    parser = argparse.ArgumentParser(description="Classifier Rejection Evaluation via AUC")
    parser.add_argument('--test_dir', default='../datasets/sample_100/audio', type=str)
    parser.add_argument('--ref_dir', 
                        default='/data/scratch/acw723/logs/emb/valid/model_tc_35_best/ref_nmatrix', 
                        type=str)
    parser.add_argument('--config', default='config/grafp.yaml', type=str)
    parser.add_argument('--clf_ckp', default='checkpoint/clf_tc_35_4.pth', type=str)
    parser.add_argument('--enc_ckp', default='checkpoint/model_tc_35_best.pth', type=str)
    parser.add_argument('--samples', default=100, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--save_plot', default=None, type=str)
    args = parser.parse_args()

    # Load config and set up environment
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Initializing model and classifier...")
    model = SimCLR(cfg, encoder=GraphEncoderDGL(
        cfg=cfg, in_channels=cfg['n_filters'], k=5, size='t')).to(device)
    
    if os.path.isfile(args.enc_ckp):
        ckp = args.enc_ckp
        print("=> loading checkpoint '{}'".format(ckp))
        checkpoint = torch.load(ckp)
        # Check for DataParallel
        if 'module' in list(checkpoint['state_dict'].keys())[0] and torch.cuda.device_count() == 1:
            checkpoint['state_dict'] = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(ckp))
        raise FileNotFoundError
    
    model.eval()

    classifier = CrossAttentionClassifier(in_dim=512, num_nodes=32).to(device)
    classifier.load_state_dict(torch.load(args.clf_ckp, map_location=device))
    classifier.eval()

    transform = GPUTransformSampleID(cfg=cfg, ir_dir=None, noise_dir=None, train=False).to(device)

    print("Preparing datasets...")
    query_dataset = Sample100Dataset(cfg, path=args.test_dir,
                                     annot_path=cfg['annot_path'], mode="query")
    dummy_dataset = Sample100Dataset(cfg, path='data/sample_100.json',
                                     annot_path=cfg['annot_path'], mode="dummy")

    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=1, shuffle=True)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1, shuffle=True)

    print(f"Sampling {args.samples} query-reference and dummy-reference pairs...")

    ref_dir = args.ref_dir
    real_scores = collect_scores(model, ref_dir, classifier, query_loader, transform, args.samples)
    dummy_scores = collect_scores(model, ref_dir, classifier, dummy_loader, transform, args.samples)

    compute_rejection_stats(real_scores, dummy_scores, threshold=args.threshold, save_path=args.save_plot)


if __name__ == "__main__":
    main()
