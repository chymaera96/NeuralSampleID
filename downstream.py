import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler

from util import load_augmentation_index, load_config
from modules.transformations import GPUTransformSampleID
from encoder.dgl.graph_encoder import GraphEncoderDGL
from simclr.simclr import SimCLR
from modules.data import NeuralSampleIDDataset
from util import load_config, save_ckp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description='Train Classifier on Pretrained GraphEncoderDGL')
parser.add_argument('--config', default='config/grafp.yaml', type=str, help='Path to config file')
parser.add_argument('--epochs', default=5, type=int, metavar='N', help='Number of total epochs to run')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='Path to latest checkpoint')
parser.add_argument('--ckp', default='test', type=str, help='Checkpoint name')
parser.add_argument('--enc_wts', required=True, type=str, help='Path to pretrained GraphEncoderDGL weights')



class CrossAttentionClassifier(nn.Module):
    def __init__(self, in_dim, num_heads=4, hidden_dim=128, num_nodes=100, pos_embed=True):
        """
        Cross-Attention Classifier with optional spatial positional embeddings.

        Args:
            in_dim: Feature dimension of each node.
            num_heads: Number of attention heads.
            hidden_dim: Hidden layer size.
            num_nodes: Maximum number of nodes (for learnable positional embeddings).
            pos_embed: Boolean flag to enable/disable positional embeddings.
        """
        super().__init__()

        self.pos_embed = pos_embed  # Store flag for later use

        if self.pos_embed:
            # Register positional embedding as a buffer so it moves with .to(device)
            self.register_buffer("positional_embedding", torch.randn(1, num_nodes, in_dim))  

        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_i, x_j):
        """
        x_i, x_j: (B, C, N) graph embeddings
        """

        x_i = x_i.permute(0, 2, 1)  # Convert to (B, N, C)
        x_j = x_j.permute(0, 2, 1)

        # print(f"x_i shape: {x_i.shape}, x_j shape: {x_j.shape}")

        if self.pos_embed:
            pos = self.positional_embedding[:, :x_i.shape[1], :].to(x_i.device)
            # print(f"Positional embedding shape: {pos.shape}")
            x_i = x_i + pos
            x_j = x_j + pos

        attn_out, _ = self.attn(x_i, x_j, x_j)  
        x = attn_out.mean(dim=1)

        return self.fc(x)


def mine_hard_negatives(z_i, z_j, negatives, num_negatives=3):
    """
    Mines `num_negatives` hardest negatives per sample based on cosine similarity.
    Instead of returning the embeddings, returns the indices of hardest negatives.
    """
    B = z_i.shape[0]
    similarity_matrix = torch.matmul(z_i, negatives.T)  # Cosine similarities

    hard_negative_indices = []
    for idx in range(B):
        neg_indices = torch.argsort(similarity_matrix[idx], descending=True)  # Sort by highest similarity
        hard_negative_indices.append(neg_indices[1:num_negatives+1])  # Exclude self, take hardest ones

    return torch.stack(hard_negative_indices)  # (B, num_negatives)

def train(cfg, train_loader, model, classifier, optimizer, scaler, augment=None):
    model.eval()  # Keep the encoder frozen
    classifier.train()
    criterion = nn.BCELoss()
    loss_epoch = 0

    for idx, (x_i, x_j) in enumerate(train_loader):
        x_i, x_j = x_i.to(device), x_j.to(device)
        optimizer.zero_grad()

        # Extract features using frozen encoder
        with torch.no_grad():
            x_i, x_j = augment(x_i, x_j)
            p_i = model.peak_extractor(x_i)
            p_j = model.peak_extractor(x_j)
            x_before_proj_i, _ = model.encoder(p_i, return_pre_proj=True)  # (B, C, N)
            x_before_proj_j, _ = model.encoder(p_j, return_pre_proj=True)  # (B, C, N)
            _, _, z_i, z_j = model(x_i, x_j)  # Projector outputs for mining negatives

        x_before_proj_all = torch.cat((x_before_proj_i, x_before_proj_j), dim=0)    # (2B, C, N)
        z_all = torch.cat((z_i, z_j), dim=0)    # (2B, C)

        # Mine hardest negatives
        hard_negative_idxs = mine_hard_negatives(z_i, z_j, z_all, num_negatives=3)  # (B, C)

        # Use indices to fetch corresponding x_before_proj negatives
        hard_negatives = x_before_proj_all[hard_negative_idxs.view(-1)]  # (B*3, C, N)

        logits_pos = classifier(x_before_proj_i, x_before_proj_j)  # (B, 1)
        logits_neg = classifier(x_before_proj_i.repeat(3, 1, 1), hard_negatives)  # (B*3, 1)
        pos_labels = torch.ones(logits_pos.shape[0], 1).to(device)
        neg_labels = torch.zeros(logits_neg.shape[0], 1).to(device)

        clf_loss = criterion(logits_pos, pos_labels) + criterion(logits_neg, neg_labels)

        scaler.scale(clf_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if idx % 20 == 0:
            print(f"Step [{idx}/{len(train_loader)}]\t Loss: {clf_loss.item()}")

        loss_epoch += clf_loss.item()

    return loss_epoch / len(train_loader)


def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = load_config(args.config)
    wts = args.enc_wts
    
    print("Intializing augmentation pipeline...")
    # ir_train_idx = load_augmentation_index(ir_dir, splits=0.8)["train"]
    gpu_augment = GPUTransformSampleID(cfg=cfg, train=True).to(device)
    cpu_augment = GPUTransformSampleID(cfg=cfg, cpu=True)

    train_dataset = NeuralSampleIDDataset(cfg=cfg, train=True, transform=cpu_augment)
    train_loader = DataLoader(train_dataset, batch_size=cfg['clf_bsz'], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    print("Loading pretrained encoder...")
    model = SimCLR(cfg, encoder=GraphEncoderDGL(cfg=cfg, in_channels=cfg['n_filters'], k=5))

    print("=> loading checkpoint '{}'".format(wts))
    checkpoint = torch.load(wts, map_location=device)    
    if 'module' in list(checkpoint['state_dict'].keys())[0] and torch.cuda.device_count() == 1:
        checkpoint['state_dict'] = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
    
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    for param in model.encoder.parameters():
        param.requires_grad = False  # Keep encoder frozen
    
    classifier = CrossAttentionClassifier(in_dim=512, num_nodes=32).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg['clf_lr'])
    scaler = GradScaler()
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            classifier.load_state_dict(torch.load(args.resume))
        else:
            print(f"=> No checkpoint found at '{args.resume}'")
    
    print("Starting training...")
    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss_epoch = train(cfg, train_loader, model, classifier, optimizer, scaler, augment=gpu_augment)        
        print(f"Epoch {epoch}, Loss: {loss_epoch}")
        torch.save(classifier.state_dict(), f'checkpoint/clf_{args.ckp}_{epoch}.pth')

if __name__ == '__main__':
    main()