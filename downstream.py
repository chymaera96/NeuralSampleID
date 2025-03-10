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

from util import load_augmentation_index
from modules.transformations import GPUTransformSampleID
from encoder.dgl.graph_encoder import GraphEncoderDGL
from simclr.simclr import SimCLR
from modules.data import NeuralSampleIDDataset
from util import load_config, save_ckp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description='Train Classifier on Pretrained GraphEncoderDGL')
parser.add_argument('--config', default='config/grafp.yaml', type=str, help='Path to config file')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='Number of total epochs to run')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='Path to latest checkpoint')
parser.add_argument('--ckp', default='test', type=str, help='Checkpoint name')
parser.add_argument('--enc_wts', required=True, type=str, help='Path to pretrained GraphEncoderDGL weights')

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def train(cfg, train_loader, encoder, classifier, optimizer, scaler):
    encoder.eval()  # Keep the encoder frozen
    classifier.train()
    loss_epoch = 0
    criterion = nn.BCELoss()

    for x, labels in train_loader:
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            x_before_proj, _ = encoder(x, return_pre_proj=True)

        preds = classifier(x_before_proj)
        loss = criterion(preds, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_epoch += loss.item()
    
    return loss_epoch / len(train_loader)

def main():
    args = parser.parse_args()
    noise_dir = args.noise_dir
    ir_dir = args.ir_dir
    cfg = load_config(args.config)
    wts = args.enc_wts
    
    print("Intializing augmentation pipeline...")
    noise_train_idx = load_augmentation_index(noise_dir, splits=0.8)["train"]
    ir_train_idx = load_augmentation_index(ir_dir, splits=0.8)["train"]

    gpu_augment = GPUTransformSampleID(cfg=cfg, ir_dir=ir_train_idx, noise_dir=noise_train_idx, train=True).to(device)
    cpu_augment = GPUTransformSampleID(cfg=cfg, ir_dir=ir_train_idx, noise_dir=noise_train_idx, cpu=True)
    train_dataset = NeuralSampleIDDataset(cfg=cfg, train=True, transform=cpu_augment)
    train_loader = DataLoader(train_dataset, batch_size=cfg['clf_bsz'], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    print("Loading pretrained encoder...")
    model = SimCLR(cfg, encoder=GraphEncoderDGL(cfg=cfg, in_channels=cfg['n_filters'], k=cfg['k']))

    print("=> loading checkpoint '{}'".format(wts))
    checkpoint = torch.load(wts, map_location=device)    
    if 'module' in list(checkpoint['state_dict'].keys())[0] and torch.cuda.device_count() == 1:
        checkpoint['state_dict'] = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
    
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    encoder = model.encoder
    
    for param in encoder.parameters():
        param.requires_grad = False  # Keep encoder frozen
    
    classifier = GraphClassifier(in_dim=encoder.channels[-1], hidden_dim=256).to(device)
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
        loss_epoch = train(cfg, train_loader, encoder, classifier, optimizer, scaler)        
        print(f"Epoch {epoch}, Loss: {loss_epoch}")
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            torch.save(classifier.state_dict(), f'checkpoint/clf_{args.ckp}_best.pth')
        elif epoch % 10 == 0:
            torch.save(classifier.state_dict(), f'checkpoint/clf_{args.ckp}_{epoch}.pth')

if __name__ == '__main__':
    main()