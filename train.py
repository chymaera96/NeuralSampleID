import os
import numpy as np
import argparse
import torch
import gc
import sys
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler
import torchaudio

from util import *
from simclr.ntxent import ntxent_loss
from simclr.simclr import SimCLR   
from modules.transformations import GPUTransformSampleID
from modules.data import NeuralSampleIDDataset
from encoder.dgl.graph_encoder import GraphEncoderDGL
from encoder.resnet_ibn import ResNetIBN

# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root, "checkpoint")
parent_dir = os.path.abspath(os.path.join(root, os.pardir))
sys.path.append(parent_dir)
nan_counter = 0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description='ASID Training')
parser.add_argument('--config', default='config/grafp.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--train_dir', default=None, type=str, metavar='PATH',
                    help='path to training data')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training.')
parser.add_argument('--ckp', default='test', type=str,
                    help='checkpoint_name')
parser.add_argument('--encoder', default='grafp', type=str)
parser.add_argument('--size_opt', default='t', type=str)
parser.add_argument('--k', default=5, type=int)

def train(cfg, train_loader, model, optimizer, scaler, augment=None):
    model.train()
    loss_epoch = 0
    global nan_counter

    for idx, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        with torch.no_grad():
            x_i, x_j = augment(x_i, x_j)

        _, _, z_i, z_j = model(x_i, x_j)

        loss = ntxent_loss(z_i, z_j, cfg)

        if torch.isnan(loss):
            print(f"NaN detected in loss at step {idx}, skipping batch")
            nan_counter = save_nan_batch(x_i, x_j, save_dir="nan_batches", counter=nan_counter)
            continue

        scaler.scale(loss).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(train_loader)}]\t SimCLR Loss: {loss.item()}")

        loss_epoch += loss.item()

    return loss_epoch

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    writer = SummaryWriter(f'runs/{args.ckp}')
    
    # Hyperparameters
    batch_size = cfg['bsz_train']
    learning_rate = cfg['lr']
    num_epochs = override(cfg['n_epochs'], args.epochs)
    model_name = args.ckp
    random_seed = args.seed
    shuffle_dataset = True

    print("Initializing augmentation pipeline...")
    # ir_train_idx = load_augmentation_index(ir_dir, splits=0.8)["train"]
    gpu_augment = GPUTransformSampleID(cfg=cfg, train=True).to(device)
    cpu_augment = GPUTransformSampleID(cfg=cfg, cpu=True)

    print("Loading dataset...")
    train_dataset = NeuralSampleIDDataset(cfg=cfg, train=True, transform=cpu_augment)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    print("Creating new model...")
    if args.encoder == 'grafp':
        model = SimCLR(cfg, encoder=GraphEncoderDGL(cfg=cfg, in_channels=cfg['n_filters'], k=args.k, size=args.size_opt))
    elif args.encoder == 'resnet-ibn':
        model = SimCLR(cfg, encoder=ResNetIBN())
    else:
        raise NotImplementedError(f"Encoder {args.encoder} is not supported")
        
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(device)
        
    print(count_parameters(model, args.encoder))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['T_max'], eta_min=cfg['min_lr'])
    scaler = DummyScaler()
       
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            model, optimizer, scheduler, start_epoch, loss_log = load_ckp(args.resume, model, optimizer, scheduler)
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    else:
        start_epoch = 0
        loss_log = []

    print("Calculating initial loss...")
    best_loss = float('inf')

    # Training loop
    for epoch in range(start_epoch + 1, num_epochs + 1):
        print(f"####### Epoch {epoch} #######")
        loss_epoch = train(cfg, train_loader, model, optimizer, scaler, gpu_augment)
        writer.add_scalar("Loss/train", loss_epoch, epoch)
        loss_log.append(loss_epoch)

        checkpoint = {
            'epoch': epoch,
            'loss': loss_log,
            'hit_rate_log': [],  # Save an empty hit_rate_log for backward compatibility
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_ckp(checkpoint, model_name, model_folder, 'current')
        assert os.path.exists(f'checkpoint/model_{model_name}_current.pth'), "Checkpoint not saved"

        if loss_epoch < best_loss:
            best_loss = loss_epoch
            save_ckp(checkpoint, model_name, model_folder, 'best')

        if epoch % 10 == 0 or epoch in [105, 108, 113, 115, 118]:
            save_ckp(checkpoint, model_name, model_folder, epoch)

        scheduler.step()

if __name__ == '__main__':
    main()