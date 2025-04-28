import os
import numpy as np
import argparse
import torch
import sys
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
# torchaudio.set_audio_backend("soundfile")


from util import *
from simclr.triplet import triplet_loss, classifier_loss, BaselineModel

from modules.transformations import GPUTransformSampleID
from modules.data import NeuralSampleIDDataset
from encoder.resnet_ibn import ResNetIBN


# Directories
root = os.path.dirname(__file__)
model_folder = os.path.abspath(os.path.join(root, os.pardir, "checkpoint"))
parent_dir = os.path.abspath(os.path.join(root, os.pardir))
sys.path.append(parent_dir)
nan_counter = 0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


parser = argparse.ArgumentParser(description='Baseline Training')
parser.add_argument('--config', default='config/resnet_ibn.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--train_dir', default=None, type=str, metavar='PATH',
                    help='path to training data')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--ckp', default='test', type=str,
                    help='checkpoint_name')
parser.add_argument('--encoder', default='resnet-ibn', type=str)
parser.add_argument('--n_dummy_db', default=None, type=int)
parser.add_argument('--n_query_db', default=None, type=int)




def train(cfg, train_loader, model, optimizer, scaler, ir_idx, noise_idx, augment=None):
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

        # Combine both views and normalize
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1, p=2)
        B = z_i.size(0)
        labels = torch.cat([torch.arange(B), torch.arange(B)], dim=0).to(z.device)

        loss_cls = classifier_loss(z_i, z_j)
        # loss_cls = torch.tensor(0.0)
        loss_trip = triplet_loss(z, labels, margin=cfg['margin'])
        # loss_trip = torch.tensor(0.0)

        # Final loss
        loss = cfg['beta'] * loss_cls + cfg['gamma'] * loss_trip

        assert not torch.isnan(loss), "Loss is NaN"

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(train_loader)}] | "
                  f"Cls: {loss_cls.item():.4f} | Triplet: {loss_trip.item():.4f}")

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
    
    valid_dataset = NeuralSampleIDDataset(cfg=cfg, train=False)
    print("Creating validation dataloaders...")
    dataset_size = len(valid_dataset)
    indices = list(range(dataset_size))
    split1 = override(cfg['n_dummy'],args.n_dummy_db)
    split2 = override(cfg['n_query'],args.n_query_db)
    
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    dummy_indices, query_db_indices = indices[:split1], indices[split1: split1 + split2]

    dummy_db_sampler = SubsetRandomSampler(dummy_indices)
    query_db_sampler = SubsetRandomSampler(query_db_indices)

    dummy_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=1, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=dummy_db_sampler)
    
    query_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=1, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=query_db_sampler)
    

    print("Creating new model...")
    model = BaselineModel(cfg, encoder=ResNetIBN())
        
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = DataParallel(model).to(device)
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(device)
        
    print(count_parameters(model, args.encoder))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
    # scaler = GradScaler(enabled=True)
    scaler = DummyScaler()
       
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model, optimizer, scheduler, start_epoch, loss_log, hit_rate_log = load_ckp(args.resume, model, optimizer, scheduler)
            output_root_dir = create_fp_dir(resume=args.resume)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    else:
        start_epoch = 0
        loss_log = []
        hit_rate_log = []

    print("Calculating initial loss ...")
    best_loss = float('inf')
    # training

    for epoch in range(start_epoch+1, num_epochs+1):
        print("#######Epoch {}#######".format(epoch))
        loss_epoch = train(cfg, train_loader, model, optimizer, scaler, ir_train_idx, noise_train_idx, gpu_augment)
        writer.add_scalar("Loss/train", loss_epoch, epoch)
        loss_log.append(loss_epoch)

        checkpoint = {
            'epoch': epoch,
            'loss': loss_log,
            'valid_acc' : hit_rate_log,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        save_ckp(checkpoint, model_name, model_folder, 'current')
        assert os.path.exists(f'checkpoint/model_{model_name}_current.pth'), "Checkpoint not saved"

        if loss_epoch < best_loss:
            best_loss = loss_epoch
            save_ckp(checkpoint, model_name, model_folder, 'best')

        if epoch % 10 == 0:
            save_ckp(checkpoint, model_name, model_folder, epoch)
            
        scheduler.step()
    
  
        
if __name__ == '__main__':
    main()