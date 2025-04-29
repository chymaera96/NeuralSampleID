import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import faiss
import json
import shutil
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
import torchaudio
import traceback
torchaudio.set_audio_backend("soundfile")



from util import \
create_fp_dir, load_config, \
query_len_from_seconds, seconds_from_query_len, \
load_augmentation_index
from modules.data import Sample100Dataset
from encoder.dgl.graph_encoder import GraphEncoderDGL
from encoder.resnet_ibn import ResNetIBN
from simclr.simclr import SimCLR   
from modules.transformations import GPUTransformSampleID
from downstream import CrossAttentionClassifier
from eval import get_index, load_memmap_data, eval_faiss
from eval_map import eval_faiss_map_clf
from eval_hr import eval_faiss_clf


# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")

parser = argparse.ArgumentParser(description='Neuralfp Testing')
parser.add_argument('--config', default='config/grafp.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--test_config', default='config/test_config.yaml')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing testing. ')
parser.add_argument('--test_dir', default='data/fma_medium.json', type=str,
                    help='path to test data')
parser.add_argument('--noise_idx', default=None, type=str)
parser.add_argument('--noise_split', default='all', type=str,
                    help='Noise index file split to use for testing (all, test)')
parser.add_argument('--fp_dir', default='fingerprints', type=str)
parser.add_argument('--query_lens', default=None, type=str)
parser.add_argument('--encoder', default='grafp_t', type=str)
parser.add_argument('--n_dummy_db', default=None, type=int)
# parser.add_argument('--n_query_db', default=350, type=int)
parser.add_argument('--small_test', action='store_true', default=False)
parser.add_argument('--text', default='test', type=str)
# parser.add_argument('--test_snr', default=None, type=int)
parser.add_argument('--recompute', action='store_true', default=False)
parser.add_argument('--map', action='store_true', default=False)
# parser.add_argument('--hit_rate', action='store_true', default=True)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--test_ids', default='1000', type=str)
parser.add_argument('--clf_ckp', default='clf_test_best.pth', type=str)
parser.add_argument('--ismir25', action='store_true', default=False)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')



def create_table(hit_rates, overlap, dur, test_seq_len=[1,3,5,9,11,19], text="test"):
    table = f'''<table>
    <tr>
    <th colspan="5"{text}</th>
    <th>Query Length</th>
    <th>Top-1 Exact</th>
    <th>Top-3 Exact</th>
    <th>Top-10 Exact</th>
    </tr>
    '''
    for idx, q_len in enumerate(test_seq_len):
        table += f'''
        <tr>
        <td>{seconds_from_query_len(q_len, overlap, dur)}</td>
        <td>{hit_rates[0][idx]}</td>
        <td>{hit_rates[1][idx]}</td>
        <td>{hit_rates[2][idx]}</td>
        </tr>
        '''
    table += '</table>'
    return table

def create_query_db(dataloader, augment, model, output_root_dir, fname='query_db', verbose=False, max_size=128):
    fp = []
    lookup_table = []  # Initialize lookup table
    print("=> Creating query fingerprints...")
    for idx, (nm,audio) in enumerate(dataloader):
        nm = nm[0] # Extract filename from list
        audio = audio.to(device)
        x_i, _ = augment(audio, None)
        x_list = torch.split(x_i, max_size, dim=0)
        fp_size = 0
        for x in x_list:
            with torch.no_grad():
                _, _, z_i, _= model(x.to(device),x.to(device))  

            fp.append(z_i.detach().cpu().numpy())
            fp_size += z_i.shape[0]

        # Append song number to lookup table for each segment in the batch
        if fname == 'query_db':
            lookup_table.extend([nm + "_" + str(idx)] * x_i.shape[0])
            log_mod = 100
        else:
            lookup_table.extend([nm] * x_i.shape[0])
            log_mod = 20

        if verbose and idx % log_mod == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: [{fp_size,z_i.shape[1]}]")

    fp = np.concatenate(fp)
    arr_shape = (len(fp), z_i.shape[-1])

    arr = np.memmap(f'{output_root_dir}/{fname}.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr[:] = fp[:]
    arr.flush(); del(arr)   #Close memmap

    np.save(f'{output_root_dir}/{fname}_shape.npy', arr_shape)

    # Save lookup table
    json.dump(lookup_table, open(f'{output_root_dir}/{fname}_lookup.json', 'w'))

def create_ref_db(dataloader, augment, model, output_root_dir, fname='ref_db', verbose=False, max_size=128):
    fp = []
    lookup_table = []  # Initialize lookup table
    print("=> Creating reference fingerprints...")
    for idx, (nm,audio) in enumerate(dataloader):
        nm = nm[0] # Extract filename from list
        audio = audio.to(device)
        x_i, _ = augment(audio, None)
        x_list = torch.split(x_i, max_size, dim=0)
        fp_size = 0
        for x in x_list:
            with torch.no_grad():
                _, _, z_i, _= model(x.to(device),x.to(device))  

            fp.append(z_i.detach().cpu().numpy())
            fp_size += z_i.shape[0]

        # Append song number to lookup table for each segment in the batch
        lookup_table.extend([nm] * x_i.shape[0])

        if verbose and idx % 20 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {fp_size}")

    fp = np.concatenate(fp)
    arr_shape = (len(fp), z_i.shape[-1])

    arr = np.memmap(f'{output_root_dir}/{fname}.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr[:] = fp[:]
    arr.flush(); del(arr)   #Close memmap

    np.save(f'{output_root_dir}/{fname}_shape.npy', arr_shape)

    # Save lookup table
    json.dump(lookup_table, open(f'{output_root_dir}/{fname}_lookup.json', 'w'))


def create_dummy_db(dataloader, augment, model, output_root_dir, fname='dummy_db', verbose=False, max_size=128):
    fp = []
    print("=> Creating dummy fingerprints...")
    for idx, (nm,audio) in enumerate(dataloader):
        audio = audio.to(device)
        x_i, _ = augment(audio, None)
        # print(f"Shape of x_i (dummy): {x_i.shape}")
        x_list = torch.split(x_i, max_size, dim=0)
        # print(f"Number of splits: {len(x_list)}")
        fp_size = 0
        for x in x_list:
            try:
                with torch.no_grad():
                    _, _, z_i, _= model(x.to(device),x.to(device)) 

            except Exception as e:
                print(f"Error in model forward pass in file {nm}")
                print(f"Shape of x_i (dummy): {x.shape}")
                print(f"x_i mean: {x.mean()}, x_i std: {x.std()}")
                print(f"All x shapes in list: {[x_.shape for x_ in x_list]}")
                print(f"Index of data {idx}")
                # print(traceback.format_exc())
                # print(f"Shape of z_i (dummy): {z_i.shape}")
                continue 

            fp.append(z_i.detach().cpu().numpy())
            fp_size += z_i.shape[0]
        
        if verbose and idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {fp_size}")
        # fp = torch.cat(fp)
    
    fp = np.concatenate(fp)
    arr_shape = (len(fp), z_i.shape[-1])

    arr = np.memmap(f'{output_root_dir}/{fname}.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr[:] = fp[:]
    arr.flush(); del(arr)   #Close memmap

    np.save(f'{output_root_dir}/{fname}_shape.npy', arr_shape)


def create_ref_nmatrix(dataloader, augment, model, save_dir, max_size=512, verbose=False):

    os.makedirs(save_dir, exist_ok=True)

    ref_nmatrix = {}  
    print("=> Creating reference node matrices...")
    with torch.no_grad():
        for idx, (nm, audio) in enumerate(dataloader):
            nm = nm[0]  # Extract filename from list
            audio = audio.to(device)
            x_i, _ = augment(audio, None)
            x_list = torch.split(x_i, max_size, dim=0)
            nmat = []
            fp_size = 0
            for x in x_list:
                p = model.peak_extractor(x)
                x_before_proj, _ = model.encoder(p, return_pre_proj=True)  # (B, C, N)
                x_before_proj = x_before_proj.cpu().numpy()
                nmat.append(x_before_proj)
                fp_size += x_before_proj.shape[0]

            ref_nmatrix[nm] = np.concatenate(nmat)
            if verbose and idx % 20 == 0:
                print(f"Step [{idx}/{len(dataloader)}]\t shape: {ref_nmatrix[nm].shape}")

    for song_id, nmatrices in ref_nmatrix.items():
        save_path = os.path.join(save_dir, f"{song_id}.npy")
        np.save(save_path, nmatrices)  # Save as (num_segments, C, N)

    print(f"Saved node matrices for {len(ref_nmatrix)} songs in {save_dir}")



def create_query_nmatrix(dataloader, augment, model, save_path, max_size=512, verbose=False):

    query_nmatrix = {}
    print("=> Creating query node matrices...")
    with torch.no_grad():
        for idx, (nm, audio) in enumerate(dataloader):
            nm = nm[0]  # Extract filename from list
            audio = audio.to(device)
            x_i, _ = augment(audio, None)
            x_list = torch.split(x_i, max_size, dim=0)  # Prevent OOM errors
            nmat = []
            fp_size = 0
            for x in x_list:
                p = model.peak_extractor(x)
                x_before_proj, _ = model.encoder(p, return_pre_proj=True)  # (B, C, N)
                x_before_proj = x_before_proj.cpu().numpy()
                nmat.append(x_before_proj)
                fp_size += x_before_proj.shape[0]

            query_nmatrix[nm] = np.concatenate(nmat)
            if verbose and idx % 20 == 0:
                print(f"Step [{idx}/{len(dataloader)}]\t shape: {query_nmatrix[nm].shape}")

    np.save(save_path, query_nmatrix)
    print(f"Saved node matrices for {len(query_nmatrix)} queries in {save_path}")



def main():

    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.test_config.endswith('.yaml'):
        test_cfg = load_config(args.test_config)
    elif args.test_config.startswith('{'):
        test_cfg = json.loads(args.test_config)
    else:
        raise ValueError("Invalid test_config format. Must be a YAML file or JSON string.")
    annot_path = cfg['annot_path']
    enc = args.encoder.split('_')[0]
    size = args.encoder.split('_')[1]

    # Hyperparameters
    random_seed = 42
    shuffle_dataset =True


    print("Creating new model...")
    if enc == 'grafp':
        model = SimCLR(cfg, encoder=GraphEncoderDGL(cfg=cfg, in_channels=cfg['n_filters'], k=args.k, size=size))
    elif enc == 'resnet-ibn':
        model = SimCLR(cfg, encoder=ResNetIBN())
    else:
        raise ValueError(f"Invalid encoder: {enc}")
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = DataParallel(model).to(device)
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(device)

    model.eval()
    classifier = CrossAttentionClassifier(in_dim=512, num_nodes=32).to(device)
    # classifier = nn.Identity()

    # Load classifier checkpoint
    clf_ckpt = f'checkpoint/{args.clf_ckp}'
    if os.path.isfile(clf_ckpt):
        print(f"=> Loading classifier checkpoint '{clf_ckpt}'")
        classifier.load_state_dict(torch.load(clf_ckpt))
    else:
        print(f"=> No classifier checkpoint found at '{clf_ckpt}'")

    if not args.ismir25:
        print("Creating dataloaders ...")

        # ir_test_idx = load_augmentation_index(ir_dir, splits=0.8)["test"]
        test_augment = GPUTransformSampleID(cfg=cfg, train=False).to(device)

        query_dataset = Sample100Dataset(cfg, path=args.test_dir, annot_path=annot_path, mode="query")
        query_full_dataset = Sample100Dataset(cfg, path=args.test_dir, annot_path=annot_path, mode="query_full")
        ref_dataset = Sample100Dataset(cfg, path=args.test_dir, annot_path=annot_path, mode="ref")
        dummy_dataset = Sample100Dataset(cfg, path=args.test_dir, annot_path=annot_path, mode="dummy")


        # Create DataLoader instances for each dataset
        dummy_db_loader = DataLoader(dummy_dataset, batch_size=1, 
                                    shuffle=False, num_workers=4, 
                                    pin_memory=True, drop_last=False)

        query_db_loader = DataLoader(query_dataset, batch_size=1, 
                                    shuffle=False, num_workers=4, 
                                    pin_memory=True, drop_last=False)
        
        query_full_db_loader = DataLoader(query_full_dataset, batch_size=1, 
                                    shuffle=False, num_workers=4, 
                                    pin_memory=True, drop_last=False)

        ref_db_loader = DataLoader(ref_dataset, batch_size=1, 
                                shuffle=False, num_workers=4, 
                                pin_memory=True, drop_last=False)
        
    else:   
        print("Using pre-computed fingerprints...")

    if args.small_test:
        index_type = 'l2'
    else:
        index_type = 'ivfpq'

    if args.query_lens is not None:
        args.query_lens = [float(q) for q in args.query_lens.split(',')]
        test_seq_len = [query_len_from_seconds(q, cfg['overlap'], dur=cfg['dur'])
                        for q in args.query_lens]
        

    for ckp_name, epochs in test_cfg.items():
        if not type(epochs) == list:
            epochs = [epochs]   # Hack to handle case where only best ckp is to be tested
        writer = SummaryWriter(f'runs/{ckp_name}')

        for epoch in epochs:
            ckp = os.path.join(model_folder, f'model_{ckp_name}_{str(epoch)}.pth')
            if os.path.isfile(ckp):
                print("=> loading checkpoint '{}'".format(ckp))
                checkpoint = torch.load(ckp)
                # Check for DataParallel
                if 'module' in list(checkpoint['state_dict'].keys())[0] and torch.cuda.device_count() == 1:
                    checkpoint['state_dict'] = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(ckp))
                continue
            
            fp_dir = create_fp_dir(resume=ckp, train=False)
            if args.recompute or os.path.isfile(f'{fp_dir}/dummy_db.mm') is False:
                print("=> Computing dummy fingerprints...")
                create_dummy_db(dummy_db_loader, augment=test_augment,
                                model=model, output_root_dir=fp_dir, verbose=True)
            else:
                print("=> Skipping dummy db creation...")

            if not args.ismir25:
                create_ref_db(ref_db_loader, augment=test_augment,
                                model=model, output_root_dir=fp_dir, verbose=True)
                
                create_query_db(query_db_loader, augment=test_augment,
                                model=model, output_root_dir=fp_dir, verbose=True)

                create_ref_nmatrix(ref_db_loader, augment=test_augment,
                                    model=model, save_dir=f'{fp_dir}/ref_nmatrix', verbose=True)
                
                create_query_nmatrix(query_db_loader, augment=test_augment,
                                    model=model, save_path=f'{fp_dir}/query_nmatrix.npy', verbose=True)
                
                if args.map:
                    create_query_db(query_full_db_loader, augment=test_augment,
                                    model=model, output_root_dir=fp_dir, fname='query_full_db', verbose=True)
                    
                    create_query_nmatrix(query_full_db_loader, augment=test_augment,
                                    model=model, save_path=f'{fp_dir}/query_full_nmatrix.npy', verbose=True)
                # pass
            
            text = f'{args.text}_{str(epoch)}'
            label = epoch if type(epoch) == int else 0


            if args.query_lens is not None:
                hit_rates = eval_faiss_clf(emb_dir=fp_dir,
                                    test_seq_len=test_seq_len, 
                                    classifier=classifier,
                                    index_type=index_type,
                                    nogpu=True) 


                writer.add_text("table", 
                                create_table(hit_rates, 
                                            cfg['overlap'], cfg['dur'],
                                            test_seq_len, text=text), 
                                label)
  
            else:
                hit_rates = eval_faiss_clf(emb_dir=fp_dir, 
                                    classifier=classifier,
                                    index_type=index_type,
                                    nogpu=True)
                
                writer.add_text("table", 
                                create_table(hit_rates, 
                                            cfg['overlap'], cfg['dur'], text=text), 
                                label)
                
            print("-------Test hit-rates-------")
            # Create table
            print(f'Top-1 exact hit rate = {hit_rates[0]}')
            print(f'Top-3 exact hit rate = {hit_rates[1]}')
            print(f'Top-10 exact hit rate = {hit_rates[2]}')
            
            if args.map:

                map_score, k_map = eval_faiss_map_clf(emb_dir=fp_dir, 
                                        classifier=classifier,
                                        index_type='ivfpq',
                                        nogpu=True)



                print("-------Test MAP-------")
                print(f'Mean Average Precision (MAP@{k_map}): {map_score:.4f}')


if __name__ == '__main__':
    main()