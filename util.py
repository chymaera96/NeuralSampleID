import os
import torch
import numpy as np
import json
import glob
import soundfile as sf
import shutil
import yaml
from prettytable import PrettyTable
import re
import matplotlib.pyplot as plt
from audiomentations import BandPassFilter


class DummyScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        pass



def load_index(cfg, data_dir, ext=['wav','mp3'], shuffle_dataset=True, mode="train"):

    if data_dir.endswith('.json'):
        print(f"=>Loading indices from index file {data_dir}")
        with open(data_dir, 'r') as fp:
            dataset = json.load(fp)
        return dataset
    
    print(f"=>Loading indices from {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found")
    
    json_path = os.path.join(cfg['data_dir'], data_dir.split('/')[-1] + ".json")
    if os.path.exists(json_path):
        print(f"Loading indices from {json_path}")
        with open(json_path, 'r') as fp:
            dataset = json.load(fp)
        return dataset
    
    fpaths = glob.glob(os.path.join(data_dir,'**/*.*'), recursive=True)
    fpaths = [p for p in fpaths if p.split('.')[-1] in ext]
    dataset_size = len(fpaths)
    indices = list(range(dataset_size))
    if shuffle_dataset :
        np.random.seed(42)
        np.random.shuffle(indices)
    if mode == "train":
        size = cfg['train_sz']
    else:
        size = cfg['val_sz']
    dataset = {str(i):fpaths[ix] for i,ix in enumerate(indices[:size])}

    with open(json_path, 'w') as fp:
        json.dump(dataset, fp)

    return dataset


def load_nsid_index(cfg, json_path=None, overwrite=False):
    """
    Load or create the nsid index.

    Args:
        htdemucs_dir (str): Directory containing the HTDemucs separated stems.
        fma_dir (str): Directory containing the FMA dataset.
        json_path (str): Path to the JSON file to save/load the index.
        overwrite (bool): If True, overwrite the existing JSON file.

    Returns:
        list: List of dictionaries containing paths to the mix and stems.
    """
    htdemucs_dir = cfg['htdemucs_dir']
    fma_dir = cfg['fma_dir']
    data_dir = cfg['data_dir']

    if json_path is None:
        json_path = os.path.join(data_dir, 'nsid.json')

    if os.path.exists(json_path) and not overwrite:
        print(f"Loading indices from {json_path}")
        with open(json_path, 'r') as fp:
            index = json.load(fp)
        return index

    print(f"Creating index from {htdemucs_dir} and {fma_dir}")
    index = []

    # Create a dictionary to map filenames to their full paths in fma_dir
    fma_files = {}
    for root, _, files in os.walk(fma_dir):
        for file in files:
            if file.endswith('.mp3') and 'htdemucs' not in root:
                fma_files[file] = os.path.join(root, file)

    for fname in os.listdir(htdemucs_dir):
        dict = {}
        mix_file = fname + '.mp3'
        if mix_file in fma_files:
            dict['mix'] = fma_files[mix_file]
        else:
            print(f"Warning: {mix_file} not found in {fma_dir}")
            continue
        dict['vocals'] = os.path.join(htdemucs_dir, fname, 'vocals.mp3')
        dict['drums'] = os.path.join(htdemucs_dir, fname, 'drums.mp3')
        dict['bass'] = os.path.join(htdemucs_dir, fname, 'bass.mp3')
        dict['other'] = os.path.join(htdemucs_dir, fname, 'other.mp3')
        index.append(dict)

    with open(json_path, 'w') as fp:
        json.dump(index, fp)

    return index

def load_augmentation_index(data_dir, splits, json_path=None, ext=['wav','mp3'], shuffle_dataset=True):
    dataset = {'train' : [], 'test' : [], 'validate': []}
    if json_path is None:
        json_path = os.path.join(data_dir, data_dir.split('/')[-1] + ".json")
    if not os.path.exists(json_path):
        fpaths = glob.glob(os.path.join(data_dir,'**/*.*'), recursive=True)
        fpaths = [p for p in fpaths if p.split('.')[-1] in ext]
        dataset_size = len(fpaths)   
        indices = list(range(dataset_size))
        if shuffle_dataset :
            np.random.seed(42)
            np.random.shuffle(indices)
        if type(splits) == list or type(splits) == np.ndarray:
            splits = [int(splits[ix]*dataset_size) for ix in range(len(splits))]
            train_idxs, valid_idxs, test_idxs = indices[:splits[0]], indices[splits[0]: splits[0] + splits[1]], indices[splits[1]:]
            dataset['validate'] = [fpaths[ix] for ix in valid_idxs]
        else:
            splits = int(splits*dataset_size)
            train_idxs, test_idxs = indices[:splits], indices[splits:]
        
        dataset['train'] = [fpaths[ix] for ix in train_idxs]
        dataset['test'] = [fpaths[ix] for ix in test_idxs]

        with open(json_path, 'w') as fp:
            json.dump(dataset, fp)
    
    else:
        with open(json_path, 'r') as fp:
            dataset = json.load(fp)

    return dataset


def get_frames(y, frame_length, hop_length):
    # frames = librosa.util.frame(y.numpy(), frame_length, hop_length, axis=0)
    frames = y.unfold(0, size=frame_length, step=hop_length)
    return frames

def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + torch.quantile(y.abs(),q=q))

def qtile_norm(y, q, eps=1e-8):
    return eps + torch.quantile(y.abs(),q=q)


def query_len_from_seconds(seconds, overlap, dur):
    hop = dur*(1-overlap)
    return int((seconds-dur)/hop + 1)

def seconds_from_query_len(query_len, overlap, dur):
    hop = dur*(1-overlap)
    return (query_len-1)*hop + dur

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    # # Check if dataparallel is used
    # if 'module' in list(checkpoint['state_dict'].keys())[0]:
    #     print("Loading model with dataparallel...")
    #     checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['loss'], checkpoint['valid_acc']

def save_ckp(state,model_name,model_folder,text):
    if not os.path.exists(model_folder): 
        print("Creating checkpoint directory...")
        os.mkdir(model_folder)
    torch.save(state, "{}/model_{}_{}.pth".format(model_folder, model_name, text))

def load_config(config_path):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    return config

def override(config_val, arg):
    return arg if arg is not None else config_val 

def create_fp_dir(resume=None, ckp=None, epoch=1, train=True):
    if train:
        parent_dir = 'logs/emb/valid'
    else:
        parent_dir = '/data/scratch/acw723/logs/emb/valid'

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if resume is not None:
        ckp_name = resume.split('/')[-1].split('.pt')[0]
    else:
        ckp_name = f'model_{ckp}_epoch_{epoch}'
    fp_dir = os.path.join(parent_dir, ckp_name)
    if not os.path.exists(fp_dir):
        os.mkdir(fp_dir)
    return fp_dir



def create_train_set(data_dir, dest, size=10000):
    if not os.path.exists(dest):
        os.mkdir(dest)
        print(data_dir)
        print(dest)
    for ix,fname in enumerate(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if ix <= size and fpath.endswith('mp3'):
            shutil.move(fpath,dest)
            print(ix)
        if len(os.listdir(dest)) >= size:
            return dest
    
    return dest

def create_downstream_set(data_dir, size=5000):
    src = os.path.join(data_dir, f'fma_downstream')
    dest = data_dir
    # if not os.path.exists(dest):
    #     os.mkdir(dest)   
    # if len(os.listdir(dest)) >= size:
    #     return dest
    for ix,fname in enumerate(os.listdir(src)):
        fpath = os.path.join(src, fname)
        if not fpath.endswith('mp3'):
            continue
        # if ix < size:
        if len(os.listdir(src)) > 500:
            shutil.move(fpath,dest)

    return dest

def preprocess_aug_set_sr(data_dir, sr=22050):
    for fpath in glob.iglob(os.path.join(data_dir,'**/*.wav'), recursive=True):
        y, sr = sf.read(fpath)
        print(sr)
        break
        # sf.write(fpath, data=y, samplerate=sr)
    return

def count_parameters(model, encoder):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    # Write table in text file
    with open(f'model_summary_{encoder}.txt', 'w') as f:
        f.write(str(table))
    return total_params

def calculate_output_sparsity(output):
    total_elements = torch.numel(output)
    zero_elements = torch.sum((output == 0).int()).item()

    sparsity = zero_elements / total_elements * 100
    return sparsity

def calculate_snr(signal, noise):
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr

    # Get paths of files not in the index
def get_test_index(data_dir):
    train_idx = load_index(data_dir)
    all_file_list = glob.glob(os.path.join(data_dir,'**/*.mp3'), recursive=True)
    print(f'Number of files in {data_dir}: {len(all_file_list)}')
    # test_idx = {str(i):f for i,f in enumerate(all_file_list) if f not in train_idx.values()}
    idx = 0
    test_idx = {}
    for i, fpath in enumerate(all_file_list):
        if i % 200 == 0:
            print(f"Processed {i}/{len(all_file_list)} files")
        if fpath not in train_idx.values():
            test_idx[str(idx)] = fpath
            idx += 1

    return test_idx


def extract_losses(filename):
    simclr_losses = []
    mixco_losses = []

    with open(filename, 'r') as file:
        for line in file:
            simclr_match = re.search(r'SimCLR Loss: (\d+\.\d+)', line)
            mixco_match = re.search(r'MixCo Loss: (\d+\.\d+)', line)
            if simclr_match:
                simclr_loss = float(simclr_match.group(1))
                simclr_losses.append(simclr_loss)
            if mixco_match:
                mixco_loss = float(mixco_match.group(1))
                mixco_losses.append(mixco_loss)

    return simclr_losses, mixco_losses




def save_nan_batch(x_i, x_j, save_dir="nan_batches", counter=0):
    """
    Save batches with NaN losses for later visualization.
    """
    os.makedirs(save_dir, exist_ok=True)
    x_i = x_i.detach().cpu()
    x_j = x_j.detach().cpu()
    x_i_path = os.path.join(save_dir, f"x_i_{counter}.pt")
    x_j_path = os.path.join(save_dir, f"x_j_{counter}.pt")
    torch.save(x_i, x_i_path)
    torch.save(x_j, x_j_path)
    
    print(f"Saved NaN batch as {x_i_path} and {x_j_path}")
    
    # Increment the counter for the next batch
    return counter + 1



def main():
    # simclr_losses, mixco_losses = extract_losses('hpc_out/nsid_tc_1.o3925245')

    # plt.figure()
    # plt.plot(simclr_losses, label='SimCLR Loss')
    # plt.plot(mixco_losses, label='MixCo Loss')
    # plt.legend()

    # plt.title('Losses per step')
    # plt.xlabel('Step')
    # plt.ylabel('Loss')

    # plt.savefig('losses.jpg')

    # Create index file for sample_100 dataset
    # fma_dir = '/data/home/acw723/datasets/fma/fma_medium'
    # htdemucs_dir = '/data/EECS-Studiosync/datasets/fma_medium/htdemucs'

    # cfg = load_config('config/grafp.yaml')
    # index = load_nsid_index(cfg=cfg)
    # print(index[:2])
    from encoder.pyg.graph_encoder import GraphEncoder
    model = GraphEncoder()
    dummy_tensor = torch.rand(16,64,256)
    out = model(dummy_tensor)
    print(out.shape)

if __name__ == '__main__':
    main()
