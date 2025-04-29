import os
import torch
import numpy as np
import json
import glob
import yaml
from prettytable import PrettyTable


class DummyScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        pass



def load_index(cfg, json_path, data_dir=None):

    print(f"=>Loading indices from index file {json_path}")
    with open(json_path, 'r') as fp:
        dataset = json.load(fp)

    if data_dir is not None:
        for db_type, index in dataset.items():
            for ix,fpath in enumerate(index):
                if '/' not in fpath:
                    fpath = os.path.join(data_dir, fpath)
                    dataset[db_type][ix] = fpath
                else:
                    break
    
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
        parent_dir = 'logs/emb/test'

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


def calculate_snr(signal, noise):
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr


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

def create_subsets(input_file, sample_ids, name):
    # Load the original JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    subsets = []

    # Filter the data based on sample_id
    for entry in data:
        if entry['sample_id'] in sample_ids:
            subsets.append(entry)

    # Save the filtered data to a new JSON file
    output_file = f"data/{name}.json"
    with open(output_file, 'w') as f:
        json.dump(subsets, f, indent=4)

    return subsets

def main():

    from encoder.dgl.graph_encoder import GraphEncoderDGL

    cfg = 'config/grafp.yaml'
    cfg = load_config(cfg)
    model = GraphEncoderDGL(cfg=cfg, in_channels=8).to('cuda')
    dummy_tensor = torch.rand(16, 8, 256).to('cuda')  # (B, C, N)
    out = model(dummy_tensor)
    print(out.shape)



if __name__ == '__main__':
    main()
