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



class BandEQ:
    def __init__(self, num_bands=[1,8], center_freq=[50.0, 8000.0],
                 bandwidth_fraction=[0.01,1.0], gain=[-20.0, 10.0]):
        """
        Band EQ using multiple band-pass filters with adjustable frequency, Q, and gain.
        """
        self.num_bands = np.random.randint(num_bands[0], num_bands[1] + 1)
        self.min_center_freq = center_freq[0]
        self.max_center_freq = center_freq[1]
        self.min_bandwidth_fraction = bandwidth_fraction[0]
        self.max_bandwidth_fraction = bandwidth_fraction[1]
        self.min_gain = gain[0]
        self.max_gain = gain[1]

        self.band_pass_filters = [
            BandPassFilter(
                min_center_freq=self.min_center_freq,
                max_center_freq=self.max_center_freq,
                min_bandwidth_fraction=self.min_bandwidth_fraction,
                max_bandwidth_fraction=self.max_bandwidth_fraction,
                p=1.0  
            ) for _ in range(num_bands)
        ]

        self.gains = torch.empty(num_bands).uniform_(self.min_gain, self.max_gain).tolist()

    def apply_gain(self, audio, gain_db):

        gain_factor = 10 ** (gain_db / 20)
        return audio * gain_factor

    def __call__(self, audio):

        for filter, gain in zip(self.band_pass_filters, self.gains):
            audio = filter(audio) 
            audio = self.apply_gain(audio, gain) 

        return audio



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
    data_dir = '../datasets/sample_100/audio'
    index = {'db': [], 'dummy': []}
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fpath.endswith('.mp3'):
            if fname.startswith('T'):
                index['db'].append(fpath)
            else:
                index['dummy'].append(fpath)
        
    with open('data/sample_100.json', 'w') as fp:
        json.dump(index, fp)

if __name__ == '__main__':
    main()
