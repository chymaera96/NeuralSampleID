import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import numpy as np
import warnings
from itertools import compress

from util import load_index, load_nsid_index, qtile_norm


class NeuralSampleIDDataset(Dataset):
    def __init__(self, cfg, transform=None, train=False):
        self.transform = transform
        self.train = train
        self.norm = cfg['norm']
        self.offset = cfg['offset']
        self.sample_rate = cfg['fs']
        self.dur = cfg['dur']
        self.silence = cfg['silence']
        self.error_threshold = cfg['error_threshold']
        self.filenames = load_nsid_index(cfg)

        print(f"Dataset loaded with {len(self.filenames)} samples containing {len(self.filenames[0])} stems each")
        self.ignore_idx = []
        self.error_counts = {}

    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]

        datapath = self.filenames[idx]  
        try:
            audio_dict = {stem: torchaudio.load(path)[0].mean(dim=0) for stem, path in datapath.items()}
            sr = torchaudio.load(datapath['mix'])[1]
        except Exception:
            print("Error loading:", datapath)
            self.error_counts[idx] = self.error_counts.get(idx, 0) + 1
            if self.error_counts[idx] > self.error_threshold:
                self.ignore_idx.append(idx)
            return self[idx + 1]

        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        audio_dict = {stem: resampler(audio) for stem, audio in audio_dict.items()}

        clip_frames = int(self.sample_rate * self.dur)

        if len(audio_dict['mix']) <= clip_frames:
            return self[idx + 1]

        bass_mix = audio_dict['bass'] + audio_dict['other']
        vocals = audio_dict['vocals']
        drums = audio_dict['drums']
        combined_stems = {
            'bass_mix': bass_mix,
            'vocals': vocals,
            'drums': drums,
        }

        # Silence detection using SNR
        valid_stems = []
        for key, stem in combined_stems.items():
            other_keys = [k for k in combined_stems.keys() if k != key]
            signal = sum(combined_stems[other] for other in other_keys)
            noise = stem
            signal_power = torch.mean(signal**2)
            noise_power = torch.mean((signal - noise)**2)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))

            if snr >= -20:  # SNR threshold
                valid_stems.append(key)

        if len(valid_stems) < 2:  # Not enough stems to split into x_i and x_j
            print(f"Skipping due to insufficient valid stems")
            return self[idx + 1]

        # Sample up to N-1 stems for x_j and the remaining for x_i
        np.random.shuffle(valid_stems)
        x_j_keys = valid_stems[:len(valid_stems) - 1]
        x_i_keys = valid_stems[len(valid_stems) - 1:]

        x_i = torch.cat([combined_stems[key] for key in x_i_keys])      # x_i = non-sample
        x_j = torch.cat([combined_stems[key] for key in x_j_keys])      # x_j = sample

        if self.transform is not None:
            x_i, x_j = self.transform(x_i, x_j)

        if x_i is None or x_j is None:
            return self[idx + 1]
        
        
        # Ensure lengths match clip_frames
        if len(x_i) < clip_frames:
            x_i = F.pad(x_i, (0, clip_frames - len(x_i)))
        else:
            x_i = x_i[:clip_frames]

        if len(x_j) < clip_frames:
            x_j = F.pad(x_j, (0, clip_frames - len(x_j)))
        else:
            x_j = x_j[:clip_frames]

        # Apply normalization if required
        if self.norm is not None:
            norm_val = qtile_norm(torch.cat([x_i, x_j]), q=self.norm)
            x_i = x_i / norm_val
            x_j = x_j / norm_val


        return x_i, x_j
    
    def __len__(self):
        return len(self.filenames)




    

class Sample100Dataset(Dataset):
    def __init__(self, cfg, path, annot_path, transform=None, mode=None):
        self.path = path
        self.annot_path = annot_path
        self.norm = cfg['norm']
        self.sample_rate = cfg['fs']
        self.n_frames = cfg['n_frames']
        self.dur = cfg['dur']
        self.silence = cfg['silence']
        self.error_threshold = cfg['error_threshold']
        self.transform = transform
        self.mode = mode

        if self.mode == "dummy":
            self.filenames = load_index(cfg, path, mode="valid")
        else:
            self.filenames = {}

        self.annotations = pd.read_csv(annot_path)
        with open(f'{cfg["data_dir"]}/query_dict.json', 'r') as fp:
            self.query_dict = json.load(fp)

        self.error_counts = {}
        self.ignore_idx = set()

    def _get_safe_index(self, idx):
        """Get next valid index, avoiding infinite recursion."""
        next_idx = idx
        max_attempts = len(self)
        attempts = 0
        
        while next_idx in self.ignore_idx and attempts < max_attempts:
            next_idx = (next_idx + 1) % len(self)
            attempts += 1
            
        if attempts >= max_attempts:
            raise RuntimeError("No valid indices available in dataset")
        return next_idx

    def __getitem__(self, idx):
        
        idx = self._get_safe_index(idx)
        
        if self.mode == "query":

            rel = self.annotations.iloc[idx]
            fname = rel['sample_track_id']
            query_path = os.path.join(self.path, fname+'.mp3')
            try:
                audio, sr = torchaudio.load(query_path)
            except Exception as e:
                self.error_counts[idx] = self.error_counts.get(idx, 0) + 1
                if self.error_counts[idx] > self.error_threshold:
                    self.ignore_idx.add(idx)
                raise e

            audio_mono = audio.mean(dim=0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio_resampled = resampler(audio_mono)    
            try:
                segment = np.array(self.query_dict[rel['sample_id']][0]) * self.sample_rate
            except KeyError:
                self.ignore_idx.add(idx)
                next_idx = self._get_safe_index(idx)
                return self[next_idx]

            x = audio_resampled[int(segment[0]): int(segment[1])]
    

        elif self.mode == "ref":

            rel = self.annotations.iloc[idx]
            fname = rel['original_track_id']
            ref_path = os.path.join(self.path, fname+'.mp3')
            assert os.path.exists(ref_path), f"File not found: {ref_path}"
            try:
                audio, sr = torchaudio.load(ref_path)
            except Exception as e:
                self.error_counts[idx] = self.error_counts.get(idx, 0) + 1
                if self.error_counts[idx] > self.error_threshold:
                    self.ignore_idx.add(idx)
                raise e


            audio_mono = audio.mean(dim=0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio_resampled = resampler(audio_mono)   
            x = audio_resampled

        elif self.mode == "dummy":

            datapath = self.filenames['dummy'][idx]
            fname = datapath.split('/')[-1].split('.')[0]
            if not datapath.split('/')[-1].startswith('N'):
                warnings.warn(f"filename not a dummy file: {datapath.split('/')[-1]}")
                return self[idx + 1]
            try:
                audio, sr = torchaudio.load(datapath)
            except Exception as e:
                self.error_counts[idx] = self.error_counts.get(idx, 0) + 1
                if self.error_counts[idx] > self.error_threshold:
                    self.ignore_idx.add(idx)
                raise e
            
            audio_mono = audio.mean(dim=0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio_resampled = resampler(audio_mono)   
            x = audio_resampled
        
        else:
            raise ValueError("Invalid Evaluation mode")

        if self.norm is not None:
            norm_val = qtile_norm(audio_resampled, q=self.norm)
            x = x / norm_val
        
        clip_frames = int(self.sample_rate * self.dur)
        if x.shape[-1] < clip_frames:
            x = F.pad(x, (0, clip_frames - x.shape[-1]))

        if self.transform is not None:
            x, _ = self.transform(x, None)

        assert x is not None, f"Error loading (data.py) {fname}"

        return fname, x

    def __len__(self):
        if self.mode == "dummy":
            return len(self.filenames['dummy'])

        else:
            return len(self.annotations)