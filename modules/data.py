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

from zipfile import BadZipFile

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
        offset_frames = int(self.sample_rate * self.offset)

        bass_mix = audio_dict['bass'] + audio_dict['other']
        vocals = audio_dict['vocals']
        drums = audio_dict['drums']

        total_frames = min(len(bass_mix), len(vocals), len(drums))
        segment_length = clip_frames + offset_frames

        if total_frames < segment_length:
            return self[idx + 1]

        start_idx = np.random.randint(0, total_frames - segment_length + 1)
        bass_mix_segment = bass_mix[start_idx:start_idx + segment_length]
        vocals_segment = vocals[start_idx:start_idx + segment_length]
        drums_segment = drums[start_idx:start_idx + segment_length]

        segment = torch.stack([bass_mix_segment, vocals_segment, drums_segment], dim=0)


        # Silence detection using SNR
        valid_channels = []
        for i, stem in enumerate(segment):
            signal = segment.sum(dim=0) - stem
            signal_power = torch.mean(signal**2)
            noise_power = torch.mean((signal - stem)**2)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))

            if snr >= -10:  # SNR threshold
                valid_channels.append(i)

        if len(valid_channels) < 2:  # Not enough valid channels
            return self[idx + 1]

        np.random.shuffle(valid_channels)
        x_j_channels = valid_channels[:len(valid_channels) - 1]     
        x_i_channels = valid_channels[len(valid_channels) - 1:]
        x_j = segment[x_j_channels].sum(dim=0)              # x_j = up to N-1 stems
        x_i = segment[x_i_channels].sum(dim=0)              # x_i = remaining stems 

        # Introduce offset by extracting a random dur-length segment
        x_i_start = np.random.randint(0, offset_frames)
        x_j_start = np.random.randint(0, offset_frames)
        x_i = x_i[x_i_start:x_i_start + clip_frames]
        x_j = x_j[x_j_start:x_j_start + clip_frames]

        if self.transform is not None:
            x_i, x_j = self.transform(x_i, x_j)

        if x_i is None or x_j is None:
            return self[idx + 1]

        if len(x_i) < clip_frames:
            x_i = F.pad(x_i, (0, clip_frames - len(x_i)))
        else:
            x_i = x_i[:clip_frames]

        if len(x_j) < clip_frames:
            x_j = F.pad(x_j, (0, clip_frames - len(x_j)))
        else:
            x_j = x_j[:clip_frames]

        # if self.norm is not None:
        #     norm_val = qtile_norm(torch.cat([x_i, x_j]), q=self.norm)
        #     x_i = x_i / norm_val
        #     x_j = x_j / norm_val

        # Check for silence
        if x_i.abs().max() < self.silence or x_j.abs().max() < self.silence:
            print("Silence detected. Skipping...")
            return self[idx + 1]

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
        self.cfg = cfg

        if self.mode == "dummy":
            self.filenames = load_index(cfg, path, mode="valid")
        else:
            self.filenames = {}

        # self.annotations = pd.read_csv(annot_path)
        with open(self.annot_path, 'r') as fp:
            self.annotations = json.load(fp)
        
        self.ref_names = list(set([rel['ref_file'] for rel in self.annotations]))

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

            rel = self.annotations[idx]
            fname = rel['query_file']
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
                s = rel['start_time']
                e = rel['end_time']
            except KeyError:
                self.ignore_idx.add(idx)
                next_idx = self._get_safe_index(idx)
                return self[next_idx]

            if e == -1:
                x = audio_resampled[int(s*self.sample_rate):]
            elif (e - s) < self.dur / 2:
                self.ignore_idx.add(idx)
                next_idx = self._get_safe_index(idx)
                return self[next_idx]
            else:
                x = audio_resampled[int(s*self.sample_rate):int(e*self.sample_rate)]
    

        elif self.mode == "ref":

            fname = self.ref_names[idx]
            # fname = rel['ref_file']
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

        # if self.norm is not None:
        #     norm_val = qtile_norm(audio_resampled, q=self.norm)
        #     x = x / norm_val
        
        clip_frames = int(self.sample_rate * self.dur)
        # if x.shape[-1] < clip_frames:
        #     x = F.pad(x, (0, clip_frames - x.shape[-1]))
        if x.shape[0] < clip_frames:
            if self.cfg['arch'] == 'resnet-ibn':
                x = F.pad(x, (0, clip_frames - x.shape[0]))
            else:
                repeat_times = (clip_frames // x.shape[0]) + 1
                x = torch.tile(x, (repeat_times,))  # Repeat x along its only dimension
                x = x[:clip_frames] 

        if self.transform is not None:
            x, _ = self.transform(x, None)

        assert x is not None, f"Error loading (data.py) {fname}"

        return fname, x

    def __len__(self):
        if self.mode == "dummy":
            return len(self.filenames['dummy'])
        elif self.mode == "ref":
            return len(self.ref_names)
        else:
            return len(self.annotations)




    

def additive_collate_fn(batch):
    """Collate function that can handle both training and validation data"""
    # Check if we're in validation mode (second element is None)
    if batch[0][1] is None:
        # Validation/test mode
        x_i = torch.stack([item[0] for item in batch])
        metadata = [item[2] for item in batch]
        return x_i, None, metadata
    else:
        # Training mode
        x_i = torch.stack([item[0] for item in batch])
        x_j = torch.stack([item[1] for item in batch])
        metadata = [item[2] for item in batch]

    return x_i, x_j, metadata

class AdditiveSampleIDDataset(Dataset):
    def __init__(self, cfg, path, transform=None, train=False):
        self.path = path
        self.transform = transform
        self.train = train
        self.norm = cfg["norm"]
        self.offset = cfg["offset"]
        self.sample_rate = cfg["fs"]
        self.dur = cfg["dur"]
        self.n_frames = cfg["n_frames"]
        self.silence = cfg["silence"]
        self.error_threshold = cfg["error_threshold"]
        if "stem" in cfg:
            self.stem = cfg["stem"]
        else:
            self.stem = None
        self.keys_dir = cfg.get("train_keys_dir" if train else "val_keys_dir", None)
        self.key_data = None
        if self.keys_dir:
            self.key_data = np.load(self.keys_dir, allow_pickle=True)

        self.beats_dir = cfg.get("train_beats_dir" if train else "val_beats_dir", None)
        self.min_beats_required = int(cfg.get("min_beats_required", 0))

        if train:
            self.filenames = load_index(cfg, path, mode="train")#, stem=self.stem)
        else:
            self.filenames = load_index(cfg, path, mode="valid")#, stem=self.stem)


        print(f"Initial file count: {len(self.filenames)} files from {path}")


        # Filter out files with insufficient beats
        if self.beats_dir:
            filtered_filenames = {}
            original_count = len(self.filenames)
            new_idx = 0  # Keep track of new continuous indices
            for idx, filepath in self.filenames.items():
                beats_data = self.load_beats_file(filepath, train)
                if (
                    beats_data is not None
                    and len(beats_data["times"]) >= self.min_beats_required
                ):
                    filtered_filenames[str(new_idx)] = (
                        filepath  # Use str(new_idx) as key
                    )
                    new_idx += 1
            self.filenames = filtered_filenames
            print(
                f"Filtered out {original_count - len(self.filenames)} files with insufficient beats"
            )

        print(f"Loaded {len(self.filenames)} files from {path}")
        self.ignore_idx = []
        self.error_counts = {}

    def get_key_for_file(self, filepath):
        if self.key_data is None:
            return -1  # Unknown key

        try:
            # Extract relative path from full filepath to match key_data format
            rel_path = "/".join(filepath.split("/")[-2:])  # Gets "091/091869.mp3" format
            return int(self.key_data.get(rel_path, -1))
        except (BadZipFile, EOFError):
            print(f"Warning: Corrupted file {filepath}, returning -1")
            return -1

    def load_beats_file(self, filepath, train=True):
        """Load beats file corresponding to audio file"""
        # Convert audio filepath to beats filepath
        # TODO: This has to be reviewed for stem-separated files (in demucs)
        if train:
            # e.g. /path/to/000/000002.mp3 -> /path/to/beats/000/000002.beats
            beats_path = os.path.join(
                self.beats_dir,
                os.path.basename(os.path.dirname(filepath)),
                os.path.splitext(os.path.basename(filepath))[0] + ".beats",
            )
        else:
            # e.g. /path/to/audio/N003.mp3 -> /path/to/beats/N003.beats
            beats_path = os.path.join(
                self.beats_dir,
                os.path.splitext(os.path.basename(filepath))[0] + ".beats",
            )

        # print(f"Looking for beats file at: {beats_path}")
        if not os.path.exists(beats_path):
            print(f"Beats file not found: {beats_path}")
            return None

        # Load beats file
        beats = []
        beat_numbers = []
        try:
            with open(beats_path) as f:
                for line in f:
                    time, beat = line.strip().split("\t")
                    beats.append(float(time))
                    beat_numbers.append(int(beat))
            return {"times": beats, "numbers": beat_numbers}
        except:
            return None

    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]

        datapath = self.filenames[str(idx)]
        try:
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            audio, sr = torchaudio.load(datapath)

        except Exception:
            print("Error loading:" + self.filenames[str(idx)])
            self.error_counts[idx] = self.error_counts.get(idx, 0) + 1
            if self.error_counts[idx] > self.error_threshold:
                self.ignore_idx.append(idx)
            return self[idx + 1]

        audio_mono = audio.mean(dim=0)

        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        audio_resampled = resampler(audio_mono)

        clip_frames = int(self.sample_rate * self.dur)
        offset_frames = int(self.sample_rate * self.offset)

        segment_length = clip_frames + offset_frames

        if len(audio_resampled) < segment_length:
            return self[idx + 1]

        key = self.get_key_for_file(datapath)
        beats = self.load_beats_file(datapath, self.train)
        metadata = {
            "file_path": datapath,
            "idx": idx,
            "original_sr": sr,
            "length": len(audio_resampled),
            "key": key,
            "beats": beats,
        }

        a_i = audio_resampled

        start_idx = np.random.randint(0, len(audio_resampled) - segment_length + 1)
        a_i = a_i[start_idx : start_idx + segment_length]

        a_j = a_i.clone()

        # Introduce offset by extracting a random dur-length segment
        x_i_start = np.random.randint(0, offset_frames)
        x_j_start = np.random.randint(0, offset_frames)

        x_i = a_i[x_i_start : x_i_start + clip_frames]
        x_j = a_j[x_j_start : x_j_start + clip_frames]

        # Add timestamps to metadata
        metadata.update(
            {
                "start_i": start_idx + x_i_start,
                "start_j": start_idx + x_j_start,
                "clip_length": clip_frames,
            }
        )

        if x_i.abs().max() < self.silence or x_j.abs().max() < self.silence:
            print("Silence detected. Skipping...")
            return self[idx + 1]

        # if self.norm is not None:
        #     norm_val = qtile_norm(audio_resampled, q=self.norm)
        #     x_i = x_i / norm_val
        #     x_j = x_j / norm_val

        if self.transform is not None:
            x_i, x_j, transform_metadata = self.transform(x_i, x_j, metadata)

        if x_i is None or x_j is None:
            return self[idx + 1]

        # Pad or truncate to sample_rate * dur
        if len(x_i) < clip_frames:
            x_i = F.pad(x_i, (0, clip_frames - len(x_i)))
        else:
            x_i = x_i[:clip_frames]

        if len(x_j) < clip_frames:
            x_j = F.pad(x_j, (0, clip_frames - len(x_j)))
        else:
            x_j = x_j[:clip_frames]

        return x_i, x_j, metadata

    def __len__(self):
        return len(self.filenames)
