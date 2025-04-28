import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_audiomentations import Identity
from audiomentations import Compose, ApplyImpulseResponse, PitchShift, BitCrush, TimeStretch, SevenBandParametricEQ, Limiter, Gain
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, AmplitudeToDB
from nnAudio.Spectrogram import CQT

from fx_util import BandEQ, Compressor, FrameLevelCorruption

class GPUTransformSampleID(nn.Module):
    def __init__(self, cfg, ir_dir=None, train=True, cpu=False, max_transforms_1=1, max_transforms_2=1):
        super(GPUTransformSampleID, self).__init__()
        self.sample_rate = cfg['fs']
        self.ir_dir = ir_dir
        self.overlap = cfg['overlap']
        self.arch = cfg['arch']
        self.n_frames = cfg['n_frames']
        self.train = train
        self.cpu = cpu
        self.cfg = cfg

        self.max_transforms_1 = max_transforms_1
        self.max_transforms_2 = max_transforms_2

        self.logmelspec = nn.Sequential(
            MelSpectrogram(sample_rate=self.sample_rate, 
                           win_length=cfg['win_len'], 
                           hop_length=cfg['hop_len'], 
                           n_fft=cfg['n_fft'], 
                           n_mels=cfg['n_mels']),
            AmplitudeToDB()
        )

        self.cqt = CQT(sr=self.sample_rate, hop_length=cfg['hop_len'])

        if self.arch == 'grafp':
            self.spec_func = self.logmelspec
            self.train_transform_1_options = [
                Gain(min_gain_db=-cfg['gain'], max_gain_db=cfg['gain'], p=1.0),
            ]
            self.train_transform_2_options = [
                PitchShift(min_semitones=-cfg['pitch_shift'], max_semitones=cfg['pitch_shift'], p=1.0),
                TimeStretch(min_rate=cfg['min_rate'], max_rate=cfg['max_rate'], p=1.0),
            ]
        elif self.arch == 'resnet-ibn':
            self.spec_func = self.cqt
            self.train_transform_1_options = [
                BandEQ(),
                Compressor(min_threshold_db=cfg['DC_threshold'][0], max_threshold_db=cfg['DC_threshold'][1],
                           ratio=cfg['DC_ratio'],
                           min_attack=cfg['DC_attack'][0], max_attack=cfg['DC_attack'][1],
                           min_release=cfg['DC_release'][0], max_release=cfg['DC_release'][1],
                           p=1.0),
                Gain(min_gain_db=-cfg['gain'], max_gain_db=cfg['gain'], p=1.0),
            ]
            self.train_transform_2_options = [
                PitchShift(min_semitones=-cfg['pitch_shift'], max_semitones=cfg['pitch_shift'], p=1.0),
                TimeStretch(min_rate=cfg['min_rate'], max_rate=cfg['max_rate'], p=1.0),
                FrameLevelCorruption(remove_prob=0.0, silence_prob=0.0),
                FrameLevelCorruption(duplicate_prob=0.0, silence_prob=0.0),
                FrameLevelCorruption(duplicate_prob=0.0, remove_prob=0.0),
            ]
        else:
            raise ValueError(f"Unsupported arch: {self.arch}")

    def apply_random_transforms(self, audio, transform_options, max_transforms):
        if len(transform_options) == 0 or max_transforms <= 0:
            return audio

        selected_transforms = np.random.choice(transform_options, size=min(max_transforms, len(transform_options)), replace=False)
        for transform in selected_transforms:
            audio = transform(audio, sample_rate=self.sample_rate)
        return audio

    def train_transform_1(self, audio):
        return self.apply_random_transforms(audio, self.train_transform_1_options, self.max_transforms_1)

    def train_transform_2(self, audio):
        return self.apply_random_transforms(audio, self.train_transform_2_options, self.max_transforms_2)

    def forward(self, x_i, x_j):
        if self.cpu:
            x_s = x_j.sum(dim=0) if x_j.ndim > 1 else x_j
            x_ns = x_i.sum(dim=0) if x_i.ndim > 1 else x_i
            x_i = self.train_transform_2(self.train_transform_1(x_s.numpy()) + x_ns.numpy())
            x_j = x_s
            return torch.from_numpy(x_i), x_j

        if self.train:
            X_i = self.spec_func(x_i)
            assert X_i.device == torch.device('cuda:0'), f"X_i device: {X_i.device}"
            X_j = self.spec_func(x_j)
        else:
            X_i = self.spec_func(x_i.squeeze(0))
            if X_i.ndim == 2:
                X_i = X_i.transpose(1, 0)
            elif X_i.ndim == 3:
                X_i = X_i.squeeze(0).transpose(1, 0)
            try:
                X_i = X_i.unfold(0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap)))
            except RuntimeError:
                pass
            X_j = None

        return X_i, X_j
