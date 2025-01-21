import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from torch_audiomentations import Compose,AddBackgroundNoise, ApplyImpulseResponse, ApplyPitchShift
from torch_audiomentations import Identity
from audiomentations import \
    Compose, ApplyImpulseResponse, \
    PitchShift, BitCrush, TimeStretch, \
    SevenBandParametricEQ, Limiter, Gain
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, AmplitudeToDB
import warnings

from util import BandEQ

class GPUTransformSampleID(nn.Module):
    def __init__(self, cfg, ir_dir, noise_dir, train=True, cpu=False, max_transforms_i=1, max_transforms_j=1):
        super(GPUTransformSampleID, self).__init__()
        self.sample_rate = cfg['fs']
        self.ir_dir = ir_dir
        self.noise_dir = noise_dir
        self.overlap = cfg['overlap']
        self.arch = cfg['arch']
        self.n_frames = cfg['n_frames']
        self.train = train
        self.cpu = cpu
        self.cfg = cfg

        # Maximum number of transforms to apply
        self.max_transforms_i = max_transforms_i
        self.max_transforms_j = max_transforms_j

        self.train_transform_i_options = [
            PitchShift(min_semitones=-cfg['pitch_shift'], max_semitones=cfg['pitch_shift'], p=1.0),
            TimeStretch(min_rate=cfg['min_rate'], max_rate=cfg['max_rate'], p=1.0),
        ]

        self.train_transform_j_options = [
            # BandEQ(),
            # BitCrush(min_bit_depth=cfg['min_bit_depth'], max_bit_depth=cfg['min_bit_depth'], p=1.0),
            Gain(min_gain_db=-cfg['gain'], max_gain_db=cfg['gain'], p=1.0),
            ApplyImpulseResponse(ir_path=self.ir_dir, p=1.0),
        ]

        self.val_transform = Identity()

        self.logmelspec = nn.Sequential(
            MelSpectrogram(sample_rate=self.sample_rate, 
                           win_length=cfg['win_len'], 
                           hop_length=cfg['hop_len'], 
                           n_fft=cfg['n_fft'], 
                           n_mels=cfg['n_mels']),
            AmplitudeToDB()
        )

        self.spec_aug = nn.Sequential(
            TimeMasking(cfg['time_mask'], True),
            FrequencyMasking(cfg['freq_mask'], True)
        )

        self.melspec = MelSpectrogram(sample_rate=self.sample_rate, 
                                      win_length=cfg['win_len'], 
                                      hop_length=cfg['hop_len'], 
                                      n_fft=cfg['n_fft'], 
                                      n_mels=cfg['n_mels'])

    def apply_random_transforms(self, audio, transform_options, max_transforms):

        if len(transform_options) == 0 or max_transforms <= 0:
            return audio

        selected_transforms = np.random.choice(transform_options, size=min(max_transforms, len(transform_options)), replace=False)
        for transform in selected_transforms:
            audio = transform(audio, sample_rate=self.sample_rate)
        return audio

    def train_transform_i(self, audio):
        """
        Applies at most max_transforms_i transforms from train_transform_i options.
        """
        return self.apply_random_transforms(audio, self.train_transform_i_options, self.max_transforms_i)

    def train_transform_j(self, audio):
        """
        Applies at most max_transforms_j transforms from train_transform_j options.
        """
        return self.apply_random_transforms(audio, self.train_transform_j_options, self.max_transforms_j)

    
    def forward(self, x_i, x_j):
        if self.cpu:
            x_i = self.train_transform_i(x_i.numpy())
            if x_j.ndim > 1:  
                x_j = x_j.sum(dim=0)  
            # try:
            #     x_j = self.train_transform_j(x_j.numpy(), sample_rate=self.sample_rate)
            # except ValueError:
            #     print("Error loading noise file. Hack to solve issue...")
            #     # Increase length of x_j by 1 sample and retry
            #     x_j = F.pad(x_j, (0, 1))
            #     x_j = self.train_transform_j(x_j.numpy(), sample_rate=self.sample_rate)
            x_j = self.train_transform_j(x_j.numpy())

            return torch.from_numpy(x_i), torch.from_numpy(x_j)[:int(self.sample_rate * self.cfg['dur'])]

        if self.train:
            X_i = self.logmelspec(x_i)
            assert X_i.device == torch.device('cuda:0'), f"X_i device: {X_i.device}"
            X_j = self.logmelspec(x_j)

        else:
            # Test-time transformation: x_i is waveform and x_j is None
            X_i = self.logmelspec(x_i.squeeze(0)).transpose(1, 0)
            try:
                X_i = X_i.unfold(0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap)))
            except RuntimeError:
                print("Error in unfolding. x_i shape: ", X_i.shape)
            X_j = None

        return X_i, X_j
