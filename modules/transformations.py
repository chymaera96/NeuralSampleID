import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from torch_audiomentations import Compose,AddBackgroundNoise, ApplyImpulseResponse, ApplyPitchShift
# from torch_audiomentations import Identity
from audiomentations import \
    Compose, ApplyImpulseResponse, \
    PitchShift, BitCrush, TimeStretch, \
    SevenBandParametricEQ, Limiter, Gain
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, AmplitudeToDB
import warnings

from fx_util import BandEQ, Compressor, Identity

class GPUTransformSampleID(nn.Module):
    def __init__(self, cfg, ir_dir, noise_dir, train=True, cpu=False, max_transforms_1=1, max_transforms_2=1):
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
        self.max_transforms_1 = max_transforms_1
        self.max_transforms_2 = max_transforms_2


        self.train_transform_1_options = [
            # BandEQ(),
            # Compressor(min_threshold=-24, max_threshold=-12, 
            #            min_ratio=1.5, max_ratio=4.0, 
            #            min_attack=0.001, max_attack=0.1, 
            #            min_release=0.01, max_release=0.5,
            #            p=1.0),
            # BitCrush(min_bit_depth=cfg['min_bit_depth'], max_bit_depth=cfg['min_bit_depth'], p=1.0),
            # Gain(min_gain_db=-cfg['gain'], max_gain_db=cfg['gain'], p=1.0),
            # ApplyImpulseResponse(ir_path=self.ir_dir, p=1.0),
            Identity()
        ]

        self.train_transform_2_options = [
            # PitchShift(min_semitones=-cfg['pitch_shift'], max_semitones=cfg['pitch_shift'], p=1.0),
            # TimeStretch(min_rate=cfg['min_rate'], max_rate=cfg['max_rate'], p=1.0),
            Identity()
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

    def train_transform_1(self, audio):
        """
        Applies at most max_transforms_1 transforms from train_transform_1 options.
        """
        return self.apply_random_transforms(audio, self.train_transform_1_options, self.max_transforms_1)

    def train_transform_2(self, audio):
        """
        Applies at most max_transforms_2 transforms from train_transform_2 options.
        """
        return self.apply_random_transforms(audio, self.train_transform_2_options, self.max_transforms_2)

    
    def forward(self, x_i, x_j):
        if self.cpu:

            # print(f"In CPU transform x_i shape: {x_i.shape}, x_ns shape: {x_j.shape}")    
            x_s = x_j.sum(dim=0) if x_j.ndim > 1 else x_j
            x_ns = x_i.sum(dim=0) if x_i.ndim > 1 else x_i
            
            x_i = self.train_transform_2(self.train_transform_1(x_s.numpy()) + x_ns.numpy())
            x_j = x_s 

            # print(f"In CPU transform x_i shape: {x_i.shape}, x_j shape: {x_j.shape}")
            
            return torch.from_numpy(x_i), x_j


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

        # print(f"In GPU transform X_i shape: {X_i.shape}, X_j shape: {X_j.shape}")
        return X_i, X_j
