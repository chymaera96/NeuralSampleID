import random
import numpy as np
import torch
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.transforms_interface import BaseTransform

from audiomentations import BandPassFilter


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
            ) for _ in range(self.num_bands)
        ]

        self.gains = torch.empty(self.num_bands).uniform_(self.min_gain, self.max_gain).tolist()

    def apply_gain(self, audio, gain_db):

        gain_factor = 10 ** (gain_db / 20)
        return audio * gain_factor

    def __call__(self, audio, sample_rate):

        for filter, gain in zip(self.band_pass_filters, self.gains):
            audio = filter(audio, sample_rate=sample_rate) 
            audio = self.apply_gain(audio, gain) 

        return audio


import random
import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import convert_decibels_to_amplitude_ratio


class Compressor(BaseWaveformTransform):
    """
    Apply dynamic range compression to the audio waveform using a feed-forward design.
    """

    def __init__(
        self,
        min_threshold_db=-24.0,
        max_threshold_db=-12.0,
        ratio=[2, 4, 8, 20],
        min_attack=0.001,
        max_attack=0.1,
        min_release=0.01,
        max_release=0.5,
        knee_db=6.0,  # Fixed knee width
        makeup_gain_db=0.0,  # Fixed makeup gain
        p=0.5,
    ):
        """
        :param min_threshold_db: Minimum threshold in dB
        :param max_threshold_db: Maximum threshold in dB
        :param ratio: List of possible compression ratios
        :param min_attack: Minimum attack time in seconds
        :param max_attack: Maximum attack time in seconds
        :param min_release: Minimum release time in seconds
        :param max_release: Maximum release time in seconds
        :param knee_db: Fixed knee width in dB
        :param makeup_gain_db: Fixed makeup gain in dB
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.ratio = ratio
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.min_release = min_release
        self.max_release = max_release
        self.knee_db = knee_db
        self.makeup_gain_db = makeup_gain_db

    @staticmethod
    def convert_time_to_coefficient(t, sample_rate):
        """
        Convert time (attack/release) to a filter coefficient.
        """
        return np.exp(-1.0 / (sample_rate * t))

    def randomize_parameters(self, samples, sample_rate):
        """
        Randomize compression parameters if the transform should apply.
        """
        super().randomize_parameters(samples, sample_rate)

        if self.parameters["should_apply"]:
            self.parameters.update({
                "threshold": convert_decibels_to_amplitude_ratio(
                    random.uniform(self.min_threshold_db, self.max_threshold_db)
                ),
                "ratio": random.choice(self.ratio),
                "attack": self.convert_time_to_coefficient(
                    random.uniform(self.min_attack, self.max_attack), sample_rate
                ),
                "release": self.convert_time_to_coefficient(
                    random.uniform(self.min_release, self.max_release), sample_rate
                ),
            })

    def apply(self, samples, sample_rate):
        """
        Apply dynamic range compression to the audio samples.
        """
        # if not self.parameters["should_apply"]:
        #     return samples

        # Get parameters
        threshold = self.parameters["threshold"]
        ratio = self.parameters["ratio"]
        attack = self.parameters["attack"]
        release = self.parameters["release"]

        # Process audio (simplified dynamic range compression logic)
        compressed_samples = np.zeros_like(samples)
        gain_reduction = 1.0

        for i, sample in enumerate(samples):
            abs_sample = abs(sample)
            if abs_sample > threshold:
                # Apply gain reduction above the threshold
                target_reduction = threshold + (abs_sample - threshold) / ratio
                if gain_reduction > target_reduction:
                    gain_reduction = attack * gain_reduction + (1 - attack) * target_reduction
                else:
                    gain_reduction = release * gain_reduction + (1 - release) * target_reduction
            compressed_samples[i] = sample * gain_reduction

        return compressed_samples


class FrameLevelCorruption(BaseWaveformTransform):
    """
    Applies frame-level corruption by dividing the audio into contiguous segments
    based on a randomly chosen FPS and then applying duplication, removal, and silence insertion
    with their corresponding probabilities.
    """
    def __init__(self, 
                 min_fps=0.5, max_fps=5.0,
                 duplicate_prob=0.1, 
                 remove_prob=0.1, 
                 silence_prob=0.1,
                 p=1.0 
                 ):
        
        super().__init__(p)
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.duplicate_prob = duplicate_prob
        self.remove_prob = remove_prob
        self.silence_prob = silence_prob

    def apply(self, samples, sample_rate):
        num_samples = len(samples)
        fps = random.uniform(self.min_fps, self.max_fps)
        frame_size = int(sample_rate / fps)

        corrupted_samples = []
        i = 0

        while i < num_samples:
            frame = samples[i:i+frame_size]
            if len(frame) == 0:
                break

            # Apply duplication
            if random.random() < self.duplicate_prob:
                frame = np.concatenate((frame, frame))

            # Apply removal
            if random.random() < self.remove_prob:
                frame = np.array([])  # Remove this frame
            
            # Apply silence
            if random.random() < self.silence_prob:
                frame = np.zeros_like(frame)

            corrupted_samples.append(frame)
            i += frame_size

        return np.concatenate(corrupted_samples).astype(np.float32) if corrupted_samples else samples.astype(np.float32)
    


class Identity(BaseTransform):
    """
    A no-op augmentation that simply returns the input audio unchanged.
    """
    def __init__(self, p=1.0):
        super().__init__(p)

    def apply(self, samples, sample_rate):
        return samples

    def __call__(self, samples, sample_rate):
        """
        Makes the Identity class callable and compatible with apply_random_transforms.
        """
        return self.apply(samples, sample_rate)


