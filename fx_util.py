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

class Compressor(BaseWaveformTransform):
    """
    Apply dynamic range compression to the audio waveform using a feed-forward design.
    Randomizes only the core compression parameters: threshold, ratio, attack, and release.
    """

    def __init__(
        self,
        min_threshold=-24,
        max_threshold=-12,
        ratio = [2, 4, 8, 20],
        min_attack=0.001,
        max_attack=0.1,
        min_release=0.01,
        max_release=0.5,
        knee_db=6.0,  # Fixed knee width
        makeup_gain_db=0.0,  # Fixed makeup gain
        p=0.5,
    ):
        """
        :param min_threshold: Minimum threshold in dB
        :param max_threshold: Maximum threshold in dB
        :param min_ratio: Minimum compression ratio
        :param max_ratio: Maximum compression ratio
        :param min_attack: Minimum attack time in seconds
        :param max_attack: Maximum attack time in seconds
        :param min_release: Minimum release time in seconds
        :param max_release: Maximum release time in seconds
        :param knee_db: Fixed knee width in dB
        :param makeup_gain_db: Fixed makeup gain in dB
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.ratio = ratio
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.min_release = min_release
        self.max_release = max_release
        self.knee_db = knee_db
        self.makeup_gain_db = makeup_gain_db

    def randomize_parameters(self, samples, sample_rate):
        """
        Randomize only threshold, ratio, attack, and release parameters.
        """
        self.parameters = {
            "threshold": random.uniform(self.min_threshold, self.max_threshold),
            "ratio": random.choice(self.ratio),
            "attack": random.uniform(self.min_attack, self.max_attack),
            "release": random.uniform(self.min_release, self.max_release),
        }

    def calculate_attack_release_filters(self, attack_time, release_time, sample_rate):
        """
        Calculate attack and release filter coefficients.
        """
        attack_samples = max(1, int(attack_time * sample_rate))
        release_samples = max(1, int(release_time * sample_rate))
        
        attack_coef = np.exp(-1 / attack_samples)
        release_coef = np.exp(-1 / release_samples)
        
        return attack_coef, release_coef

    def compute_gain_reduction(self, level_db, threshold, ratio):
        """
        Compute the gain reduction in dB using fixed knee width.
        """
        knee_half = self.knee_db / 2
        knee_lower = threshold - knee_half
        knee_upper = threshold + knee_half
        
        gain_reduction = np.zeros_like(level_db)
        
        # Knee region
        knee_mask = (level_db >= knee_lower) & (level_db <= knee_upper)
        knee_range = level_db[knee_mask] - knee_lower
        slope = (1 - 1/ratio) * knee_range / self.knee_db
        gain_reduction[knee_mask] = -slope * knee_range / 2
        
        # Above knee
        above_mask = level_db > knee_upper
        gain_reduction[above_mask] = -(level_db[above_mask] - threshold) * (1 - 1/ratio)
        
        return gain_reduction

    def smooth_gain_reduction(self, gain_reduction_db, attack_coef, release_coef):
        """
        Apply attack and release smoothing to gain reduction.
        """
        smoothed = np.zeros_like(gain_reduction_db)
        smoothed[0] = gain_reduction_db[0]
        
        for i in range(1, len(gain_reduction_db)):
            if gain_reduction_db[i] <= smoothed[i-1]:  # More reduction (attack)
                smoothed[i] = attack_coef * smoothed[i-1] + (1 - attack_coef) * gain_reduction_db[i]
            else:  # Less reduction (release)
                smoothed[i] = release_coef * smoothed[i-1] + (1 - release_coef) * gain_reduction_db[i]
                
        return smoothed

    def apply(self, samples, sample_rate):
        """
        Apply compression to the audio samples.
        """
        if not self.parameters:
            raise RuntimeError("Parameters not randomized! Call randomize_parameters()")

        # Get randomized parameters
        threshold = self.parameters["threshold"]
        ratio = self.parameters["ratio"]
        attack = self.parameters["attack"]
        release = self.parameters["release"]

        # Calculate signal level in dB
        abs_samples = np.abs(samples)
        eps = np.finfo(abs_samples.dtype).eps
        level_db = 20 * np.log10(np.maximum(eps, abs_samples))

        # Compute initial gain reduction
        gain_reduction_db = self.compute_gain_reduction(level_db, threshold, ratio)

        # Get attack/release coefficients and smooth gain reduction
        attack_coef, release_coef = self.calculate_attack_release_filters(
            attack, release, sample_rate
        )
        smoothed_gain_db = self.smooth_gain_reduction(
            gain_reduction_db, attack_coef, release_coef
        )

        # Apply gain reduction and makeup gain
        gain_linear = np.power(10.0, (smoothed_gain_db + self.makeup_gain_db) / 20.0)
        compressed_samples = samples * gain_linear

        return compressed_samples
    



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


