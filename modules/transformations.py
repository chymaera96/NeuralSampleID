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
from nnAudio.Spectrogram import CQT

from pedalboard import (
    Pedalboard,
    Chorus,
    Reverb,
    Distortion,
    PitchShift,
    time_stretch,
)


from fx_util import BandEQ, Compressor

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
            # Compressor(min_threshold_db=cfg['DC_threshold'][0], max_threshold_db=cfg['DC_threshold'][1],
            #            ratio=cfg['DC_ratio'],
            #            min_attack=cfg['DC_attack'][0], max_attack=cfg['DC_attack'][1],
            #            min_release=cfg['DC_release'][0], max_release=cfg['DC_release'][1],
            #            p=1.0),
            # BitCrush(min_bit_depth=cfg['min_bit_depth'], max_bit_depth=cfg['min_bit_depth'], p=1.0),
            # Gain(min_gain_db=-cfg['gain'], max_gain_db=cfg['gain'], p=1.0),
            # ApplyImpulseResponse(ir_path=self.ir_dir, p=1.0),
        ]

        self.train_transform_2_options = [
            # PitchShift(min_semitones=-cfg['pitch_shift'], max_semitones=cfg['pitch_shift'], p=1.0),
            # TimeStretch(min_rate=cfg['min_rate'], max_rate=cfg['max_rate'], p=1.0),
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

        # self.spec_aug = nn.Sequential(
        #     TimeMasking(cfg['time_mask'], True),
        #     FrequencyMasking(cfg['freq_mask'], True)
        # )

        self.melspec = MelSpectrogram(sample_rate=self.sample_rate, 
                                      win_length=cfg['win_len'], 
                                      hop_length=cfg['hop_len'], 
                                      n_fft=cfg['n_fft'], 
                                      n_mels=cfg['n_mels'])
        
        self.cqt = CQT(sr=self.sample_rate, hop_length=cfg['hop_len'])

        if self.cfg['arch'] == 'grafp':
            self.spec_func = self.logmelspec
        elif self.cfg['arch'] == 'resnet-ibn':
            self.spec_func = self.cqt
        else:
            raise ValueError(f"Invalid architecture: {self.cfg['arch']}")

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
            X_i = self.spec_func(x_i)
            assert X_i.device == torch.device('cuda:0'), f"X_i device: {X_i.device}"
            X_j = self.spec_func(x_j)


        else:
            # Test-time transformation: x_i is waveform and x_j is None
            X_i = self.spec_func(x_i.squeeze(0))
            # print(f"X_i shape: {X_i.shape}")
            if X_i.ndim == 2:
                X_i = X_i.transpose(1, 0)
            elif X_i.ndim == 3:
                X_i = X_i.squeeze(0).transpose(1, 0)
            # print(f"X_i shape: {X_i.shape}")
            try:
                X_i = X_i.unfold(0, size=self.n_frames, step=int(self.n_frames * (1 - self.overlap)))
            except RuntimeError:
                print("Error in unfolding. x_i shape: ", X_i.shape)
            X_j = None

        # print(f"In GPU transform X_i shape: {X_i.shape}, X_j shape: {X_j.shape}")
        return X_i, X_j









class GPUTransformAdditiveSampleid(nn.Module):
    """
    Transform includes pedalboard effects and song mixing.
    Handles batched data and keeps track of song metadata.
    """

    def __init__(self, cfg, train=True, cpu=False):
        super(GPUTransformAdditiveSampleid, self).__init__()
        self.sample_rate = cfg["fs"]
        self.train = train
        self.cpu = cpu
        self.cfg = cfg

        # Define effects configuration
        self.effects_config = {
            "chorus": {
                "rate_hz": 1.0,
                "depth": 0.25,
                "centre_delay_ms": 7.0,
                "feedback": 0.0,
                "mix": 0.5,
            },
            "reverb": {
                "room_size": 0.8,
                "damping": 0.1,
                "wet_level": 0.5,
                "dry_level": 0.5,
            },
            "distortion": {"drive_db": 15},
        }

        # Create pedalboard effects
        self.chorus = Chorus(**self.effects_config["chorus"])
        self.reverb = Reverb(**self.effects_config["reverb"])
        self.distortion = Distortion(**self.effects_config["distortion"])

        self.mix_prob = float(cfg.get("mix_prob", 0.95))
        self.mix_gain_range = cfg.get("mix_gain_range", [0.05, 0.5])  # Narrower range
        self.mix_gain_range = [float(i) for i in self.mix_gain_range]

        # Keep melspec transform
        self.logmelspec = nn.Sequential(
            MelSpectrogram(
                sample_rate=self.sample_rate,
                win_length=cfg["win_len"],
                hop_length=cfg["hop_len"],
                n_fft=cfg["n_fft"],
                n_mels=cfg["n_mels"],
            ),
            AmplitudeToDB(),
        )

    def get_transpose_semitones(self, from_key, to_key):
        """
        Calculate semitones needed to transpose from one key to another

        Keys 0-11 are major keys (A to G#)
        Keys 12-23 are minor keys (B to Bb)
        Key -1 is unknown
        """
        if from_key == -1 or to_key == -1:  # If either key is unknown
            print(
                "Warning: Unknown key when trying to calculate transpose, returning 0"
            )
            return 0

        # Convert minor keys to their relative major
        if from_key > 11:  # Minor key
            from_key = (from_key - 7) % 12  # -7 to get relative major
        if to_key > 11:
            to_key = (to_key - 7) % 12

        # Calculate the smallest semitone difference needed
        # Calculate direct difference first
        direct_diff = to_key - from_key
        
        # Normalize to find shortest path
        if direct_diff > 6:
            direct_diff -= 12
        elif direct_diff < -6:
            direct_diff += 12
        return direct_diff

    def analyze_tempo(self, beats_data):
        """Calculate tempo and time between beats"""
        if not beats_data or len(beats_data["times"]) < 2:
            print("Warning: Missing beat data, when trying to calculate tempo")
            return None

        beat_times = beats_data["times"]
        # intervals = np.diff(beat_times)
        intervals = torch.diff(torch.tensor(beat_times))
        # median_interval = np.median(intervals)
        median_interval = torch.median(intervals)
        tempo = 60.0 / median_interval

        return {"tempo": tempo, "beat_interval": median_interval}

    def get_tempo_ratio(self, source_tempo, target_tempo):
        """Calculate ratio to match tempos"""
        if not source_tempo or not target_tempo:
            print("Warning: Missing tempo data, using 1.0 ratio")
            return 1.0

        raw_ratio = target_tempo / source_tempo

        # Find the closest power of 2 multiple/divisor that keeps ratio between 0.5 and 2.0
        while raw_ratio > 2.0:
            raw_ratio /= 2.0
        while raw_ratio < 0.5:
            raw_ratio *= 2.0

        return raw_ratio

    def find_alignment_beats(self, main_beats, other_beats):
        """
        Try to find downbeats (beat 1) in both segments, fall back to first beats if not found.
        Returns tuple of (main_beat_time, other_beat_time)
        """
        try:
            # Try to find downbeats first
            main_idx = main_beats["numbers"].index(1)
            other_idx = other_beats["numbers"].index(1)
            return main_beats["times"][main_idx], other_beats["times"][other_idx]
        except ValueError:
            # If no downbeat found, use first available beats
            print(
                "Warning: No downbeat found in one or both segments, using first beats"
            )
            return main_beats["times"][0], other_beats["times"][0]

    def process_audio_batch(self, batch_audio, metadata):
        """Process a batch of audio with random effects and mixing"""
        batch_size = batch_audio.shape[0]
        processed_batch = []

        for i in range(batch_size):
            # Convert to numpy for pedalboard processing IS THIS NEEDED? I HOPE NOT
            audio = batch_audio[i].cpu().numpy()

            # Create random effect chain for this sample
            active_effects = []
            if torch.rand(1).item() < 0.5:
                active_effects.append(self.chorus)
            if torch.rand(1).item() < 0.5:
                active_effects.append(self.reverb)
            if torch.rand(1).item() < 0.5:
                active_effects.append(self.distortion)

            if active_effects:
                board = Pedalboard(active_effects)
                audio = board.process(audio, self.sample_rate)

            # Mix with another random sample from batch if desired
            if torch.rand(1).item() < 0.95:
                # Choose random sample from batch (not self)
                other_idx = (i + torch.randint(1, batch_size, (1,)).item()) % batch_size
                other_audio = batch_audio[other_idx].cpu().numpy()  # TODO: WHY CPU?

                # Get keys from metadata
                main_key = metadata[i]["key"]
                other_key = metadata[other_idx]["key"]

                # Transpose other audio to match main audio's key
                semitones = self.get_transpose_semitones(other_key, main_key)
                # (Im transposing at the same time as time stretching)

                # Get beats information and segment start times
                main_beats = metadata[i].get("beats")
                other_beats = metadata[other_idx].get("beats")

                # Start time of the segment in samples
                main_start = metadata[i].get("start_i")
                other_start = metadata[other_idx].get("start_j")

                # Convert sample indices to seconds
                main_start_sec = main_start / self.sample_rate
                other_start_sec = other_start / self.sample_rate

                # Cut beats to next beat after start until next after start+duration
                first_beat_i = np.searchsorted(main_beats["times"], main_start_sec)
                last_beat_i = np.searchsorted(
                    main_beats["times"], main_start_sec + len(audio) / self.sample_rate
                )
                main_beats = {
                    "times": (
                        np.array(main_beats["times"][first_beat_i:last_beat_i])
                        - main_start_sec
                    ).tolist(),
                    "numbers": main_beats["numbers"][first_beat_i:last_beat_i],
                }

                first_beat_i = np.searchsorted(other_beats["times"], other_start_sec)
                last_beat_i = np.searchsorted(
                    other_beats["times"],
                    other_start_sec + len(audio) / self.sample_rate,
                )
                other_beats = {
                    "times": (
                        np.array(other_beats["times"][first_beat_i:last_beat_i])
                        - other_start_sec
                    ).tolist(),
                    "numbers": other_beats["numbers"][first_beat_i:last_beat_i],
                }

                tempo_ratio = 1.0
                # If we have more than 2 beats, calculate tempo and time stretch
                if len(main_beats["times"]) > 2 and len(other_beats["times"]) > 2:
                    # Calculate tempos if we have beat information
                    main_tempo_data = self.analyze_tempo(main_beats)
                    other_tempo_data = self.analyze_tempo(other_beats)

                    # Handle beat synchronization
                    # if main_tempo_data and other_tempo_data:
                    # Calculate tempo ratio for time stretching
                    tempo_ratio = self.get_tempo_ratio(
                        main_tempo_data["tempo"], other_tempo_data["tempo"]
                    )
                    # If tempo_ratio is a tensor, convert to float
                    if isinstance(tempo_ratio, torch.Tensor):
                        tempo_ratio = tempo_ratio.item()

                    # Time stretch using pedalboard if needed
                    if (
                        abs(1 - tempo_ratio) > 0.02
                    ):  # Only stretch if difference is significant
                        # Convert to audio segment and stretch
                        audio = time_stretch(
                            input_audio=audio,
                            samplerate=self.sample_rate,
                            stretch_factor=tempo_ratio,
                            pitch_shift_in_semitones=semitones,
                            high_quality=True,
                            transient_mode="crisp",
                            transient_detector="compound",
                        )[0]
                        # Stretch beats times
                        main_beats["times"] = [
                            t * tempo_ratio for t in main_beats["times"]
                        ]

                    # Find alignment points (downbeats or first beats)
                    main_nearest_beat, other_nearest_beat = self.find_alignment_beats(
                        main_beats, other_beats
                    )

                    # print("Main nearest beat", main_nearest_beat)
                    # print("Other nearest beat", other_nearest_beat)

                    # Calculate offset needed to align beats
                    offset = int(
                        (main_nearest_beat - other_nearest_beat) * self.sample_rate
                    )

                    # print("Offset", offset)

                    # Apply offset and padding/trimming to same length
                    target_length = len(audio)
                    if offset >= 0:
                        # Add offset zeros at the start
                        other_audio = np.pad(other_audio, (offset, 0))
                        # Then trim/pad to target length
                        if len(other_audio) > target_length:
                            other_audio = other_audio[:target_length]
                        else:
                            other_audio = np.pad(
                                other_audio, (0, target_length - len(other_audio))
                            )
                    elif offset < 0:
                        other_audio = other_audio[-offset:]
                        if len(other_audio) > target_length:
                            # If longer than target, trim the end
                            other_audio = other_audio[:target_length]
                        else:
                            # If shorter than target, pad the end
                            other_audio = np.pad(
                                other_audio, (0, target_length - len(other_audio))
                            )

                    # Verify lengths match before mixing
                    assert len(audio) == len(
                        other_audio
                    ), f"Length mismatch: {len(audio)} vs {len(other_audio)}, after messing with time stretch. Offset was: {offset}."

                else:
                    # If we don't have enough beats, just transpose and mix
                    print(
                        "Warning: Not enough beats in one of the segments to calculate tempo, transposing only"
                    )

                    main_tempo_data = None
                    other_tempo_data = None
                    offset = None
                    # Transpose other audio to match main audio's key
                    if semitones != 0:
                        pitch_shifter = Pedalboard([PitchShift(semitones=semitones)])
                        audio = pitch_shifter.process(
                            audio, self.sample_rate
                        )
                    # Verify lengths match before mixing
                    assert len(audio) == len(
                        other_audio
                    ), f"Length mismatch: {len(audio)} vs {len(other_audio)}, WITHOUT messing with time stretch"

                # Normalize both audios before mixing
                audio = audio / np.abs(audio).max() + 1e-8
                other_audio = other_audio / np.abs(other_audio).max() + 1e-8

                # Random gain between min=0.05 and max=0.5
                gain = (
                    torch.rand(1).item()
                    * (self.mix_gain_range[1] - self.mix_gain_range[0])
                    + self.mix_gain_range[0]
                )

                audio = (1 - gain) * audio + gain * other_audio

                # Update metadata to note the mixing
                metadata[i].update(
                    {
                        "mixed_with": metadata[other_idx]["file_path"],
                        "transpose_semitones": semitones,
                        "mix_gain": gain,
                        "tempo_ratio": tempo_ratio,
                        "main_tempo": (
                            main_tempo_data["tempo"] if main_tempo_data else None
                        ),
                        "other_tempo": (
                            other_tempo_data["tempo"] if other_tempo_data else None
                        ),
                        "beat_offset_samples": (
                            offset if main_beats and other_beats else None
                        ),
                    }
                )

            # Normalize the final mix
            audio = audio / np.abs(audio).max() + 1e-8
            processed_batch.append(torch.from_numpy(audio))

        return torch.stack(processed_batch).to(batch_audio.device)

    def forward(self, x_i, x_j, metadata=None):
        """
        Args:
            x_i: First set of audio segments [B, T]
            x_j: Second set of audio segments [B, T]
            metadata: List of dicts containing info about each sample
        """
        if self.cpu:
            X_i = self.logmelspec(x_i)
            X_j = self.logmelspec(x_j)
            return X_i, X_j, metadata

        if self.train:
            x_j_processed = self.process_audio_batch(x_j, metadata)

            # Add safety checks
            if torch.isnan(x_j_processed).any():
                print("Warning: NaN detected in processed batch")
                # Return original audio as fallback
                x_j_processed = x_j

            # Convert to mel spectrograms
            X_i = self.logmelspec(x_i)
            # X_i = self.logmelspec(x_i_processed)
            X_j = self.logmelspec(x_j_processed)
            # X_i = x_i
            # X_j = x_j_processed

            # Add more safety checks
            if torch.isnan(X_i).any() or torch.isnan(X_j).any():
                print("Warning: NaN detected in spectrograms")

            # Update metadata with mixing info if needed
            if metadata is not None:
                for meta in metadata:
                    meta["augmented"] = True

            return X_i, X_j, metadata

        else:
            # Validation
            # SHOULDNT THIS BE AUGMENTED?
            X_i = self.logmelspec(x_i.squeeze(0)).transpose(1, 0)
            X_i = X_i.unfold(
                0,
                size=self.cfg["n_frames"],
                step=int(self.cfg["n_frames"] * (1 - self.cfg["overlap"])),
            )

            if x_j is None:
                return X_i, X_i, metadata

            X_j = self.logmelspec(x_j.squeeze(0)).transpose(1, 0)
            X_j = X_j.unfold(
                0,
                size=self.cfg["n_frames"],
                step=int(self.cfg["n_frames"] * (1 - self.cfg["overlap"])),
            )

            return X_i, X_j, metadata
