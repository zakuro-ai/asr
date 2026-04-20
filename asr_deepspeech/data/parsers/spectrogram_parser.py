import librosa
import numpy as np
import torch

from asr_deepspeech.audio import load_audio, load_randomly_augmented_audio
from asr_deepspeech.data import NoiseInjection
from asr_deepspeech.data.parsers import AudioParser


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, speed_volume_perturb=False, spec_augment=False):
        super().__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window
        self.normalize = normalize
        self.speed_volume_perturb = speed_volume_perturb
        self.spec_augment = spec_augment
        noise_levels = (
            getattr(audio_conf, "noise_min", 0.0),
            getattr(audio_conf, "noise_max", 0.5),
        )
        self.noiseInjector = (
            NoiseInjection(audio_conf.noise_dir, self.sample_rate, noise_levels)
            if getattr(audio_conf, "noise_dir", None) is not None
            else None
        )
        self.noise_prob = getattr(audio_conf, "noise_prob", 0.4)

    def parse_audio(self, audio_path):
        if self.speed_volume_perturb:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path, self.sample_rate)
        if self.noiseInjector and np.random.binomial(1, self.noise_prob):
            y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        hop_length = int(self.sample_rate * self.window_stride)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=self.window)
        spect, _ = librosa.magphase(D)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            spect = (spect - spect.mean()) / (spect.std() + 1e-8)
        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError
