import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from types import SimpleNamespace

# Set config path before importing asr_deepspeech to avoid path resolution issues on Windows
config_path = Path(__file__).parent.parent / "asr_deepspeech" / "config.yml"
os.environ["ZAK_ASR_CONFIG"] = str(config_path)


@pytest.fixture
def wav_16k(tmp_path):
    """1-second 440 Hz sine wave at 16 kHz, saved as a WAV file."""
    sample_rate = 16_000
    t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), audio, sample_rate)
    return path


@pytest.fixture
def audio_conf():
    return SimpleNamespace(
        sample_rate=16_000,
        window_size=0.02,
        window_stride=0.01,
        window="hamming",
        speed_volume_perturb=False,
        spec_augment=False,
        noise_dir=None,
        noise_prob=0.4,
        noise_levels=(0.0, 0.5),
    )


@pytest.fixture
def manifest_csv(tmp_path, wav_16k):
    """Two-row CSV manifest: each row points at the same synthetic WAV with a short label."""
    rows = [
        {"audio_filepath": str(wav_16k), "text": "hello"},
        {"audio_filepath": str(wav_16k), "text": "world"},
    ]
    csv_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(str(csv_path), index=False)
    return csv_path
