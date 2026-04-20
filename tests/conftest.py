"""Shared fixtures for the ASR test suite."""
import os
import tempfile
import types

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def labels() -> dict[str, int]:
    """Small character vocabulary for unit tests."""
    chars = list(" abcdefghijklmnopqrstuvwxyz'")
    return {c: i for i, c in enumerate(chars)}


@pytest.fixture
def label_csv(tmp_path, labels):
    """Write a labels.csv and return its path."""
    path = tmp_path / "labels.csv"
    pd.DataFrame({"label": list(labels.keys())}).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def audio_conf():
    return types.SimpleNamespace(
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        window="hamming",
        speed_volume_perturb=False,
        spec_augment=False,
        noise_dir=None,
        noise_prob=0.4,
        noise_min=0.0,
        noise_max=0.5,
    )


@pytest.fixture
def tiny_wav(tmp_path) -> str:
    """1-second 16 kHz mono WAV file."""
    import soundfile as sf

    audio = np.zeros(16000, dtype=np.float32)
    path = str(tmp_path / "test.wav")
    sf.write(path, audio, 16000)
    return path


@pytest.fixture
def manifest_csv(tmp_path, tiny_wav) -> str:
    """Single-row manifest CSV."""
    df = pd.DataFrame(
        [{"audio_filepath": tiny_wav, "duration": 1.0, "fq": 16000, "text": "hello world", "text_size": 11}]
    )
    path = str(tmp_path / "manifest.csv")
    df.to_csv(path, index=False)
    return path
