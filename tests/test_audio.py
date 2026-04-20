"""Tests for audio loading utilities."""
import numpy as np
import pytest


def test_load_audio_wav(tiny_wav):
    from asr_deepspeech.audio.functional import load_audio

    y = load_audio(tiny_wav, 16000)
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert len(y) == 16000


def test_load_audio_wrong_sample_rate_raises(tmp_path):
    import soundfile as sf
    from asr_deepspeech.audio.functional import load_audio

    audio = np.zeros(8000, dtype=np.float32)
    path = str(tmp_path / "8k.wav")
    sf.write(path, audio, 8000)
    with pytest.raises(ValueError, match="Expected sample rate"):
        load_audio(path, 16000)


def test_spectrogram_parser(audio_conf, tiny_wav):
    from asr_deepspeech.data.parsers.spectrogram_parser import SpectrogramParser
    import torch

    parser = SpectrogramParser(audio_conf)
    spec = parser.parse_audio(tiny_wav)
    assert isinstance(spec, torch.Tensor)
    assert spec.ndim == 2
    assert spec.shape[0] > 0  # freq bins
    assert spec.shape[1] > 0  # time frames
