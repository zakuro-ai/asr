import numpy as np
import pytest
import soundfile as sf

from asr_deepspeech.audio.functional import duration, fq, load_audio


def test_load_audio_shape_and_dtype(wav_16k):
    audio = load_audio(str(wav_16k))
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) == 16_000  # 1-second at 16 kHz


def test_load_audio_rejects_wrong_sample_rate(wav_16k):
    with pytest.raises(ValueError, match="Expected sample rate"):
        load_audio(str(wav_16k), fq=8_000)


def test_load_audio_stereo_collapses_to_mono(tmp_path):
    sample_rate = 16_000
    t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, sample_rate)
    audio = load_audio(str(path))
    assert audio.ndim == 1


def test_duration_is_approximately_one_second(wav_16k):
    d = duration(str(wav_16k))
    assert abs(d - 1.0) < 0.01


def test_fq_returns_sample_rate(wav_16k):
    assert fq(str(wav_16k)) == 16_000
