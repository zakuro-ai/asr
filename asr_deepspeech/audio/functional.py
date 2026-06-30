import contextlib
import subprocess
import wave
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf


def load_audio(path, fq=16000):
    sound, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != fq:
        raise ValueError(f"Expected sample rate {fq}, got {sample_rate} for {path}")
    if sound.ndim > 1:
        sound = sound.mean(axis=1)
    return sound


def duration(path):
    try:
        info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        with contextlib.closing(wave.open(path, "r")) as f:
            return f.getnframes() / float(f.getframerate())


def fq(path):
    try:
        return sf.info(path).samplerate
    except Exception:
        with contextlib.closing(wave.open(path, "r")) as f:
            return f.getframerate()


def get_audio_length(path):
    # Read duration from the file header (soundfile/wave); no external `soxi` binary
    # and no shell invocation, which removes a shell-injection vector on `path`.
    return duration(path)


def audio_with_sox(path, sample_rate, start_time, end_time):
    """Trim and resample `path` with sox, returning the loaded audio.

    Uses an argument list (no shell) so paths cannot inject shell commands.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = [
            "sox",
            str(path),
            "-r",
            str(sample_rate),
            "-c",
            "1",
            "-b",
            "16",
            "-e",
            "si",
            tar_filename,
            "trim",
            str(start_time),
            "=%s" % end_time,
        ]
        subprocess.run(sox_params, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return load_audio(tar_filename, sample_rate)


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """Apply tempo/gain augmentation with sox, returning the loaded audio.

    Uses an argument list (no shell) so paths cannot inject shell commands.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_params = [
            "sox",
            str(path),
            "-r",
            str(sample_rate),
            "-c",
            "1",
            "-b",
            "16",
            "-e",
            "si",
            augmented_filename,
            "tempo",
            "{:.3f}".format(tempo),
            "gain",
            "{:.3f}".format(gain),
        ]
        subprocess.run(sox_params, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return load_audio(augmented_filename, sample_rate)


def load_randomly_augmented_audio(
    path, sample_rate=16000, tempo_range=(0.85, 1.15), gain_range=(-6, 8)
):
    tempo_value = np.random.uniform(*tempo_range)
    gain_value = np.random.uniform(*gain_range)
    return augment_audio_with_sox(
        path=path, sample_rate=sample_rate, tempo=tempo_value, gain=gain_value
    )
