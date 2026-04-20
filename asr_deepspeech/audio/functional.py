import contextlib
import os
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
    output = subprocess.check_output(['soxi -D "%s"' % path.strip()], shell=True)
    return float(output)


def audio_with_sox(path, sample_rate, start_time, end_time):
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = (
            'sox "{}" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1'.format(
                path, sample_rate, tar_filename, start_time, end_time
            )
        )
        os.system(sox_params)
        return load_audio(tar_filename, sample_rate)


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = 'sox "{}" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1'.format(
            path, sample_rate, augmented_filename, " ".join(sox_augment_params)
        )
        os.system(sox_params)
        return load_audio(augmented_filename, sample_rate)


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15), gain_range=(-6, 8)):
    tempo_value = np.random.uniform(*tempo_range)
    gain_value = np.random.uniform(*gain_range)
    return augment_audio_with_sox(path=path, sample_rate=sample_rate, tempo=tempo_value, gain=gain_value)
