import hashlib
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from asr_deepspeech.data.parsers import SpectrogramParser


def _cache_path(audio_path: str, cache_dir: str) -> str:
    h = hashlib.md5(audio_path.encode()).hexdigest()
    return os.path.join(cache_dir, h[:2], h + ".pt")


class SpectrogramDataset(Dataset, SpectrogramParser):
    """Loads audio, computes spectrograms, and maps transcripts to label indices.

    Disk cache (``cache_dir``):
        On first access each spectrogram is computed and saved to a .pt file.
        Subsequent epochs (and restarts) load from disk — no recomputation.
        Set ``cache_dir=None`` to disable.
    """

    def __init__(
        self,
        audio_conf,
        manifest_filepath: str,
        labels,
        normalize: bool = False,
        spec_augment: bool = False,
        cache_dir: str | None = None,
    ):
        self.df = pd.read_csv(manifest_filepath)
        self.size = len(self.df)
        if isinstance(labels, str):
            labels = {v: k for k, v in pd.read_csv(labels).to_dict()["label"].items()}
        self.labels_map = labels
        self.cache_dir = cache_dir
        super().__init__(audio_conf, normalize, audio_conf.speed_volume_perturb, spec_augment)

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        audio_path: str = sample.audio_filepath
        transcript: str = sample.text

        spec = self._load_spec(audio_path)
        return spec, self._encode_transcript(transcript)

    def _load_spec(self, audio_path: str) -> torch.Tensor:
        if self.cache_dir is not None:
            path = _cache_path(audio_path, self.cache_dir)
            if os.path.exists(path):
                return torch.load(path, weights_only=True)
            spec = self.parse_audio(audio_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(spec, path)
            return spec
        return self.parse_audio(audio_path)

    def _encode_transcript(self, transcript: str) -> list:
        return list(filter(None, [self.labels_map.get(c) for c in transcript.replace("\n", "")]))

    # Keep the name parse_transcript for compatibility with SpectrogramParser
    def parse_transcript(self, transcript: str) -> list:
        return self._encode_transcript(transcript)

    def __len__(self) -> int:
        return self.size

    def warm_cache(self):
        """Pre-populate the disk cache — useful to run once before training."""
        if self.cache_dir is None:
            raise ValueError("cache_dir is not set")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Warming spectrogram cache"):
            self._load_spec(row.audio_filepath)
