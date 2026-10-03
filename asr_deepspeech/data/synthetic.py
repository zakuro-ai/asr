"""Deterministic synthetic ASR data for tests and benchmarks (no audio files needed)."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from asr_deepspeech.functional import _collate_fn

FREQ_BINS = 161


class SyntheticSpeechDataset(Dataset):
    """Synthetic spectrograms paired with random character targets (indices 1..25).

    With ``learnable=False`` (default) the spectrogram is pure noise, so the targets cannot
    be learned: useful for throughput benchmarks. With ``learnable=True`` every character is
    rendered as ``frames_per_char`` frames of a fixed per-character spectral prototype plus
    noise, so CTC training converges and *time-to-CER* can be measured.
    """

    def __init__(
        self,
        size: int,
        min_frames: int = 60,
        max_frames: int = 120,
        min_chars: int = 4,
        max_chars: int = 10,
        seed: int = 0,
        learnable: bool = False,
        frames_per_char: int = 8,
        noise: float = 0.5,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        # prototypes depend on a fixed seed so train and test splits share them
        protos = torch.randn(26, FREQ_BINS, generator=torch.Generator().manual_seed(1234))
        self.items: List[Tuple[torch.Tensor, List[int]]] = []
        for _ in range(size):
            n = int(torch.randint(min_chars, max_chars + 1, (1,), generator=g))
            target = torch.randint(1, 26, (n,), generator=g).tolist()
            if learnable:
                cols = [protos[c].unsqueeze(1).expand(-1, frames_per_char) for c in target]
                spect = torch.cat(cols, dim=1) + noise * torch.randn(
                    FREQ_BINS, n * frames_per_char, generator=g
                )
            else:
                frames = int(torch.randint(min_frames, max_frames + 1, (1,), generator=g))
                spect = torch.randn(FREQ_BINS, frames, generator=g)
            self.items.append((spect, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[int]]:
        return self.items[index]


def synthetic_loader(size: int, batch_size: int, seed: int = 0, **kwargs) -> DataLoader:
    """A ``DataLoader`` yielding the same batches as the real pipeline."""
    return DataLoader(
        SyntheticSpeechDataset(size, seed=seed, **kwargs),
        batch_size=batch_size,
        collate_fn=_collate_fn,
        shuffle=False,
    )
