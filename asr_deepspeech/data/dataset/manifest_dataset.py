"""Speech dataset driven by an in-memory manifest, with tar-shard clip addressing.

A manifest row points at one clip:

* ``audio`` is a path (absolute, or relative to ``root``), **or**
* ``audio`` is ``<shard>#<member>`` and ``offset`` / ``size`` give the clip's byte span
  inside ``<root>/<shard>`` (an uncompressed tar, as produced by Zakuro hub packaging). The
  span is read directly, so a clip costs one seek and one read - no tar scan.

When a row carries ``sha256`` the clip's bytes are verified on every read (``verify=True``).
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

from asr_deepspeech.data.parsers import SpectrogramParser

BLANK = "_"


def build_vocabulary(transcripts: Sequence[str]) -> List[str]:
    """Deterministic CTC alphabet: blank first (index 0), then every character seen."""
    chars = sorted({c for t in transcripts for c in t.replace("\n", "")})
    if BLANK in chars:
        raise ValueError(f"transcripts must not contain the CTC blank symbol {BLANK!r}")
    return [BLANK] + chars


def read_clip(
    root: Union[str, Path],
    ref: str,
    offset: Optional[int] = None,
    size: Optional[int] = None,
    sample_rate: int = 16000,
    sha256: Optional[str] = None,
) -> np.ndarray:
    """Load one mono float32 waveform from a file or a tar-shard byte span.

    With ``sha256`` the clip's raw bytes are verified first: reading different bytes than the
    manifest pinned raises ``ValueError`` instead of silently training on them."""
    root = Path(root)
    if "#" in ref:
        shard, member = ref.split("#", 1)
        shard_path = root / shard
        if offset is not None and size is not None:
            with open(shard_path, "rb") as fh:
                fh.seek(int(offset))
                blob = fh.read(int(size))
        else:  # slow path: no byte span recorded
            with tarfile.open(shard_path) as tf:
                extracted = tf.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(f"{member} not in {shard_path}")
                blob = extracted.read()
        if sha256 and hashlib.sha256(blob).hexdigest() != sha256:
            raise ValueError(
                f"{ref}: bytes do not match the manifest sha256 (corrupt or changed shard)"
            )
        sound, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    else:
        path = Path(ref)
        path = path if path.is_absolute() else root / path
        if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError(
                f"{ref}: bytes do not match the manifest sha256 (corrupt or changed file)"
            )
        sound, sr = sf.read(path, dtype="float32")
    if sr != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz, got {sr} Hz for {ref}")
    if sound.ndim > 1:
        sound = sound.mean(axis=1)
    return sound


class ManifestDataset(Dataset, SpectrogramParser):
    """``(spectrogram, target_ids)`` pairs for rows of ``audio`` / ``transcript``."""

    def __init__(
        self,
        rows: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
        labels: Mapping[str, int],
        audio_conf: Any,
        root: Union[str, Path] = ".",
        normalize: bool = True,
        spec_augment: bool = False,
        audio_col: str = "audio",
        text_col: str = "transcript",
        verify: bool = True,
    ) -> None:
        self.rows: List[Dict[str, Any]] = (
            rows.to_dict("records") if isinstance(rows, pd.DataFrame) else [dict(r) for r in rows]
        )
        self.labels_map = dict(labels)
        self.root = Path(root)
        self.audio_col, self.text_col = audio_col, text_col
        self.verify = verify
        SpectrogramParser.__init__(self, audio_conf, normalize, False, spec_augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        wave = read_clip(
            self.root,
            str(row[self.audio_col]),
            row.get("offset"),
            row.get("size"),
            self.sample_rate,
            sha256=(row.get("sha256") if self.verify else None) or None,
        )
        return self.parse_waveform(wave), self.parse_transcript(str(row[self.text_col]))

    def parse_transcript(self, transcript: str) -> List[int]:
        transcript = transcript.replace("\n", "")
        return [i for i in (self.labels_map.get(c) for c in transcript) if i is not None]


def manifest_loader(
    dataset: ManifestDataset, batch_size: int, num_workers: int = 0, shuffle: bool = True
):
    """Same bucketing loader as :func:`asr_deepspeech.data.loaders.get_loader`."""
    from asr_deepspeech.data.loaders import AudioDataLoader
    from asr_deepspeech.data.samplers import BucketingSampler

    sampler = BucketingSampler(dataset, batch_size=batch_size)
    loader = AudioDataLoader(dataset, num_workers=num_workers, batch_sampler=sampler)
    if shuffle:
        sampler.shuffle()
    return loader, sampler
