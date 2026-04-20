"""Tests for SpectrogramDataset, BucketingSampler, and collate."""
import pandas as pd
import pytest
import torch

from asr_deepspeech.data.dataset.spectrogram_dataset import SpectrogramDataset
from asr_deepspeech.data.samplers.bucketing_sampler import BucketingSampler


@pytest.fixture
def dataset(audio_conf, manifest_csv, labels):
    return SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=manifest_csv,
        labels=labels,
        normalize=True,
    )


def test_dataset_len(dataset):
    assert len(dataset) == 1


def test_dataset_getitem_shapes(dataset):
    spec, transcript = dataset[0]
    assert isinstance(spec, torch.Tensor)
    assert spec.ndim == 2  # (freq, time)
    assert isinstance(transcript, list)
    assert len(transcript) > 0


def test_dataset_cache(audio_conf, manifest_csv, labels, tmp_path):
    ds = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=manifest_csv,
        labels=labels,
        cache_dir=str(tmp_path / "cache"),
    )
    spec1, _ = ds[0]
    spec2, _ = ds[0]  # second call reads from cache
    assert torch.allclose(spec1, spec2)


def test_bucketing_sampler(manifest_csv):
    df = pd.read_csv(manifest_csv)
    sampler = BucketingSampler(df, batch_size=1)
    batches = list(sampler)
    assert len(batches) == len(df)
    # each element is a list of indices
    assert all(isinstance(b, list) for b in batches)
    assert all(isinstance(i, int) for batch in batches for i in batch)
