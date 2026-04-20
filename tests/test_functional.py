import torch
import pytest

from asr_deepspeech.functional import _collate_fn, check_loss, to_np


def _make_sample(freq=80, seq=100, label_len=5):
    spec = torch.randn(freq, seq)
    labels = list(range(label_len))
    return spec, labels


def test_collate_fn_shapes():
    batch = [_make_sample(seq=100), _make_sample(seq=80), _make_sample(seq=60)]
    inputs, targets, percentages, target_sizes = _collate_fn(batch)
    assert inputs.shape == (3, 1, 80, 100)  # sorted longest first
    assert targets.dtype == torch.int32
    assert percentages.shape == (3,)
    assert target_sizes.shape == (3,)
    assert percentages[0] == pytest.approx(1.0)
    assert len(targets) == 15  # 3 × 5 labels


def test_collate_fn_single():
    batch = [_make_sample()]
    inputs, targets, percentages, target_sizes = _collate_fn(batch)
    assert inputs.shape[0] == 1
    assert percentages[0] == pytest.approx(1.0)


def test_check_loss_valid():
    loss = torch.tensor(0.5)
    ok, msg = check_loss(loss, 0.5)
    assert ok
    assert msg == ""


def test_check_loss_inf():
    loss = torch.tensor(float("inf"))
    ok, msg = check_loss(loss, float("inf"))
    assert not ok
    assert "inf" in msg


def test_check_loss_nan():
    loss = torch.tensor(float("nan"))
    ok, msg = check_loss(loss, float("nan"))
    assert not ok
    assert "nan" in msg


def test_check_loss_negative():
    loss = torch.tensor(-1.0)
    ok, msg = check_loss(loss, -1.0)
    assert not ok
    assert "negative" in msg


def test_to_np():
    t = torch.tensor([1.0, 2.0, 3.0])
    arr = to_np(t)
    import numpy as np
    assert isinstance(arr, np.ndarray)
    assert list(arr) == [1.0, 2.0, 3.0]
