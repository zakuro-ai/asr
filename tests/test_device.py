import pytest
import torch

from asr_deepspeech.device import autocast, make_grad_scaler, resolve_device


def test_resolve_cpu_explicit():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_is_cpu_without_cuda():
    if not torch.cuda.is_available():
        assert resolve_device("auto").type == "cpu"


def test_resolve_none_defaults_auto():
    assert isinstance(resolve_device(None), torch.device)


def test_resolve_passthrough_device_object():
    d = torch.device("cpu")
    assert resolve_device(d) == d


def test_resolve_cuda_without_gpu_warns_and_falls_back():
    if not torch.cuda.is_available():
        with pytest.warns(UserWarning):
            assert resolve_device("cuda").type == "cpu"


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_grad_scaler_disabled_on_cpu():
    scaler = make_grad_scaler("cpu", enabled=True)
    assert scaler.is_enabled() is False


def test_autocast_cpu_is_noop_context():
    with autocast("cpu", enabled=True):
        x = torch.ones(2) + 1
    assert x.dtype == torch.float32
