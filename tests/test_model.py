"""Lightweight CPU tests for DeepSpeech forward pass."""
import torch
import pytest

from asr_deepspeech.modules.deepspeech import DeepSpeech


@pytest.fixture
def tiny_model(label_csv, audio_conf):
    return DeepSpeech(
        audio_conf=audio_conf,
        label_path=label_csv,
        rnn_hidden_size=64,
        rnn_hidden_layers=2,
        bidirectional=True,
        rnn_type="nn.GRU",
    )


def test_model_builds(tiny_model):
    assert tiny_model is not None
    assert tiny_model.num_classes > 0


def test_model_forward(tiny_model):
    batch, freq, time = 2, 161, 50
    x = torch.zeros(batch, 1, freq, time)
    lengths = torch.full((batch,), time, dtype=torch.int32)
    out, out_lens = tiny_model(x, lengths)
    assert out.shape[0] == batch
    assert out.shape[2] == tiny_model.num_classes
    assert out_lens.shape == (batch,)


def test_model_unknown_rnn_type(label_csv, audio_conf):
    with pytest.raises(ValueError, match="Unknown rnn_type"):
        DeepSpeech(audio_conf=audio_conf, label_path=label_csv, rnn_type="bogus")


def test_model_get_seq_lens(tiny_model):
    lengths = torch.tensor([100, 80, 60], dtype=torch.int32)
    out = tiny_model.get_seq_lens(lengths)
    assert out.shape == (3,)
    # output lengths should be shorter due to strided convolutions
    assert (out <= lengths).all()
