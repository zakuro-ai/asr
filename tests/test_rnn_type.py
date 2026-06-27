import torch.nn as nn

from asr_deepspeech.vars import resolve_rnn_type


def test_resolve_dotted_string():
    assert resolve_rnn_type("nn.LSTM") is nn.LSTM


def test_resolve_bare_lowercase():
    assert resolve_rnn_type("gru") is nn.GRU


def test_resolve_uppercase():
    assert resolve_rnn_type("RNN") is nn.RNN


def test_resolve_passthrough_class():
    assert resolve_rnn_type(nn.LSTM) is nn.LSTM
