import pytest
from asr_deepspeech.decoders.decoder import Decoder
from asr_deepspeech.decoders.greedy_decoder import GreedyDecoder


def test_wer_identical():
    d = Decoder({})
    assert d.wer("hello world", "hello world") == 0


def test_wer_one_substitution():
    d = Decoder({})
    assert d.wer("hello world", "hello earth") == 1


def test_cer_identical():
    d = Decoder({})
    assert d.cer("abc", "abc") == 0


def test_cer_one_delete():
    d = Decoder({})
    assert d.cer("abc", "ab") == 1


def test_wer_empty_strings():
    d = Decoder({})
    assert d.wer("", "") == 0


def test_greedy_decoder_convert_to_strings(labels):
    import torch

    decoder = GreedyDecoder(labels)
    # tokens for "hi" — use torch tensors (as produced by the data pipeline)
    encoded = [torch.tensor([labels["h"], labels["i"]], dtype=torch.int32)]
    result = decoder.convert_to_strings(encoded)
    assert result[0][0] == "hi"


def test_greedy_decoder_decode_shape(labels):
    import torch

    decoder = GreedyDecoder(labels)
    n_classes = len(labels)
    # batch=1, time=10, classes
    probs = torch.zeros(1, 10, n_classes)
    # Force argmax to blank (index 0) — yields empty transcript
    decoded, offsets = decoder.decode(probs, torch.tensor([10]))
    assert len(decoded) == 1
    assert isinstance(decoded[0][0], str)
