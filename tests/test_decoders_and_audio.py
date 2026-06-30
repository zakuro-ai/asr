"""Tests for decoders, audio helpers, and the rnn_type resolver.

These lock in the security/correctness fixes: the eval()-free rnn_type
resolver, the sox/shell-free audio length, and the ctcdecode error message.
"""

import pytest
import torch
import torch.nn as nn

from asr_deepspeech.audio import duration, fq, get_audio_length, load_audio
from asr_deepspeech.decoders import BeamCTCDecoder, GreedyDecoder
from asr_deepspeech.decoders.decoder import Decoder
from asr_deepspeech.modules.deepspeech import _resolve_rnn_type

# Index 0 is the CTC blank ("_").
LABELS = "_ab "


# ── GreedyDecoder ────────────────────────────────────────────────────────────


def test_greedy_decoder_decode_removes_blanks_and_repeats():
    dec = GreedyDecoder(LABELS)
    # argmax path: a a b blank  ->  "ab"  (repeat collapsed, blank dropped)
    seq = [1, 1, 2, 0]
    probs = torch.zeros(1, len(seq), len(LABELS))
    for t, i in enumerate(seq):
        probs[0, t, i] = 1.0
    strings, offsets = dec.decode(probs, torch.tensor([len(seq)]))
    assert strings[0][0] == "ab"


def test_greedy_decoder_convert_to_strings():
    dec = GreedyDecoder(LABELS)
    seq = torch.tensor([[1, 2, 3, 1]])  # a b <space> a
    out = dec.convert_to_strings(seq)
    assert out == [["ab a"]]


# ── base Decoder wer/cer ─────────────────────────────────────────────────────


def test_decoder_wer():
    dec = GreedyDecoder(LABELS)
    assert dec.wer("a b c", "a b c") == 0
    assert dec.wer("a b", "a b c") == 1


def test_decoder_cer():
    dec = GreedyDecoder(LABELS)
    assert dec.cer("abc", "abc") == 0
    assert dec.cer("abc", "abd") == 1


def test_base_decoder_decode_not_implemented():
    dec = Decoder(LABELS)
    with pytest.raises(NotImplementedError):
        dec.decode(torch.zeros(1, 1, len(LABELS)))


# ── BeamCTCDecoder (optional dep) ────────────────────────────────────────────


def test_beam_decoder_requires_ctcdecode():
    """Without the optional ctcdecode package, construction raises a clear error."""
    try:
        import ctcdecode  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError) as exc:
            BeamCTCDecoder(LABELS)
        assert "ctcdecode" in str(exc.value)
    else:
        pytest.skip("ctcdecode is installed; ImportError path not exercised")


# ── audio helpers (no sox / no shell) ────────────────────────────────────────


def test_audio_duration(wav_16k):
    assert abs(duration(str(wav_16k)) - 1.0) < 0.01


def test_audio_fq(wav_16k):
    assert fq(str(wav_16k)) == 16000


def test_get_audio_length_matches_duration(wav_16k):
    # get_audio_length is now soundfile-based (no soxi subprocess/shell).
    assert get_audio_length(str(wav_16k)) == pytest.approx(duration(str(wav_16k)))


def test_load_audio_mono_shape(wav_16k):
    y = load_audio(str(wav_16k))
    assert y.ndim == 1
    assert len(y) == 16000


def test_load_audio_wrong_sample_rate_raises(wav_16k):
    with pytest.raises(ValueError):
        load_audio(str(wav_16k), fq=8000)


# ── rnn_type resolver (eval-free) ────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [("nn.LSTM", nn.LSTM), ("lstm", nn.LSTM), ("GRU", nn.GRU), ("nn.RNN", nn.RNN)],
)
def test_resolve_rnn_type_strings(value, expected):
    assert _resolve_rnn_type(value) is expected


def test_resolve_rnn_type_passthrough_class():
    assert _resolve_rnn_type(nn.GRU) is nn.GRU


def test_resolve_rnn_type_invalid():
    with pytest.raises(ValueError):
        _resolve_rnn_type("nn.Transformer")
