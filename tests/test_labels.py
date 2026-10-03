import pandas as pd
import pytest
import torch

from asr_deepspeech.modules.deepspeech import DeepSpeech

VOCAB = ["_", " ", "a", "b", "c"]


def test_csv_label_files_drop_a_space_row(tmp_path):
    """The reason `labels=` exists: a whitespace-only row does not survive pandas."""
    path = tmp_path / "labels.csv"
    pd.DataFrame({"label": VOCAB}).to_csv(path, index=False)
    assert len(pd.read_csv(path)) == len(VOCAB) - 1


def test_explicit_labels_keep_the_space(audio_conf):
    m = DeepSpeech(
        audio_conf=audio_conf,
        decoder=None,
        labels=VOCAB,
        rnn_type="nn.GRU",
        rnn_hidden_size=8,
        rnn_hidden_layers=1,
    )
    assert m.num_classes == len(VOCAB) and m.labels[" "] == 1
    assert m.decoder.convert_to_strings([torch.tensor([2, 1, 3])])[0][0] == "a b"


def test_labels_must_be_unique(audio_conf):
    with pytest.raises(ValueError, match="unique"):
        DeepSpeech(audio_conf=audio_conf, decoder=None, labels=["_", "a", "a"])


def test_needs_labels_or_label_path(audio_conf):
    with pytest.raises(ValueError, match="labels"):
        DeepSpeech(audio_conf=audio_conf, decoder=None)
