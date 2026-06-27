import torch

import asr_deepspeech.vars as v  # must import without crashing on CPU


def test_vars_imports_on_cpu():
    assert v.supported_rnns["lstm"] is torch.nn.LSTM


def test_deepspeech_call_defaults_to_cpu(audio_conf, label_csv):
    from asr_deepspeech.modules.deepspeech import DeepSpeech

    model = DeepSpeech(
        audio_conf=audio_conf,
        decoder=None,
        label_path=str(label_csv),
        rnn_type="nn.GRU",
        rnn_hidden_size=16,
        rnn_hidden_layers=1,
        bidirectional=True,
    )
    model.eval()
    inputs = torch.randn(1, 1, 161, 40)
    loader = [
        (
            inputs,
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([1.0]),
            torch.tensor([3], dtype=torch.int32),
        )
    ]
    wer, cer, _ = model(loader=loader)  # no device arg -> cpu, must not raise
    assert isinstance(wer, float)
