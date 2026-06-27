"""End-to-end pipeline regression test on the REAL data path.

Exercises dataset-prep output -> SpectrogramParser (librosa STFT) ->
SpectrogramDataset -> BucketingSampler -> AudioDataLoader collate -> model
forward + CTC loss -> greedy-decode evaluation. This is the path
``model.get_loader`` drives; the prior loader test bypassed BucketingSampler,
so the sampler's torch-2.x breakage went unnoticed.
"""

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn

from asr_deepspeech.modules.deepspeech import DeepSpeech

SR = 16_000


def _prep(tmp_path):
    """Emulate the ETL's bronze-layer output: WAVs + manifest + labels CSV."""
    rows = []
    for i, text in enumerate(["abc", "bcd", "cde", "dab"]):
        path = tmp_path / f"a{i}.wav"
        t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
        sf.write(str(path), 0.5 * np.sin(2 * np.pi * (220 + 40 * i) * t), SR)
        rows.append({"audio_filepath": str(path), "text": text})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(str(manifest), index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"label": list("abcde")}).to_csv(str(labels), index=False)
    return manifest, labels


def _model(labels, audio_conf):
    return DeepSpeech(
        audio_conf=audio_conf,
        decoder=None,
        label_path=str(labels),
        rnn_type="nn.GRU",
        rnn_hidden_size=32,
        rnn_hidden_layers=1,
        bidirectional=True,
    )


def test_get_loader_builds_real_loader(tmp_path, audio_conf):
    """model.get_loader builds a working loader via BucketingSampler (torch-2.x safe)."""
    manifest, labels = _prep(tmp_path)
    model = _model(labels, audio_conf)
    loader, sampler = model.get_loader(str(manifest), batch_size=2, num_workers=0)
    assert len(sampler) == 2  # 4 samples / batch_size 2
    inputs, targets, input_percentages, target_sizes = next(iter(loader))
    assert inputs.ndim == 4 and inputs.shape[1] == 1  # (B, 1, freq, time)
    assert inputs.shape[0] == 2
    assert target_sizes.sum().item() == targets.numel()


def test_pipeline_forward_and_eval_on_real_loader(tmp_path, audio_conf):
    """A real batch flows through forward + CTC loss (finite) and decode/eval runs."""
    torch.manual_seed(0)
    manifest, labels = _prep(tmp_path)
    model = _model(labels, audio_conf)
    loader, _ = model.get_loader(str(manifest), batch_size=2, num_workers=0)

    criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
    model.train()
    inputs, targets, input_percentages, target_sizes = next(iter(loader))
    input_sizes = input_percentages.mul(int(inputs.size(3))).int()
    out, output_sizes = model.forward(inputs, input_sizes)
    out = out.transpose(0, 1).float().log_softmax(2)
    loss = criterion(out, targets, output_sizes, target_sizes)
    assert torch.isfinite(loss)

    model.eval()
    wer, cer, output_data = model(loader=loader)  # device defaults to auto -> cpu
    assert isinstance(wer, float) and isinstance(cer, float)
    assert isinstance(output_data, list)
