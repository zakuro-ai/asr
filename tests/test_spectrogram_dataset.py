import pytest
import torch
import torch.nn as nn

from asr_deepspeech.audio.wav_converter import WAVConverter
from asr_deepspeech.data.dataset import SpectrogramDataset
from asr_deepspeech.data.loaders import AudioDataLoader
from asr_deepspeech.data.parsers import SpectrogramParser
from asr_deepspeech.etl.jsut_dataset import JSUTDataset
from asr_deepspeech.etl.librispeech_dataset import LibriSpeechDataset
from asr_deepspeech.functional import _collate_fn, check_loss, to_np
from asr_deepspeech.loggers.tensorboard_logger import TensorBoardLogger
from asr_deepspeech.modules.blocks import (
    BatchRNN,
    InferenceBatchSoftmax,
    Lookahead,
    MaskConv,
    SequenceWise,
)
from asr_deepspeech.modules.deepspeech import DeepSpeech

# Labels map: char -> int index.  Index 0 is intentionally unused (CTC blank).
LABELS_MAP = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
# Expected STFT freq bins: n_fft/2 + 1 = int(16000 * 0.02)/2 + 1 = 161
EXPECTED_FREQ_BINS = 161


# ── SpectrogramParser ────────────────────────────────────────────────────────


def test_parse_audio_returns_float_tensor(wav_16k, audio_conf):
    parser = SpectrogramParser(audio_conf, normalize=False)
    spect = parser.parse_audio(str(wav_16k))
    assert isinstance(spect, torch.FloatTensor)


def test_parse_audio_freq_bins(wav_16k, audio_conf):
    parser = SpectrogramParser(audio_conf, normalize=False)
    spect = parser.parse_audio(str(wav_16k))
    assert spect.shape[0] == EXPECTED_FREQ_BINS


def test_parse_audio_time_frames_positive(wav_16k, audio_conf):
    parser = SpectrogramParser(audio_conf, normalize=False)
    spect = parser.parse_audio(str(wav_16k))
    assert spect.shape[1] > 0


def test_parse_audio_normalize_zero_mean(wav_16k, audio_conf):
    parser = SpectrogramParser(audio_conf, normalize=True)
    spect = parser.parse_audio(str(wav_16k))
    assert abs(spect.mean().item()) < 1e-3


def test_parse_audio_all_values_finite(wav_16k, audio_conf):
    parser = SpectrogramParser(audio_conf, normalize=False)
    spect = parser.parse_audio(str(wav_16k))
    assert torch.isfinite(spect).all()


# ── SpectrogramDataset ───────────────────────────────────────────────────────


def test_dataset_len(manifest_csv, audio_conf):
    ds = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=str(manifest_csv),
        labels=LABELS_MAP,
        normalize=False,
    )
    assert len(ds) == 2


def test_dataset_getitem_returns_tensor_and_list(manifest_csv, audio_conf):
    ds = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=str(manifest_csv),
        labels=LABELS_MAP,
        normalize=False,
    )
    spect, transcript = ds[0]
    assert isinstance(spect, torch.FloatTensor)
    assert isinstance(transcript, list)


def test_dataset_getitem_spectrogram_shape(manifest_csv, audio_conf):
    ds = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=str(manifest_csv),
        labels=LABELS_MAP,
        normalize=False,
    )
    spect, _ = ds[0]
    assert spect.shape[0] == EXPECTED_FREQ_BINS
    assert spect.shape[1] > 0


def test_dataset_parse_transcript_known_chars(manifest_csv, audio_conf):
    ds = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=str(manifest_csv),
        labels=LABELS_MAP,
        normalize=False,
    )
    # "hello" -> [h=8, e=5, l=12, l=12, o=15]
    result = ds.parse_transcript("hello")
    assert result == [
        LABELS_MAP["h"],
        LABELS_MAP["e"],
        LABELS_MAP["l"],
        LABELS_MAP["l"],
        LABELS_MAP["o"],
    ]


def test_dataset_parse_transcript_drops_unknown_chars(manifest_csv, audio_conf):
    ds = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=str(manifest_csv),
        labels=LABELS_MAP,
        normalize=False,
    )
    # digits are not in LABELS_MAP -> filtered out
    result = ds.parse_transcript("a1b")
    assert result == [LABELS_MAP["a"], LABELS_MAP["b"]]


def test_dataset_both_items_accessible(manifest_csv, audio_conf):
    ds = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=str(manifest_csv),
        labels=LABELS_MAP,
        normalize=False,
    )
    for i in range(len(ds)):
        spect, transcript = ds[i]
        assert spect.shape[0] == EXPECTED_FREQ_BINS
        assert len(transcript) > 0


# ── AudioDataLoader + _collate_fn ────────────────────────────────────────────


def test_audio_data_loader_collate(manifest_csv, audio_conf):
    """AudioDataLoader sets collate_fn to _collate_fn; test via direct collation."""
    ds = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=str(manifest_csv),
        labels=LABELS_MAP,
        normalize=False,
    )
    loader = AudioDataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    inputs, targets, input_percentages, target_sizes = batch
    assert inputs.ndim == 4  # (B, 1, freq, time)
    assert inputs.shape[1] == 1
    assert inputs.shape[2] == EXPECTED_FREQ_BINS
    assert len(targets) > 0
    assert input_percentages.shape[0] == 2
    assert target_sizes.shape[0] == 2


def test_collate_fn_output_shape():
    # Build two fake (spect, transcript) samples of different lengths
    spect1 = torch.ones(EXPECTED_FREQ_BINS, 10)
    spect2 = torch.ones(EXPECTED_FREQ_BINS, 20)
    transcript1 = [1, 2, 3]
    transcript2 = [4, 5]
    batch = [(spect1, transcript1), (spect2, transcript2)]
    inputs, targets, input_percentages, target_sizes = _collate_fn(batch)
    assert inputs.shape == (2, 1, EXPECTED_FREQ_BINS, 20)
    assert targets.tolist() == [4, 5, 1, 2, 3]  # sorted by descending length
    assert input_percentages[0].item() == pytest.approx(1.0)
    assert input_percentages[1].item() == pytest.approx(0.5)
    assert target_sizes.tolist() == [2, 3]


def test_collate_fn_single_sample():
    spect = torch.ones(EXPECTED_FREQ_BINS, 15)
    transcript = [1, 2]
    inputs, targets, input_percentages, target_sizes = _collate_fn([(spect, transcript)])
    assert inputs.shape == (1, 1, EXPECTED_FREQ_BINS, 15)
    assert input_percentages[0].item() == pytest.approx(1.0)
    assert target_sizes[0].item() == 2


# ── check_loss ───────────────────────────────────────────────────────────────


def test_check_loss_valid():
    loss = torch.tensor(0.5)
    valid, error = check_loss(loss, 0.5)
    assert valid is True
    assert error == ""


def test_check_loss_inf():
    loss = torch.tensor(float("inf"))
    valid, error = check_loss(loss, float("inf"))
    assert valid is False
    assert "inf" in error.lower()


def test_check_loss_negative():
    loss = torch.tensor(-1.0)
    valid, error = check_loss(loss, -1.0)
    assert valid is False
    assert "negative" in error.lower()


def test_check_loss_nan():
    loss = torch.tensor(float("nan"))
    valid, error = check_loss(loss, 0.0)
    assert valid is False
    assert "nan" in error.lower()


# ── to_np ────────────────────────────────────────────────────────────────────


def test_to_np_converts_tensor():
    import numpy as np

    t = torch.tensor([1.0, 2.0, 3.0])
    arr = to_np(t)
    assert isinstance(arr, np.ndarray)
    assert list(arr) == pytest.approx([1.0, 2.0, 3.0])


# ── SequenceWise ─────────────────────────────────────────────────────────────


def test_sequence_wise_forward():
    linear = nn.Linear(8, 4)
    module = SequenceWise(linear)
    # Input: (T, N, H) = (5, 2, 8)
    x = torch.randn(5, 2, 8)
    out = module(x)
    assert out.shape == (5, 2, 4)


def test_sequence_wise_repr():
    linear = nn.Linear(4, 2)
    module = SequenceWise(linear)
    r = repr(module)
    assert "SequenceWise" in r


# ── InferenceBatchSoftmax ────────────────────────────────────────────────────


def test_inference_batch_softmax_eval_mode():
    module = InferenceBatchSoftmax()
    module.eval()
    x = torch.randn(3, 5)
    out = module(x)
    # In eval mode, should return softmax probabilities summing to 1 per row
    assert out.shape == (3, 5)
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_inference_batch_softmax_train_mode():
    module = InferenceBatchSoftmax()
    module.train()
    x = torch.randn(3, 5)
    out = module(x)
    # In train mode, returns input unchanged
    assert torch.equal(out, x)


# ── Lookahead ────────────────────────────────────────────────────────────────


def test_lookahead_forward_shape():
    module = Lookahead(n_features=8, context=5)
    # Input: (T, N, H) = (10, 2, 8)
    x = torch.randn(10, 2, 8)
    out = module(x)
    assert out.shape == (10, 2, 8)


def test_lookahead_repr():
    module = Lookahead(n_features=4, context=3)
    r = repr(module)
    assert "Lookahead" in r
    assert "n_features=4" in r
    assert "context=3" in r


# ── BatchRNN ──────────────────────────────────────────────────────────────────


def test_batch_rnn_forward():
    rnn_layer = BatchRNN(
        input_size=8,
        hidden_size=16,
        rnn_type=nn.GRU,
        bidirectional=False,
        batch_norm=True,
    )
    rnn_layer.eval()
    # Input: (T, N, H) = (5, 2, 8)
    x = torch.randn(5, 2, 8)
    output_lengths = torch.tensor([5, 5])
    out = rnn_layer(x, output_lengths)
    assert out.shape == (5, 2, 16)


def test_batch_rnn_bidirectional():
    rnn_layer = BatchRNN(
        input_size=8,
        hidden_size=16,
        rnn_type=nn.GRU,
        bidirectional=True,
        batch_norm=False,
    )
    rnn_layer.eval()
    x = torch.randn(5, 2, 8)
    output_lengths = torch.tensor([5, 5])
    out = rnn_layer(x, output_lengths)
    # Bidirectional sums the two directions: output is hidden_size (not 2*hidden_size)
    assert out.shape == (5, 2, 16)


# ── MaskConv ─────────────────────────────────────────────────────────────────


def test_mask_conv_forward():
    conv = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
    )
    module = MaskConv(conv)
    # Input: (B, C, D, T) = (2, 1, 4, 10)
    x = torch.randn(2, 1, 4, 10)
    lengths = torch.tensor([10, 8])
    out, out_lengths = module(x, lengths)
    assert out.shape == (2, 8, 4, 10)
    assert torch.equal(out_lengths, lengths)


# ── DeepSpeech model ─────────────────────────────────────────────────────────


# Number of labels the CSV produces: 26 letters (a–z only; no space in the fixture)
_NUM_LABELS = 26


def test_deepspeech_init(audio_conf, label_csv):
    """DeepSpeech can be instantiated with a minimal label file and tiny RNN."""
    model = DeepSpeech(
        audio_conf=audio_conf,
        decoder=None,
        label_path=str(label_csv),
        rnn_type="nn.GRU",
        rnn_hidden_size=16,
        rnn_hidden_layers=1,
        bidirectional=True,
    )
    assert model.num_classes == _NUM_LABELS
    assert model.sample_rate == 16_000


def test_deepspeech_get_seq_lens(audio_conf, label_csv):
    """get_seq_lens returns a 1D tensor of output lengths from conv layers."""
    model = DeepSpeech(
        audio_conf=audio_conf,
        decoder=None,
        label_path=str(label_csv),
        rnn_type="nn.GRU",
        rnn_hidden_size=16,
        rnn_hidden_layers=1,
        bidirectional=True,
    )
    lengths = torch.tensor([100, 80, 60])
    out_lens = model.get_seq_lens(lengths)
    assert out_lens.shape == (3,)
    assert (out_lens > 0).all()


def test_deepspeech_forward(audio_conf, label_csv):
    """DeepSpeech.forward produces output of shape (B, T_out, num_classes)."""
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
    # Input: (B, 1, freq, time) = (1, 1, 161, 40)
    x = torch.randn(1, 1, EXPECTED_FREQ_BINS, 40)
    lengths = torch.tensor([40])
    out, out_lengths = model.forward(x, lengths)
    assert out.ndim == 3  # (B, T_out, num_classes)
    assert out.shape[0] == 1
    assert out.shape[2] == _NUM_LABELS


# ── JSUTDataset static helpers ───────────────────────────────────────────────


def test_jsut_clean_text_strips_japanese_punctuation():
    result = JSUTDataset.clean_text(["JSUT001", "こんにちは。\n"])
    assert result == ("JSUT001", "こんにちは")


def test_jsut_clean_text_removes_spaces_and_commas():
    result = JSUTDataset.clean_text(["JSUT002", "テスト、 テスト\n"])
    assert result == ("JSUT002", "テストテスト")


# ── DeepSpeech unidirectional path (Lookahead) ───────────────────────────────


def test_deepspeech_unidirectional_forward(audio_conf, label_csv):
    """Unidirectional DeepSpeech activates the Lookahead layer."""
    model = DeepSpeech(
        audio_conf=audio_conf,
        decoder=None,
        label_path=str(label_csv),
        rnn_type="nn.GRU",
        rnn_hidden_size=16,
        rnn_hidden_layers=1,
        bidirectional=False,
        context=3,
    )
    model.eval()
    x = torch.randn(1, 1, EXPECTED_FREQ_BINS, 40)
    lengths = torch.tensor([40])
    out, out_lengths = model.forward(x, lengths)
    assert out.ndim == 3
    assert out.shape[2] == _NUM_LABELS


# ── TensorBoardLogger ────────────────────────────────────────────────────────


def test_tensorboard_logger_init(tmp_path):
    logger = TensorBoardLogger(
        id="test_run",
        log_dir=str(tmp_path / "tb_logs"),
        log_params=False,
    )
    assert logger.id == "test_run"
    assert logger.log_params is False


def test_tensorboard_logger_load_previous_values(tmp_path):
    import torch

    logger = TensorBoardLogger(
        id="test_run",
        log_dir=str(tmp_path / "tb_logs2"),
        log_params=False,
    )
    values = {
        "loss_results": torch.tensor([0.5, 0.4, 0.3]),
        "wer_results": torch.tensor([0.9, 0.8, 0.7]),
        "cer_results": torch.tensor([0.8, 0.7, 0.6]),
    }
    # Should not raise
    logger.load_previous_values(start_epoch=2, values=values)


def test_tensorboard_logger_update(tmp_path):
    import torch

    logger = TensorBoardLogger(
        id="test_run",
        log_dir=str(tmp_path / "tb_logs3"),
        log_params=False,
    )
    values = {
        "loss_results": torch.tensor([0.5, 0.4, 0.3]),
        "wer_results": torch.tensor([0.9, 0.8, 0.7]),
        "cer_results": torch.tensor([0.8, 0.7, 0.6]),
    }
    # Should not raise
    logger.update(epoch=0, values=values)


# ── LibriSpeechDataset ───────────────────────────────────────────────────────


def test_librispeech_dataset_init():
    ds = LibriSpeechDataset(fq=16000)
    assert ds._fq == 16000
    assert ds._df is None


def test_librispeech_dataset_default_fq():
    ds = LibriSpeechDataset()
    assert ds._fq == 16000


# ── WAVConverter ─────────────────────────────────────────────────────────────


def test_wav_converter_init(tmp_path):
    src = str(tmp_path)
    dst = str(tmp_path / "out")
    conv = WAVConverter(src=src, dst=dst, fq=16000, overwrite=False)
    assert conv._fq == 16000
    assert conv._overwrite is False


def test_wav_converter_fq(wav_16k):
    """WAVConverter.fq reads sample rate from a real WAV file."""
    fq = WAVConverter.fq(str(wav_16k))
    assert fq == 16000


# ── DeepSpeech.__call__ (inference loop) ─────────────────────────────────────


def test_deepspeech_call_with_loader(audio_conf, label_csv):
    """DeepSpeech.__call__ runs the inference loop and returns WER, CER, output."""
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

    # Build a minimal fake loader: one batch with shape matching model expectations
    # Input: (B=1, 1, freq, time=40), targets: flat int tensor, percentages, sizes
    inputs = torch.randn(1, 1, EXPECTED_FREQ_BINS, 40)
    targets = torch.IntTensor([1, 2, 3])  # 3 chars
    input_percentages = torch.FloatTensor([1.0])
    target_sizes = torch.IntTensor([3])

    loader = [(inputs, targets, input_percentages, target_sizes)]

    wer, cer, output_data = model(loader=loader, device="cpu")
    assert isinstance(wer, float)
    assert isinstance(cer, float)
    assert isinstance(output_data, list)
