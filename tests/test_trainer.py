import pytest
import torch
from torch.nn import CTCLoss

from asr_deepspeech.checkpoint import load_checkpoint
from asr_deepspeech.data.synthetic import synthetic_loader
from asr_deepspeech.modules.deepspeech import DeepSpeech
from asr_deepspeech.trainers import DeepSpeechTrainer


def _trainer(tmp_path, label_csv, audio_conf, **kw):
    torch.manual_seed(0)
    model = DeepSpeech(
        audio_conf=audio_conf,
        decoder=None,
        label_path=str(label_csv),
        rnn_type="nn.GRU",
        rnn_hidden_size=16,
        rnn_hidden_layers=1,
        bidirectional=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
    args = dict(
        epochs=3,
        model_path=str(tmp_path / "best.pth"),
        device="cpu",
        device_test="cpu",
        mixed_precision=False,
        seed=0,
    )
    args.update(kw)
    return DeepSpeechTrainer(
        model, CTCLoss(reduction="sum", zero_infinity=True), opt, sched, **args
    )


def _loaders():
    return synthetic_loader(8, 4, seed=1), synthetic_loader(8, 4, seed=2)


@pytest.mark.parametrize("runtime", ["vanilla", "sakura"])
def test_trains_and_writes_best(tmp_path, label_csv, audio_conf, runtime):
    t = _trainer(tmp_path, label_csv, audio_conf, runtime=runtime)
    metrics = t.run(*_loaders())
    assert len(metrics.history) == 3
    assert all("cer" in r and "train_loss" in r for r in metrics.history)
    assert metrics.best_cer is not None
    best = load_checkpoint(tmp_path / "best.pth")
    assert best["epoch"] == metrics.best_epoch
    assert len(t.epoch_seconds) == 3


def test_resume_continues_from_last_epoch(tmp_path, label_csv, audio_conf):
    _trainer(tmp_path, label_csv, audio_conf, epochs=2, runtime="sakura").run(*_loaders())
    t = _trainer(tmp_path, label_csv, audio_conf, epochs=4, runtime="sakura")
    metrics = t.run(*_loaders())
    assert t.start_epoch == 2
    assert [r["epoch"] for r in metrics.history][-2:] == [2, 3]
    assert len(t.epoch_seconds) == 2


def test_sakura_and_vanilla_agree(tmp_path, label_csv, audio_conf):
    a = _trainer(tmp_path / "a", label_csv, audio_conf, runtime="vanilla").run(*_loaders())
    b = _trainer(tmp_path / "b", label_csv, audio_conf, runtime="sakura", async_eval=False).run(
        *_loaders()
    )
    assert a.best_cer == pytest.approx(b.best_cer, rel=1e-4)


def test_rejects_bad_runtime(tmp_path, label_csv, audio_conf):
    with pytest.raises(ValueError):
        _trainer(tmp_path, label_csv, audio_conf, runtime="nope")


def test_all_invalid_batches_raise(tmp_path, label_csv, audio_conf):
    t = _trainer(tmp_path, label_csv, audio_conf, runtime="vanilla")
    t.criterion = lambda *a, **k: torch.tensor(float("nan"), requires_grad=True)
    with pytest.raises(RuntimeError, match="invalid loss"):
        t.run(*_loaders())
