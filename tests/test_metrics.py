from asr_deepspeech.metrics import asr_metrics


def test_asr_metrics_structure():
    m = asr_metrics()
    assert hasattr(m, "train")
    assert hasattr(m, "test")
    for split in (m.train, m.test):
        assert hasattr(split, "current")
        assert hasattr(split, "best")
        for ns in (split.current, split.best):
            assert ns.loss == 0.0
            assert ns.wer is None
            assert ns.cer is None


def test_asr_metrics_mutable():
    m = asr_metrics()
    m.train.current.loss = 1.5
    assert m.train.current.loss == 1.5
    # best should be independent
    assert m.train.best.loss == 0.0
