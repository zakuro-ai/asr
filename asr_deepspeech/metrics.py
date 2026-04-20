from types import SimpleNamespace


def _split():
    return SimpleNamespace(loss=0.0, wer=None, cer=None)


def asr_metrics():
    train = SimpleNamespace(current=_split(), best=_split())
    test = SimpleNamespace(current=_split(), best=_split())
    return SimpleNamespace(train=train, test=test)
