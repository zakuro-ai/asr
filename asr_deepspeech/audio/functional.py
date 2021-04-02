import contextlib, wave


def duration(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def fq(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        rate = f.getframerate()
        return rate
