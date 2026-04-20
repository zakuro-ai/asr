from torch.utils.data import DataLoader

from asr_deepspeech.functional import _collate_fn


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        # Inject the CTC collate function; caller controls pin_memory / workers.
        kwargs.setdefault("collate_fn", _collate_fn)
        super().__init__(*args, **kwargs)
