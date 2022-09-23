from torch.utils.data import DataLoader
from tqdm import tqdm
from asr_deepspeech.functional import _collate_fn


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


if __name__ == "__main__":
    from asr_deepspeech import cfg
    from asr_deepspeech.data.loaders import get_loader

    # Init the loaders
    (train_loader, _), (test_loader, _) = get_loader(
        labels=cfg.model.label_path,
        audio_conf=cfg.model.audio_conf,
        manifest=cfg.loaders.train_manifest,
        batch_size=cfg.loaders.batch_size,
        num_workers=cfg.loaders.num_workers,
        caching=cfg.loaders.caching,
    ), get_loader(
        labels=cfg.model.label_path,
        audio_conf=cfg.model.audio_conf,
        manifest=cfg.loaders.val_manifest,
        batch_size=cfg.loaders.batch_size,
        num_workers=cfg.loaders.num_workers,
        caching=cfg.loaders.caching,
    )
    for inputs, targets, input_percentages, target_sizes in tqdm(
        train_loader, total=len(train_loader), desc="Processing"
    ):
        pass

    for inputs, targets, input_percentages, target_sizes in tqdm(
        test_loader, total=len(test_loader), desc="Processing"
    ):
        pass
