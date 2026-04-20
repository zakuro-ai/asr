from asr_deepspeech.data.dataset import SpectrogramDataset
from asr_deepspeech.data.loaders import AudioDataLoader
from asr_deepspeech.data.samplers import BucketingSampler


def get_loader(
    audio_conf,
    labels,
    manifest: str,
    batch_size: int,
    num_workers: int,
    cache_dir: str | None = None,
):
    dataset = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=manifest,
        labels=labels,
        normalize=True,
        spec_augment=audio_conf.spec_augment,
        cache_dir=cache_dir,
    )
    sampler = BucketingSampler(dataset, batch_size=batch_size)
    loader = AudioDataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=sampler,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    sampler.shuffle()
    return loader, sampler
