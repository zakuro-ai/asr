from asr_deepspeech.data.dataset import SpectrogramDataset
from asr_deepspeech.data.loaders import AudioDataLoader
from asr_deepspeech.data.samplers import BucketingSampler


def get_loader(audio_conf, labels, manifest, batch_size, num_workers, caching=False):
    dataset = SpectrogramDataset(
        audio_conf=audio_conf,
        manifest_filepath=manifest,
        labels=labels,
        normalize=True,
        spec_augment=audio_conf.spec_augment,
        caching=caching,
    )
    sampler = BucketingSampler(dataset, batch_size=batch_size)
    loader = AudioDataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=sampler
        # pin_memory=True
    )

    sampler.shuffle()
    return loader, sampler
