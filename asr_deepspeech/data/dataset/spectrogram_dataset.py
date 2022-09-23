from torch.utils.data import Dataset
from asr_deepspeech.data.parsers import SpectrogramParser
import json
from tqdm import tqdm
import pandas as pd


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(
        self,
        audio_conf,
        manifest_filepath,
        labels,
        normalize=False,
        spec_augment=False,
        caching=False,
    ):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        self.df = pd.read_csv(manifest_filepath)
        self.size = len(self.df)
        if type(labels) == str:
            labels = dict([(v, k) for k, v in pd.read_csv(labels).to_dict()["label"].items()])
        self.labels_map = labels
        self.caching = caching
        super(SpectrogramDataset, self).__init__(
            audio_conf, normalize, audio_conf.speed_volume_perturb, spec_augment
        )
        if self.caching:
            self.specs = dict(
                [
                    (r.audio_filepath, self.parse_audio(r.audio_filepath))
                    for k, r in tqdm(
                        self.df.iterrows(), total=len(self.df), desc="Loading audio"
                    )
                ]
            )
            self.transcripts = dict(
                [
                    (r.text, self.parse_transcript(transcript=r.text))
                    for k, r in tqdm(
                        self.df.iterrows(),
                        total=len(self.df),
                        desc="Loading transcripts",
                    )
                ]
            )

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        audio_path, transcript = sample.audio_filepath, sample.text
        if self.caching:
            spec, transcript = self.specs[audio_path], self.transcripts[transcript]
        else:
            spec, transcript = self.parse_audio(audio_path), self.parse_transcript(
                transcript
            )

        return spec, transcript

    def parse_transcript(self, transcript):
        transcript = transcript.replace("\n", "")
        transcript = list(
            filter(None, [self.labels_map.get(x) for x in list(transcript)])
        )
        return transcript

    def __len__(self):
        return self.size


if __name__ == "__main__":

    from asr_deepspeech import cfg

    dataset = SpectrogramDataset(
        audio_conf=cfg.model.audio_conf,
        manifest_filepath="/workspace/filestore/gold/jsut_6-10/test.csv",
        labels="/workspace/filestore/gold/jsut_6-10/labels.csv",
    )
    # sampler = BucketingSampler(dataset,
    #                             batch_size=batch_size)
    # loader = AudioDataLoader(dataset,
    #                             num_workers=num_workers,
    #                             batch_sampler=sampler
    #                             # pin_memory=True
    #                             )

    # sampler.shuffle()
