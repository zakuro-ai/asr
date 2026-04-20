import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from asr_deepspeech.data.parsers import SpectrogramParser


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
        self.df = pd.read_csv(manifest_filepath)
        self.size = len(self.df)
        if isinstance(labels, str):
            labels = {v: k for k, v in pd.read_csv(labels).to_dict()["label"].items()}
        self.labels_map = labels
        self.caching = caching
        super().__init__(audio_conf, normalize, audio_conf.speed_volume_perturb, spec_augment)
        if self.caching:
            self.specs = {
                r.audio_filepath: self.parse_audio(r.audio_filepath)
                for _, r in tqdm(self.df.iterrows(), total=len(self.df), desc="Caching audio")
            }
            self.transcripts = {
                r.text: self.parse_transcript(transcript=r.text)
                for _, r in tqdm(self.df.iterrows(), total=len(self.df), desc="Caching transcripts")
            }

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        audio_path, transcript = sample.audio_filepath, sample.text
        if self.caching:
            return self.specs[audio_path], self.transcripts[transcript]
        return self.parse_audio(audio_path), self.parse_transcript(transcript)

    def parse_transcript(self, transcript):
        return list(filter(None, [self.labels_map.get(x) for x in transcript.replace("\n", "")]))

    def __len__(self):
        return self.size
