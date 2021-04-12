from torch.utils.data import Dataset
from asr_deepspeech.data.parsers import SpectrogramParser
import json
from tqdm import tqdm


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self,
                 audio_conf,
                 manifest_filepath,
                 labels,
                 normalize=False,
                 spec_augment=False):
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
        data = dict([(k, v) for k, v in json.load(open(manifest_filepath, "r")).items()])
        ids = list(data.values())
        self.ids = ids
        self.size = len(self.ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self._cache = False
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, audio_conf.speed_volume_perturb, spec_augment)
        if self._cache:
            self.specs = dict([(v["audio_filepath"], self.parse_audio(v["audio_filepath"])) for key, v in tqdm(data.items(), total=len(data), desc="Loading audio")])
            self.transcripts = dict([(v["text"], self.parse_transcript(transcript=v["text"])) for key, v in tqdm(data.items(), total=len(data), desc="Loading transcripts")])

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript = sample["audio_filepath"], sample["text"]
        if self._cache:
            spec, transcript = self.specs[audio_path], self.transcripts[transcript]
        else:
            spec, transcript = self.parse_audio(audio_path), self.parse_transcript(transcript)

        return spec, transcript

    def parse_transcript(self, transcript):
        transcript = transcript.replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size
