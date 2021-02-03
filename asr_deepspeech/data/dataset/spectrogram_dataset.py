import json
from torch.utils.data import Dataset
from asr_deepspeech.data.parsers import SpectrogramParser
from asr_deepspeech.vars import N

class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, speed_volume_perturb=False, spec_augment=False):
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
        with open(manifest_filepath) as f:
            lines = f.readlines()
            ids = []
            for k, line in enumerate(lines):
                try:
                    ids.append(json.loads(line))
                except:
                    print(manifest_filepath, k, '\n', line)
                    raise AssertionError
        self.ids = ids[:N]
        self.size = len(self.ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, speed_volume_perturb, spec_augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript = sample["audio_filepath"], sample["text"]
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript=transcript)
        return spect, transcript

    def parse_transcript(self, transcript):
        transcript = transcript.replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size
