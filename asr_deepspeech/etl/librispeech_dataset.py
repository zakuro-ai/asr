import os
import tarfile
import urllib.request

import pandas as pd
import soundfile as sf
from gnutools.fs import listfiles

# Public OpenSLR mirrors. "mini" LibriSpeech (resource 31) is small and ideal for
# smoke tests; the full corpus lives under resource 12.
OPENSLR_URLS = {
    # mini LibriSpeech — https://www.openslr.org/31/
    "dev-clean-2": "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
    "train-clean-5": "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
    # full LibriSpeech — https://www.openslr.org/12/
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
}


class LibriSpeechDataset:
    """Build bronze-layer manifest CSVs from a LibriSpeech-style directory.

    Directory layout expected::

        root/
          speaker_id/
            chapter_id/
              <id>.flac
              <id>.trans.txt   # space-separated: utt_id TRANSCRIPT...
    """

    def __init__(self, fq: int = 16000):
        self._fq = fq
        self._df = None

    @classmethod
    def url_for(cls, subset: str) -> str:
        """Return the OpenSLR download URL for a known LibriSpeech subset."""
        try:
            return OPENSLR_URLS[subset]
        except KeyError as exc:
            raise ValueError(
                f"Unknown LibriSpeech subset {subset!r}; known: {sorted(OPENSLR_URLS)}"
            ) from exc

    def download(self, landing: str, subset: str = "dev-clean-2", url: str = None) -> str:
        """Download and extract a LibriSpeech subset from OpenSLR into ``landing``.

        Idempotent: if ``landing/LibriSpeech/<subset>`` already exists, returns
        immediately without touching the network. Uses only the standard library
        (``urllib`` + ``tarfile``) so the project stays dependency-light.
        """
        url = url or self.url_for(subset)
        os.makedirs(landing, exist_ok=True)
        extracted = os.path.join(landing, "LibriSpeech", subset)
        if os.path.isdir(extracted):
            return landing
        archive = os.path.join(landing, f"{subset}.tar.gz")
        if not os.path.exists(archive):
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r:gz") as tar:
            try:
                tar.extractall(landing, filter="data")  # py>=3.12 / backported
            except TypeError:
                tar.extractall(landing)  # older python without the `filter` kwarg
        return landing

    def run(self, landing: str, bronze: str) -> "LibriSpeechDataset":
        records = []
        trans_files = listfiles(landing, [".trans.txt"])
        for trans_file in trans_files:
            chapter_dir = os.path.dirname(trans_file)
            utterances = {}
            for line in open(trans_file, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                utt_id, transcript = line.split(" ", 1)
                utterances[utt_id] = transcript.lower()

            for utt_id, text in utterances.items():
                # LibriSpeech uses FLAC
                src_path = os.path.join(chapter_dir, f"{utt_id}.flac")
                if not os.path.exists(src_path):
                    continue
                info = sf.info(src_path)
                dur = info.frames / info.samplerate
                # Store the flac path directly — load_audio now supports soundfile
                rel = os.path.relpath(src_path, landing)
                dst_path = os.path.join(bronze, rel)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                if not os.path.exists(dst_path):
                    os.symlink(os.path.abspath(src_path), dst_path)
                records.append((dst_path, dur, self._fq, text, len(text)))

        self._df = pd.DataFrame.from_records(
            records,
            columns=["audio_filepath", "duration", "fq", "text", "text_size"],
        )
        return self

    def filter_duration(self, start: float = 1.0, stop: float = 20.0) -> pd.DataFrame:
        return self._df[(self._df["duration"] >= start) & (self._df["duration"] <= stop)]

    def export_labels(self, output_file: str) -> None:
        chars = set()
        for text in self._df["text"]:
            chars.update(set(text))
        pd.DataFrame.from_records(sorted(chars), columns=["label"]).to_csv(output_file, index=False)
