import os
import pandas as pd
import soundfile as sf
from gnutools.fs import listfiles


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
                if not os.path.exists(dst_path) and not os.path.islink(dst_path):
                    os.symlink(os.path.abspath(src_path), dst_path)
                records.append((dst_path, dur, self._fq, text, len(text)))

        self._df = pd.DataFrame.from_records(
            records,
            columns=["audio_filepath", "duration", "fq", "text", "text_size"],
        )
        return self

    def filter_duration(self, start: float = 1.0, stop: float = 20.0) -> pd.DataFrame:
        return self._df[
            (self._df["duration"] >= start) & (self._df["duration"] <= stop)
        ]

    def export_labels(self, output_file: str) -> None:
        chars = set()
        for text in self._df["text"]:
            chars.update(set(text))
        pd.DataFrame.from_records(sorted(chars), columns=["label"]).to_csv(
            output_file, index=False
        )
