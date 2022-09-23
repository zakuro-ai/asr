from asr_deepspeech.audio import WAVConverter
import pandas as pd
from gnutools.fs import listfiles, name


class JSUTDataset:
    def __init__(self, fq=None, overwrite=False):
        self._overwrite = overwrite
        self._fq = fq
        self._df = None

    def run(self, landing, bronze):
        # Process text
        transcripts = listfiles(landing, ["transcript_utf8.txt"])
        texts = dict()
        for file in transcripts:
            d = dict([self.clean_text(l.split(":")) for l in open(file, "r")])
            texts.update(d)

        df = pd.DataFrame.from_records(
            data=[
                (file, duration, self._fq, texts[name(file)], len(texts[name(file)]))
                for file, duration in WAVConverter(landing, bronze, self._fq).run()
            ],
            columns=["audio_filepath", "duration", "fq", "text", "text_size"],
        )

        self._df = df
        return self

    @staticmethod
    def clean_text(t):
        k, v = t
        v = v.replace(" ", "").replace("\n", "").replace("、", "").replace("。", "")
        return (k, v)

    def filter_duration(self, start=1, stop=5):
        return self._df[
            (self._df["duration"] >= start) & (self._df["duration"] <= stop)
        ]

    def export_labels(self, output_file):
        s = set()
        for r in self._df["text"]:
            s = s.union(set(r))
        pd.DataFrame.from_records(list(s), columns=["label"]).to_csv(
            output_file, index=False
        )
