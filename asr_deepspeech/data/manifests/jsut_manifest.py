from gnutools.fs import listfiles, name
import wave
import contextlib
from tqdm import tqdm
import json

class JSUTManifest(dict):
    def __init__(self, root):
        super(JSUTManifest, self).__init__()
        self.__root=root
        self.__labels = set()


    def build(self):
        wav_files = listfiles(self.__root, [".wav"])
        daudio = dict([(name(file), file) for file in wav_files])
        transcript_files = listfiles(self.__root, ["transcript_utf8.txt"])
        dtext = {}
        for transcript_file in transcript_files:
            records = dict([l.rsplit()[0].split(":") for l in open(transcript_file, "r").readlines()])
            dtext.update(records)
        assert len(daudio)==len(dtext)
        for id, text in tqdm(dtext.items(), total=len(dtext), desc="Building the manifest"):
            d = {
                "audio_filepath": daudio[id],
                "duration": self.__duration(daudio[id]),
                "text": dtext[id]
            }
            self.__labels = self.__labels.union(set(dtext[id]))
            self.setdefault(id, d)

    def __duration(self, path):
        with contextlib.closing(wave.open(path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration

    def export(self, manifest, labels=None):
        json.dump(self, open(manifest, "w"), indent=4, ensure_ascii=False)
        json.dump(list(self.__labels), open(labels, "w"), indent=4, ensure_ascii=False) if labels is not None else None

if __name__=="__main__":
    manifest = JSUTManifest(root="/mnt/.cdata/ASR/ja/raw/CLEAN/JSUT/jsut_ver1.1")
    manifest.build()
    manifest.export(manifest="jsut_ver1.1.json", labels="labels_jsut.json")
