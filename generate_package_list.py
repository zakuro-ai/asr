from gnutools.fs import path2modules, listfiles
import json
import os
def rename(f):
    return "asr_deepspeech." + f.split("/asr_deepspeech/")[1].replace("/", ".")

json.dump(path2modules("./asr_deepspeech"),
          open("packages.json", "w"),
          indent=4)


data_files = [f for f in listfiles("./asr_deepspeech") if "__data__" in f if os.path.isfile(f)]

json.dump({'' :[rename(f) for f in data_files]},
          open("package_data.json", "w"),
          indent=4)

