from gnutools.fs import listfiles, parent
import os
from tqdm import tqdm
import wave
def fq(file):
    with wave.open(file, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        assert frame_rate==16000
root = "/mnt/.cdata/ASR/ja/raw/CLEAN/JSUT/jsut_ver1.1"
files = [f for f in listfiles(root) if f.endswith(".wav")]
for wav_file in tqdm(files, total=len(files), desc="Converting"):
    wav_output = wav_file.replace('/raw/', '/processed/')
    os.makedirs(parent(wav_output), exist_ok=True)
    os.system(f"ffmpeg -i {wav_file} -ar 16000 {wav_output} -y")
    fq(wav_output)
