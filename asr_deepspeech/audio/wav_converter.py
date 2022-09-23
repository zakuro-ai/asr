import os 
from gnutools.concurrent import ProcessPoolExecutorBar
from gnutools.fs import listfiles, parent
import wave
from asr_deepspeech.audio import duration


class WAVConverter:
    def __init__(self, src, dst, fq, overwrite=False):
        self._src = os.path.realpath(src)
        self._dst = os.path.realpath(dst)
        self._fq = fq
        self._overwrite = overwrite
        
    @staticmethod
    def fq(file):
        with wave.open(file, "rb") as wave_file:
            return wave_file.getframerate()
    @staticmethod
    def main(wav_file, landing, bronze, overwrite):
        wav_output = wav_file.replace(landing, bronze)
        try:
            assert os.path.exists(wav_output) & (not overwrite)
            assert WAVConverter.fq(wav_output)==16000
        except:
            os.makedirs(parent(wav_output), exist_ok=True)
            os.system(f"nohup ffmpeg -i {wav_file} -ar 16000 {wav_output} -y >/dev/null 2>&1")
        finally:
            assert WAVConverter.fq(wav_output)==16000

        return (wav_output, duration(wav_output))
        
    def run(self):
        wav_files = [f for f in listfiles(self._src) if f.endswith(".wav")]
        bar = ProcessPoolExecutorBar()
        bar.submit([(WAVConverter.main, 
                     wav_file, 
                     self._src,
                     self._dst,
                     self._overwrite) for wav_file in wav_files])
        return bar._results
    
if __name__ == "__main__":
    landing = "/dbfs/FileStore/asr/landing/jsut_ver1.1"
    bronze = "/dbfs/FileStore/asr/bronze/jsut_ver1.1"
    WAVConverter(landing, bronze, 16000).run()
    