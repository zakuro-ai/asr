import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile as sf
from gnutools.fs import listfiles, parent
from tqdm import tqdm

from asr_deepspeech.audio import duration


class WAVConverter:
    """Convert audio files (WAV/FLAC/MP3/…) to 16 kHz mono WAV using ffmpeg."""

    EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")

    def __init__(self, src: str, dst: str, fq: int = 16000, overwrite: bool = False):
        self._src = os.path.realpath(src)
        self._dst = os.path.realpath(dst)
        self._fq = fq
        self._overwrite = overwrite

    @staticmethod
    def _sample_rate(path: str) -> int:
        return sf.info(path).samplerate

    @staticmethod
    def _convert(audio_file: str, src: str, dst: str, fq: int, overwrite: bool):
        rel = os.path.relpath(audio_file, src)
        wav_output = os.path.join(dst, os.path.splitext(rel)[0] + ".wav")
        already_ok = (
            os.path.exists(wav_output)
            and not overwrite
            and WAVConverter._sample_rate(wav_output) == fq
        )
        if not already_ok:
            os.makedirs(parent(wav_output), exist_ok=True)
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_file, "-ar", str(fq), "-ac", "1", wav_output],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for {audio_file}")
        if WAVConverter._sample_rate(wav_output) != fq:
            raise RuntimeError(f"Unexpected sample rate in {wav_output}")
        return wav_output, duration(wav_output)

    def run(self, max_workers: int = os.cpu_count()):
        audio_files = [
            f for f in listfiles(self._src)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ]
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(WAVConverter._convert, f, self._src, self._dst, self._fq, self._overwrite): f
                for f in audio_files
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting audio"):
                results.append(fut.result())
        return results
