from abc import ABC, abstractmethod


class AudioParser(ABC):
    @abstractmethod
    def parse_audio(self, audio_path: str):
        """Return a spectrogram tensor for the given audio file."""

    @abstractmethod
    def parse_transcript(self, transcript: str):
        """Return an encoded label sequence for the given transcript string."""
