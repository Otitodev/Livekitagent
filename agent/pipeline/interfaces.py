from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class AudioFrame:
    data: bytes
    sample_rate_hz: int


class SpeechToText(Protocol):
    def feed(self, frame: AudioFrame) -> None:
        """Accept a chunk of audio for transcription."""

    def finalize(self) -> str:
        """Return the transcript for the audio received so far."""


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str:
        """Return a model response for the given prompt."""


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> AudioFrame:
        """Return an audio frame containing synthesized speech."""


class AudioTrack(Protocol):
    def on_frame(self, handler: Callable[[AudioFrame], None]) -> None:
        """Register a handler for audio frames coming from the track."""

    def on_end(self, handler: Callable[[], None]) -> None:
        """Register a handler fired when the track finishes."""


class LiveKitRoom(Protocol):
    def on_audio_track(self, handler: Callable[[AudioTrack], None]) -> None:
        """Register a handler for newly subscribed audio tracks."""

    def publish_audio(self, frame: AudioFrame) -> None:
        """Publish synthesized audio back to the room."""
