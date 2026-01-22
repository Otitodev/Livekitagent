from __future__ import annotations

from dataclasses import dataclass

from agent.pipeline.interfaces import AudioFrame, LanguageModel, SpeechToText, TextToSpeech


@dataclass
class EchoSpeechToText(SpeechToText):
    """A placeholder STT that echoes back a fixed transcript."""

    transcript: str = "hello from stt"

    def feed(self, frame: AudioFrame) -> None:
        _ = frame

    def finalize(self) -> str:
        return self.transcript


@dataclass
class EchoLanguageModel(LanguageModel):
    """A placeholder LLM that returns a canned response."""

    prefix: str = "llm reply: "

    def generate(self, prompt: str) -> str:
        return f"{self.prefix}{prompt}"


@dataclass
class ToneTextToSpeech(TextToSpeech):
    """A placeholder TTS that returns a dummy audio frame."""

    sample_rate_hz: int = 16000

    def synthesize(self, text: str) -> AudioFrame:
        encoded = text.encode("utf-8")
        return AudioFrame(data=encoded, sample_rate_hz=self.sample_rate_hz)
