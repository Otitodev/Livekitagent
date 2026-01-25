from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional
from urllib import request


@dataclass
class DeepgramConfig:
    api_key: str
    model: str = "nova-2"
    language: str = "en"


class DeepgramSTT:
    def __init__(self, config: DeepgramConfig) -> None:
        self.config = config

    def transcribe(self, audio_bytes: bytes, mimetype: str = "audio/wav") -> Optional[str]:
        url = (
            "https://api.deepgram.com/v1/listen"
            f"?model={self.config.model}&language={self.config.language}&smart_format=true"
        )
        headers = {
            "Authorization": f"Token {self.config.api_key}",
            "Content-Type": mimetype,
        }
        req = request.Request(url, data=audio_bytes, headers=headers, method="POST")
        with request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return _extract_deepgram_transcript(payload)


@dataclass
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    model_id: str = "eleven_turbo_v2"
    output_format: str = "mp3_44100_128"


class ElevenLabsTTS:
    def __init__(self, config: ElevenLabsConfig) -> None:
        self.config = config

    def synthesize(self, text: str) -> bytes:
        url = (
            "https://api.elevenlabs.io/v1/text-to-speech/"
            f"{self.config.voice_id}?output_format={self.config.output_format}"
        )
        headers = {
            "xi-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        body = json.dumps(
            {
                "text": text,
                "model_id": self.config.model_id,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
            }
        ).encode("utf-8")
        req = request.Request(url, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=30) as resp:
            return resp.read()


def _extract_deepgram_transcript(payload: dict[str, Any]) -> Optional[str]:
    results = payload.get("results", {})
    channels = results.get("channels", [])
    if not channels:
        return None
    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return None
    transcript = alternatives[0].get("transcript")
    return transcript or None
