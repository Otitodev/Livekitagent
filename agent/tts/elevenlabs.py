"""Streaming ElevenLabs Text-to-Speech implementation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


@dataclass
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    model_id: str = "eleven_turbo_v2"
    output_format: str = "mp3_44100_128"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True


class StreamingElevenLabsTTS:
    """
    Streaming Text-to-Speech using ElevenLabs API.

    Streams audio chunks as they are generated, enabling
    low-latency audio playback. Supports cancellation for
    barge-in handling.
    """

    def __init__(self, config: ElevenLabsConfig):
        self.config = config
        self._client = httpx.AsyncClient(timeout=30.0)
        self._cancel_event: Optional[asyncio.Event] = None
        self._is_synthesizing = False

    @property
    def is_synthesizing(self) -> bool:
        """Whether TTS is currently generating audio."""
        return self._is_synthesizing

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    async def synthesize_stream(
        self,
        text: str,
        chunk_size: int = 4096,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio synthesis for the given text.

        Args:
            text: Text to synthesize
            chunk_size: Size of audio chunks to yield

        Yields:
            Audio bytes (MP3 format by default)
        """
        if not text.strip():
            return

        self._cancel_event = asyncio.Event()
        self._is_synthesizing = True

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/"
            f"{self.config.voice_id}/stream"
        )

        headers = {
            "xi-api-key": self.config.api_key,
            "Content-Type": "application/json",
        }

        body = {
            "text": text,
            "model_id": self.config.model_id,
            "output_format": self.config.output_format,
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "style": self.config.style,
                "use_speaker_boost": self.config.use_speaker_boost,
            },
        }

        try:
            async with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=body,
            ) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes(chunk_size):
                    if self._cancel_event.is_set():
                        logger.debug("TTS synthesis cancelled")
                        break
                    yield chunk

        except httpx.HTTPStatusError as e:
            logger.error(f"ElevenLabs API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise
        finally:
            self._is_synthesizing = False
            self._cancel_event = None

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize audio and return complete bytes.

        Use synthesize_stream() for streaming playback.
        """
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)

    def cancel(self) -> None:
        """
        Cancel ongoing synthesis (for barge-in handling).

        Safe to call even if not synthesizing.
        """
        if self._cancel_event:
            self._cancel_event.set()
            logger.debug("TTS cancellation requested")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class TTSQueue:
    """
    Queue for managing TTS synthesis requests.

    Handles sentence-by-sentence synthesis with
    proper ordering and cancellation support.
    """

    def __init__(self, tts: StreamingElevenLabsTTS):
        self.tts = tts
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._audio_callback: Optional[callable] = None
        self._running = False

    async def start(self, on_audio: callable) -> None:
        """
        Start processing the TTS queue.

        Args:
            on_audio: Callback receiving audio chunks
        """
        self._audio_callback = on_audio
        self._running = True
        self._processing_task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop queue processing."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, text: str) -> None:
        """Add text to the synthesis queue."""
        await self._queue.put(text)

    def clear(self) -> None:
        """Clear pending items and cancel current synthesis."""
        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cancel current synthesis
        self.tts.cancel()

    async def _process_loop(self) -> None:
        """Background task to process queued sentences."""
        while self._running:
            try:
                text = await asyncio.wait_for(self._queue.get(), timeout=1.0)

                async for chunk in self.tts.synthesize_stream(text):
                    if self._audio_callback:
                        await self._audio_callback(chunk)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS queue error: {e}")
