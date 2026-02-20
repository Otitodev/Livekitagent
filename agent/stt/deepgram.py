"""Streaming Deepgram Speech-to-Text implementation using WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


@dataclass
class DeepgramConfig:
    api_key: str
    model: str = "nova-2"
    language: str = "en"
    sample_rate: int = 16000
    encoding: str = "linear16"
    channels: int = 1
    interim_results: bool = True
    punctuate: bool = True
    endpointing: int = 300  # ms of silence to mark end of utterance


TranscriptCallback = Callable[[str, bool], None]  # (transcript, is_final)


class StreamingDeepgramSTT:
    """
    Streaming Speech-to-Text using Deepgram's WebSocket API.

    Connects to Deepgram, streams audio chunks, and receives transcripts
    in real-time with both interim (partial) and final results.
    """

    def __init__(self, config: DeepgramConfig):
        self.config = config
        self._ws: Optional[WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._transcript_callback: Optional[TranscriptCallback] = None
        self._connected = asyncio.Event()
        self._closing = False

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.open

    async def connect(self, on_transcript: TranscriptCallback) -> None:
        """
        Connect to Deepgram WebSocket and start receiving transcripts.

        Args:
            on_transcript: Callback called with (transcript_text, is_final)
        """
        if self.is_connected:
            logger.warning("Already connected to Deepgram")
            return

        self._transcript_callback = on_transcript
        self._closing = False

        # Build WebSocket URL with query parameters
        params = [
            f"model={self.config.model}",
            f"language={self.config.language}",
            f"sample_rate={self.config.sample_rate}",
            f"encoding={self.config.encoding}",
            f"channels={self.config.channels}",
            f"interim_results={str(self.config.interim_results).lower()}",
            f"punctuate={str(self.config.punctuate).lower()}",
            f"endpointing={self.config.endpointing}",
        ]
        url = f"wss://api.deepgram.com/v1/listen?{'&'.join(params)}"

        headers = {"Authorization": f"Token {self.config.api_key}"}

        try:
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            self._connected.set()
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info("Connected to Deepgram STT")
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            raise

    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send an audio chunk to Deepgram for transcription.

        Args:
            audio_chunk: Raw audio bytes (PCM 16-bit mono)
        """
        if not self.is_connected:
            logger.warning("Cannot send audio: not connected")
            return

        try:
            await self._ws.send(audio_chunk)
        except Exception as e:
            logger.error(f"Failed to send audio chunk: {e}")
            raise

    async def close(self) -> None:
        """Close the WebSocket connection gracefully."""
        self._closing = True

        if self._ws and self._ws.open:
            try:
                # Send close message to Deepgram
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing Deepgram connection: {e}")

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        self._ws = None
        self._connected.clear()
        logger.info("Disconnected from Deepgram STT")

    async def _receive_loop(self) -> None:
        """Background task to receive and process transcripts."""
        try:
            async for message in self._ws:
                if self._closing:
                    break

                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message[:100]}")
        except websockets.ConnectionClosed as e:
            if not self._closing:
                logger.warning(f"Deepgram connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in Deepgram receive loop: {e}")

    async def _handle_message(self, data: dict) -> None:
        """Process a message from Deepgram."""
        msg_type = data.get("type")

        if msg_type == "Results":
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])

            if alternatives:
                transcript = alternatives[0].get("transcript", "").strip()
                is_final = data.get("is_final", False)
                speech_final = data.get("speech_final", False)

                if transcript and self._transcript_callback:
                    # speech_final indicates end of utterance (user stopped speaking)
                    self._transcript_callback(transcript, is_final or speech_final)

        elif msg_type == "Metadata":
            logger.debug(f"Deepgram metadata: {data}")

        elif msg_type == "UtteranceEnd":
            # Deepgram detected end of speech
            logger.debug("Deepgram detected utterance end")

        elif msg_type == "Error":
            logger.error(f"Deepgram error: {data}")
