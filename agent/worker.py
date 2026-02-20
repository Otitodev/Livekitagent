"""
Production voice agent worker.

Implements the full streaming pipeline:
Audio → VAD → STT → LLM → TTS → Audio

With barge-in support and session persistence.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from livekit import rtc

from agent.audio.vad import SileroVAD, VADConfig, VADEvent
from agent.stt.deepgram import StreamingDeepgramSTT, DeepgramConfig
from agent.llm.openai import StreamingOpenAI, OpenAIConfig
from agent.tts.elevenlabs import StreamingElevenLabsTTS, ElevenLabsConfig
from agent.conversation.store import RedisSessionStore, InMemorySessionStore

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    # LiveKit
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    room_name: str
    identity: str = "agent"

    # Deepgram STT
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-2"

    # OpenAI LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ElevenLabs TTS
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_model_id: str = "eleven_turbo_v2"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Behavior
    end_of_turn_silence_ms: int = 800
    max_response_tokens: int = 300

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        """Load configuration from environment variables."""
        return cls(
            livekit_url=os.environ["LIVEKIT_URL"],
            livekit_api_key=os.environ["LIVEKIT_API_KEY"],
            livekit_api_secret=os.environ["LIVEKIT_API_SECRET"],
            room_name=os.getenv("LIVEKIT_ROOM", "agent-room"),
            identity=os.getenv("LIVEKIT_IDENTITY", "agent"),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            deepgram_model=os.getenv("DEEPGRAM_MODEL", "nova-2"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
            elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        )


class VoiceAgentWorker:
    """
    Production voice agent that handles a single room.

    Features:
    - Streaming STT via Deepgram WebSocket
    - Silero VAD for speech detection
    - Streaming LLM responses
    - Streaming TTS with barge-in support
    - Redis session persistence
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.room: Optional[rtc.Room] = None

        # Components (initialized on start)
        self.vad: Optional[SileroVAD] = None
        self.stt: Optional[StreamingDeepgramSTT] = None
        self.llm: Optional[StreamingOpenAI] = None
        self.tts: Optional[StreamingElevenLabsTTS] = None
        self.session_store = None

        # State
        self._current_participant: Optional[str] = None
        self._is_responding = False
        self._pending_transcript = ""
        self._response_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Initialize components and connect to room."""
        logger.info("Starting voice agent worker...")

        # Initialize VAD
        self.vad = SileroVAD(VADConfig(
            min_silence_duration_ms=self.config.end_of_turn_silence_ms
        ))
        self.vad.load_model()
        self.vad.set_callback(self._on_vad_event)

        # Initialize STT
        if self.config.deepgram_api_key:
            self.stt = StreamingDeepgramSTT(DeepgramConfig(
                api_key=self.config.deepgram_api_key,
                model=self.config.deepgram_model,
            ))
            logger.info("Deepgram STT enabled")
        else:
            logger.warning("Deepgram STT disabled (no API key)")

        # Initialize LLM
        if self.config.openai_api_key:
            self.llm = StreamingOpenAI(OpenAIConfig(
                api_key=self.config.openai_api_key,
                model=self.config.openai_model,
                max_tokens=self.config.max_response_tokens,
            ))
            logger.info("OpenAI LLM enabled")
        else:
            logger.warning("OpenAI LLM disabled (no API key)")

        # Initialize TTS
        if self.config.elevenlabs_api_key and self.config.elevenlabs_voice_id:
            self.tts = StreamingElevenLabsTTS(ElevenLabsConfig(
                api_key=self.config.elevenlabs_api_key,
                voice_id=self.config.elevenlabs_voice_id,
                model_id=self.config.elevenlabs_model_id,
            ))
            logger.info("ElevenLabs TTS enabled")
        else:
            logger.warning("ElevenLabs TTS disabled (no API key)")

        # Initialize session store
        try:
            self.session_store = RedisSessionStore(redis_url=self.config.redis_url)
            await self.session_store.connect()
            logger.info("Redis session store connected")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory store: {e}")
            self.session_store = InMemorySessionStore()
            await self.session_store.connect()

        # Connect to LiveKit room
        await self._connect_to_room()

    async def _connect_to_room(self) -> None:
        """Connect to LiveKit room and set up handlers."""
        from livekit import api

        token = (
            api.AccessToken(self.config.livekit_api_key, self.config.livekit_api_secret)
            .with_identity(self.config.identity)
            .with_name(self.config.identity)
            .with_grants(api.VideoGrants(
                room_join=True,
                room=self.config.room_name,
            ))
            .to_jwt()
        )

        self.room = rtc.Room()

        # Set up event handlers
        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
            asyncio.create_task(self._handle_participant_join(participant))

        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            asyncio.create_task(self._handle_participant_leave(participant))

        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.create_task(self._handle_audio_track(track, participant))

        @self.room.on("data_received")
        def on_data_received(data: rtc.DataPacket) -> None:
            asyncio.create_task(self._handle_data_packet(data))

        # Connect
        logger.info(f"Connecting to room '{self.config.room_name}'...")
        await self.room.connect(self.config.livekit_url, token)
        logger.info("Connected to LiveKit room")

    async def _handle_participant_join(self, participant: rtc.RemoteParticipant) -> None:
        """Handle new participant joining."""
        logger.info(f"Participant joined: {participant.identity}")
        self._current_participant = participant.identity

        # Create or resume session
        session = await self.session_store.get_or_create_session(participant.identity)

        # Send greeting if new session
        if not session.messages:
            greeting = "Hello! I'm here to help you today. What service are you looking for?"
            await self._send_response(participant.identity, greeting)

        # Connect STT if available
        if self.stt and not self.stt.is_connected:
            await self.stt.connect(self._on_transcript)

    async def _handle_participant_leave(self, participant: rtc.RemoteParticipant) -> None:
        """Handle participant leaving."""
        logger.info(f"Participant left: {participant.identity}")

        if self._current_participant == participant.identity:
            self._current_participant = None
            self._cancel_response()

        # Keep session for potential reconnection
        # Session will expire via TTL

    async def _handle_audio_track(
        self,
        track: rtc.Track,
        participant: rtc.RemoteParticipant,
    ) -> None:
        """Process incoming audio track for VAD and STT."""
        logger.info(f"Audio track received from {participant.identity}")

        audio_stream = rtc.AudioStream(track)

        async for frame_event in audio_stream:
            if self._stop_event.is_set():
                break

            frame = frame_event.frame
            audio_bytes = frame.data.tobytes()

            # Process through VAD
            if self.vad:
                self.vad.process_chunk(audio_bytes)

            # Send to STT
            if self.stt and self.stt.is_connected:
                await self.stt.send_audio(audio_bytes)

    async def _handle_data_packet(self, data: rtc.DataPacket) -> None:
        """Handle incoming data packets (text transcripts)."""
        try:
            text = data.data.decode("utf-8")
        except UnicodeDecodeError:
            return

        participant_id = data.participant.identity if data.participant else "unknown"
        logger.debug(f"Data packet from {participant_id}: {text}")

        # Treat as transcript if STT not available
        if not self.stt:
            self._on_transcript(text, is_final=True)

    def _on_vad_event(self, event: VADEvent) -> None:
        """Handle VAD speech events."""
        if event == VADEvent.SPEECH_START:
            logger.debug("VAD: Speech started")
            # User started speaking - handle barge-in
            if self._is_responding:
                self._cancel_response()

        elif event == VADEvent.SPEECH_END:
            logger.debug("VAD: Speech ended")
            # User stopped speaking - trigger response if we have transcript
            if self._pending_transcript.strip():
                asyncio.create_task(self._generate_response())

    def _on_transcript(self, text: str, is_final: bool) -> None:
        """Handle incoming transcripts from STT."""
        if not text.strip():
            return

        logger.debug(f"Transcript ({'final' if is_final else 'partial'}): {text}")

        if is_final:
            self._pending_transcript = text
            # If VAD detected end of speech, response will be triggered there
            # Otherwise, we rely on STT's endpointing
            if not self.vad or not self.vad.is_speaking:
                asyncio.create_task(self._generate_response())
        else:
            # Accumulate partial transcripts
            self._pending_transcript = text

    async def _generate_response(self) -> None:
        """Generate and send LLM response."""
        if not self._current_participant:
            return

        transcript = self._pending_transcript.strip()
        self._pending_transcript = ""

        if not transcript:
            return

        # Cancel any existing response
        self._cancel_response()

        participant_id = self._current_participant
        logger.info(f"Generating response for: {transcript}")

        # Save user message
        session = await self.session_store.add_user_message(participant_id, transcript)

        if not self.llm:
            # Fallback: echo mode
            await self._send_response(participant_id, f"I heard: {transcript}")
            return

        # Generate streaming response
        self._is_responding = True
        self._response_task = asyncio.create_task(
            self._stream_response(participant_id, session)
        )

    async def _stream_response(self, participant_id: str, session) -> None:
        """Stream LLM response through TTS to participant."""
        full_response = ""

        try:
            messages = session.get_messages_for_llm()

            async for sentence in self.llm.generate_stream(messages):
                if not self._is_responding:
                    # Cancelled (barge-in)
                    break

                full_response += sentence + " "

                # Synthesize and send audio
                await self._send_audio(participant_id, sentence)

                # Also send text
                await self._publish_text(participant_id, sentence)

            # Save assistant response
            if full_response.strip():
                await self.session_store.add_assistant_message(
                    participant_id, full_response.strip()
                )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
        finally:
            self._is_responding = False
            self._response_task = None

    async def _send_response(self, participant_id: str, text: str) -> None:
        """Send a complete response (text + audio)."""
        await self._publish_text(participant_id, text)
        await self._send_audio(participant_id, text)

        # Save to session
        await self.session_store.add_assistant_message(participant_id, text)

    async def _send_audio(self, participant_id: str, text: str) -> None:
        """Synthesize and send audio for text."""
        if not self.tts or not self.room or not self.room.local_participant:
            return

        try:
            audio_chunks = []
            async for chunk in self.tts.synthesize_stream(text):
                if not self._is_responding:
                    self.tts.cancel()
                    break
                audio_chunks.append(chunk)

            if audio_chunks:
                audio_data = b"".join(audio_chunks)
                payload = json.dumps({
                    "type": "tts_audio",
                    "encoding": "base64",
                    "data": base64.b64encode(audio_data).decode("utf-8"),
                }).encode("utf-8")
                self.room.local_participant.publish_data(payload)

        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def _publish_text(self, participant_id: str, text: str) -> None:
        """Publish text response to room."""
        if not self.room or not self.room.local_participant:
            return

        try:
            payload = json.dumps({
                "type": "text",
                "text": text,
            }).encode("utf-8")
            self.room.local_participant.publish_data(payload)
        except Exception as e:
            logger.error(f"Failed to publish text: {e}")

    def _cancel_response(self) -> None:
        """Cancel ongoing response (barge-in)."""
        if self._is_responding:
            logger.debug("Cancelling response (barge-in)")
            self._is_responding = False

            if self.tts:
                self.tts.cancel()

            if self._response_task and not self._response_task.done():
                self._response_task.cancel()

    async def run(self) -> None:
        """Run the agent until stopped."""
        await self._stop_event.wait()

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        logger.info("Stopping voice agent worker...")
        self._stop_event.set()
        self._cancel_response()

        if self.stt:
            await self.stt.close()

        if self.tts:
            await self.tts.close()

        if self.session_store:
            await self.session_store.close()

        if self.room:
            await self.room.disconnect()

        logger.info("Voice agent worker stopped")
