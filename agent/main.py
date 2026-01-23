import argparse
import asyncio
import base64
import json
import logging
import os
import signal
from typing import Optional
import binascii

from livekit import api, rtc

from agent.audio_providers import DeepgramConfig, DeepgramSTT, ElevenLabsConfig, ElevenLabsTTS
from agent.conversation import ConversationStore
from agent.llm_provider import OpenAIConfig, OpenAILLM
from livekit_agent import LiveKitAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPayload:
    def __init__(self, audio_bytes: bytes, mimetype: str) -> None:
        self.audio_bytes = audio_bytes
        self.mimetype = mimetype


def publish_text(
    room: rtc.Room,
    participant_id: str,
    response: str,
    tts: Optional[ElevenLabsTTS],
) -> bool:
    _ = participant_id
    if room.local_participant is None:
        logger.warning("No local participant available to publish responses.")
        return False
    try:
        payload = json.dumps({"type": "text", "text": response}).encode("utf-8")
        room.local_participant.publish_data(payload)
        if tts is not None:
            audio_bytes = tts.synthesize(response)
            audio_payload = json.dumps(
                {
                    "type": "tts_audio",
                    "encoding": "base64",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                }
            ).encode("utf-8")
            room.local_participant.publish_data(audio_payload)
    except Exception:  # pylint: disable=broad-except
        logger.exception("Failed to publish response.")
        return False
    return True


def _build_llm() -> Optional[OpenAILLM]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OpenAI LLM disabled; set OPENAI_API_KEY to enable.")
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    logger.info("OpenAI LLM enabled with model %s.", model)
    return OpenAILLM(OpenAIConfig(api_key=api_key, model=model))


def _build_stt() -> Optional[DeepgramSTT]:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        logger.info("Deepgram STT disabled; set DEEPGRAM_API_KEY to enable.")
        return None
    model = os.getenv("DEEPGRAM_MODEL", "nova-2")
    language = os.getenv("DEEPGRAM_LANGUAGE", "en")
    logger.info("Deepgram STT enabled with model %s.", model)
    return DeepgramSTT(DeepgramConfig(api_key=api_key, model=model, language=language))


def _generate_response(
    llm: Optional[OpenAILLM],
    store: ConversationStore,
    participant_id: str,
) -> Optional[str]:
    if llm is None:
        return None
    messages = store.build_messages(participant_id)
    if not messages:
        return None
    return llm.generate(messages)


def _decode_text_payload(data: bytes) -> Optional[str]:
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return decoded
    if payload.get("type") == "text":
        return payload.get("text")
    return None


def _decode_audio_payload(data: bytes) -> Optional[AudioPayload]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if payload.get("type") != "audio_chunk":
        return None
    if payload.get("encoding") != "base64":
        return None
    raw = payload.get("data")
    if not isinstance(raw, str):
        return None
    mimetype = payload.get("mimetype", "audio/wav")
    try:
        audio_bytes = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None
    return AudioPayload(audio_bytes, mimetype)


async def _delayed_respond(
    agent: LiveKitAgent,
    participant_id: str,
    response: str,
    delay_s: float,
) -> None:
    await asyncio.sleep(delay_s)
    agent.maybe_respond(participant_id, response)


async def run_agent(
    url: str,
    api_key: str,
    api_secret: str,
    room_name: str,
    identity: str,
) -> None:
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
            )
        )
        .to_jwt()
    )

    room = rtc.Room()
    conversation_store = ConversationStore()
    llm = _build_llm()
    stt = _build_stt()
    tts = None
    if os.getenv("ELEVENLABS_API_KEY") and os.getenv("ELEVENLABS_VOICE_ID"):
        tts = ElevenLabsTTS(
            ElevenLabsConfig(
                api_key=os.environ["ELEVENLABS_API_KEY"],
                voice_id=os.environ["ELEVENLABS_VOICE_ID"],
                model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2"),
                output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"),
            )
        )
        logger.info("ElevenLabs TTS enabled.")
    else:
        logger.info("ElevenLabs TTS disabled; set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID to enable.")

    agent = LiveKitAgent(
        publish_response=lambda participant_id, response: publish_text(room, participant_id, response, tts)
    )

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        print(f"Participant connected: {participant.identity}")
        agent.participant_join(participant.identity)
        conversation_store.start(participant.identity)

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        print(f"Participant disconnected: {participant.identity}")
        agent.participant_leave(participant.identity)
        conversation_store.end(participant.identity)

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket) -> None:
        text = _decode_text_payload(data.data)
        if text is None:
            audio_request = _decode_audio_payload(data.data)
            if audio_request and stt is not None:
                text = stt.transcribe(audio_request.audio_bytes, mimetype=audio_request.mimetype)
            if text is None:
                logger.warning("Received unsupported payload.")
                return
        participant_id = data.participant.identity if data.participant else "unknown"
        agent.handle_transcript(participant_id, text)
        conversation_store.append_user(participant_id, text)
        response = _generate_response(llm, conversation_store, participant_id)
        if response:
            conversation_store.append_assistant(participant_id, response)
            if agent.maybe_respond(participant_id, response) is None:
                asyncio.create_task(
                    _delayed_respond(
                        agent,
                        participant_id,
                        response,
                        agent.turn_taking.silence_timeout_s,
                    )
                )

    print(f"Connecting to room '{room_name}' as '{identity}'...")
    await room.connect(url, token)
    print("Connected. Press Ctrl+C to disconnect.")

    stop_event = asyncio.Event()

    def request_shutdown() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda *_: request_shutdown())

    await stop_event.wait()

    print("Disconnecting...")
    await room.disconnect()
    print("Disconnected.")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiveKit agent connection CLI")
    parser.add_argument(
        "--url",
        default=os.getenv("LIVEKIT_URL"),
        help="LiveKit server URL (or LIVEKIT_URL env var)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LIVEKIT_API_KEY"),
        help="LiveKit API key (or LIVEKIT_API_KEY env var)",
    )
    parser.add_argument(
        "--api-secret",
        default=os.getenv("LIVEKIT_API_SECRET"),
        help="LiveKit API secret (or LIVEKIT_API_SECRET env var)",
    )
    parser.add_argument(
        "--room",
        default=os.getenv("LIVEKIT_ROOM", "agent-room"),
        help="Room name to join (or LIVEKIT_ROOM env var)",
    )
    parser.add_argument(
        "--identity",
        default=os.getenv("LIVEKIT_IDENTITY", "agent"),
        help="Participant identity (or LIVEKIT_IDENTITY env var)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_agent(
            url=args.url,
            api_key=args.api_key,
            api_secret=args.api_secret,
            room_name=args.room,
            identity=args.identity,
        )
    )


if __name__ == "__main__":
    main()
