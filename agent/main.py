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

from agent.audio_providers import ElevenLabsConfig, ElevenLabsTTS
from agent.lead_qualification import LeadQualificationFlow
from livekit_agent import LiveKitAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    qualification_flow = LeadQualificationFlow()
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
        qualification_flow.start_session(participant.identity)

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        print(f"Participant disconnected: {participant.identity}")
        agent.participant_leave(participant.identity)
        qualification_flow.end_session(participant.identity)

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket) -> None:
        try:
            text = data.data.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Received non-text data payload.")
            return
        participant_id = data.participant.identity if data.participant else "unknown"
        agent.handle_transcript(participant_id, text)
        response = qualification_flow.handle_message(participant_id, text)
        agent.maybe_respond(participant_id, response)

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
