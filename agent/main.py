import argparse
import asyncio
import signal
from typing import Optional

from livekit import api, rtc


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

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        print(f"Participant connected: {participant.identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        print(f"Participant disconnected: {participant.identity}")

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
    parser.add_argument("--url", required=True, help="LiveKit server URL")
    parser.add_argument("--api-key", required=True, help="LiveKit API key")
    parser.add_argument("--api-secret", required=True, help="LiveKit API secret")
    parser.add_argument("--room", required=True, help="Room name to join")
    parser.add_argument("--identity", required=True, help="Participant identity")
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
