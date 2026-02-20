"""
LiveKit Voice Agent - Production Entry Point

Usage:
    python -m agent.main [options]

Or via environment variables (see .env.example)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Optional

import structlog
from dotenv import load_dotenv

from agent.worker import VoiceAgentWorker, WorkerConfig

# Load environment variables from .env file
load_dotenv()


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    # Also configure standard logging for libraries
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LiveKit Voice Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (can also be set in .env file):
  LIVEKIT_URL          LiveKit server URL
  LIVEKIT_API_KEY      LiveKit API key
  LIVEKIT_API_SECRET   LiveKit API secret
  LIVEKIT_ROOM         Room name to join
  DEEPGRAM_API_KEY     Deepgram API key for STT
  OPENAI_API_KEY       OpenAI API key for LLM
  ELEVENLABS_API_KEY   ElevenLabs API key for TTS
  ELEVENLABS_VOICE_ID  ElevenLabs voice ID
  REDIS_URL            Redis connection URL
        """,
    )

    parser.add_argument(
        "--url",
        default=os.getenv("LIVEKIT_URL"),
        help="LiveKit server URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LIVEKIT_API_KEY"),
        help="LiveKit API key",
    )
    parser.add_argument(
        "--api-secret",
        default=os.getenv("LIVEKIT_API_SECRET"),
        help="LiveKit API secret",
    )
    parser.add_argument(
        "--room",
        default=os.getenv("LIVEKIT_ROOM", "agent-room"),
        help="Room name to join",
    )
    parser.add_argument(
        "--identity",
        default=os.getenv("LIVEKIT_IDENTITY", "agent"),
        help="Agent identity",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args(argv)

    # Validate required arguments
    if not args.url:
        parser.error("--url or LIVEKIT_URL is required")
    if not args.api_key:
        parser.error("--api-key or LIVEKIT_API_KEY is required")
    if not args.api_secret:
        parser.error("--api-secret or LIVEKIT_API_SECRET is required")

    return args


async def run_worker(config: WorkerConfig) -> None:
    """Run the voice agent worker with signal handling."""
    worker = VoiceAgentWorker(config)

    # Set up signal handlers
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        asyncio.create_task(worker.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda *_: request_shutdown())

    try:
        await worker.start()
        print(f"Agent running in room '{config.room_name}'. Press Ctrl+C to stop.")
        await worker.run()
    except Exception as e:
        logging.error(f"Worker error: {e}")
        raise
    finally:
        await worker.stop()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    config = WorkerConfig(
        livekit_url=args.url,
        livekit_api_key=args.api_key,
        livekit_api_secret=args.api_secret,
        room_name=args.room,
        identity=args.identity,
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        deepgram_model=os.getenv("DEEPGRAM_MODEL", "nova-2"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
        elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    )

    asyncio.run(run_worker(config))


if __name__ == "__main__":
    main()
