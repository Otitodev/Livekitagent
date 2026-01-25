# LiveKit Agent Setup

This repository contains configuration and documentation to help you run a LiveKit-powered agent locally.

## Prerequisites

- Docker + Docker Compose (for running a local LiveKit server)
- A LiveKit cloud project **or** a local LiveKit server
- API keys for any model/provider you plan to use

## Environment configuration

1. Copy the template and fill in your credentials:

   ```bash
   cp .env.example .env
   ```

2. Update `.env` with your LiveKit URL, API key/secret, and any provider keys you need.

## Run a local LiveKit server (self-hosted)

A lightweight `docker-compose.yml` is provided for local development.

1. Update `livekit.yaml` with your LiveKit API key/secret. For convenience, you can:
   - Replace the `${LIVEKIT_API_KEY}` / `${LIVEKIT_API_SECRET}` placeholders manually, **or**
   - Use a tool like `envsubst` to generate a real config file before launching.

2. Start the server:

   ```bash
   docker compose up -d
   ```

3. Verify the server is running at `http://localhost:7880`.

## Run the agent

1. Ensure `.env` is populated.
2. Start the agent entrypoint (uses env defaults if set). The current agent uses a
   lead-qualification question flow over LiveKit data messages. If you set
   `ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID`, the agent also publishes a
   base64-encoded `tts_audio` data message with ElevenLabs Turbo audio. If you set
   `DEEPGRAM_API_KEY` and send `audio_chunk` payloads, the agent will transcribe them
   with Deepgram and pass the text to the LLM:

   ```bash
   python -m agent.main \
     --url "$LIVEKIT_URL" \
     --api-key "$LIVEKIT_API_KEY" \
     --api-secret "$LIVEKIT_API_SECRET" \
     --room "agent-room" \
     --identity "agent"
   ```

You can also set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_ROOM`, and
`LIVEKIT_IDENTITY` in your environment and omit the flags.

## Lead qualification flow (POC)

The current proof-of-concept flow asks for:
- Service intent
- Location
- Timeline
- Budget range
- Contact information

It then responds with a booking prompt when qualified, or a polite handoff message if not.

## TTS data payload format (POC)

The agent publishes JSON data payloads with `type: text` and optionally `type: tts_audio`:

```json
{"type":"text","text":"Thanks for calling! What service are you looking for today?"}
```

```json
{"type":"tts_audio","encoding":"base64","data":"<mp3 bytes>"}
```

## Audio chunk payload format (POC)

To send audio for transcription, publish a JSON data payload with base64 audio bytes:

```json
{"type":"audio_chunk","encoding":"base64","mimetype":"audio/wav","data":"<wav bytes>"}
```

## Notes

- The default LiveKit port is `7880` for HTTP and `7881/UDP` for RTC.
- For production deployments, follow LiveKit security and TLS best practices.
