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
2. Start your agent entrypoint (replace with your actual command):

   ```bash
   python path/to/your_agent.py
   ```

If your agent uses a different runtime (Node, Go, etc.), run the equivalent command for your project.

## Notes

- The default LiveKit port is `7880` for HTTP and `7881/UDP` for RTC.
- For production deployments, follow LiveKit security and TLS best practices.
