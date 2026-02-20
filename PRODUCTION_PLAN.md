# LiveKit AI Agent - Production Architecture Plan

## Executive Summary

This document outlines the architecture, design decisions, and implementation plan to transform the current proof-of-concept into a production-grade real-time voice AI agent.

---

## Part 1: Current State Analysis

### What Exists Today

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Caller ──► LiveKit Room ──► Data Packets (text)               │
│                   │                                             │
│                   ▼                                             │
│           on_data_received()                                    │
│                   │                                             │
│                   ▼                                             │
│      LeadQualificationFlow.handle_message()                     │
│      (Hardcoded 5-question state machine)                       │
│                   │                                             │
│                   ▼                                             │
│         maybe_respond() ──► publish_text()                      │
│                                    │                            │
│                                    ▼                            │
│                         ElevenLabsTTS.synthesize() [BLOCKING]   │
│                                    │                            │
│                                    ▼                            │
│                         publish_data() (JSON payload)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Critical Gaps

| Component | Current State | Production Requirement |
|-----------|--------------|----------------------|
| **STT** | Not wired in main flow | Streaming Deepgram with VAD |
| **LLM** | Exists but unused | Integrated with conversation context |
| **TTS** | Blocking synchronous call | Streaming async synthesis |
| **State** | In-memory dict | Persistent storage (Redis/Postgres) |
| **Errors** | Basic try/except | Retry, circuit breaker, fallback |
| **Interrupts** | None | Barge-in detection, cancel synthesis |
| **Latency** | ~2-5s (full synthesis) | <500ms first byte |
| **Tests** | None | Unit, integration, load tests |
| **Observability** | print() statements | Structured logging, metrics, traces |

---

## Part 2: Target Production Architecture

### High-Level Design

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION ARCHITECTURE                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              ┌─────────────┐                                 │
│                              │   LiveKit   │                                 │
│                              │   Server    │                                 │
│                              └──────┬──────┘                                 │
│                                     │                                        │
│                    ┌────────────────┼────────────────┐                       │
│                    │                │                │                       │
│                    ▼                ▼                ▼                       │
│             ┌──────────┐     ┌──────────┐     ┌──────────┐                   │
│             │  Agent   │     │  Agent   │     │  Agent   │                   │
│             │ Worker 1 │     │ Worker 2 │     │ Worker N │                   │
│             └────┬─────┘     └────┬─────┘     └────┬─────┘                   │
│                  │                │                │                         │
│                  └────────────────┼────────────────┘                         │
│                                   │                                          │
│         ┌─────────────────────────┼─────────────────────────┐                │
│         │                         │                         │                │
│         ▼                         ▼                         ▼                │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐           │
│  │   Redis     │          │  Postgres   │          │   Metrics   │           │
│  │ (Sessions)  │          │  (History)  │          │ (Prometheus)│           │
│  └─────────────┘          └─────────────┘          └─────────────┘           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Single Agent Worker - Internal Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AGENT WORKER INTERNALS                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         AUDIO INPUT PIPELINE                             │    │
│  │                                                                          │    │
│  │   LiveKit Audio Track                                                    │    │
│  │         │                                                                │    │
│  │         ▼                                                                │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │    │
│  │   │   Resampler  │───►│  VAD Engine  │───►│  STT Stream  │               │    │
│  │   │  (48kHz→16k) │    │  (Silero)    │    │  (Deepgram)  │               │    │
│  │   └──────────────┘    └──────┬───────┘    └──────┬───────┘               │    │
│  │                              │                   │                       │    │
│  │                              ▼                   ▼                       │    │
│  │                    ┌─────────────────┐  ┌─────────────────┐              │    │
│  │                    │ Speaking Events │  │   Transcripts   │              │    │
│  │                    │ (start/stop)    │  │   (partial +    │              │    │
│  │                    └────────┬────────┘  │    final)       │              │    │
│  │                             │           └────────┬────────┘              │    │
│  │                             └──────────┬─────────┘                       │    │
│  │                                        │                                 │    │
│  └────────────────────────────────────────┼─────────────────────────────────┘    │
│                                           │                                      │
│  ┌────────────────────────────────────────┼─────────────────────────────────┐    │
│  │                      CONVERSATION ENGINE                                  │    │
│  │                                        ▼                                  │    │
│  │                           ┌────────────────────┐                          │    │
│  │                           │  Turn Taking Mgr   │                          │    │
│  │                           │  (end-of-turn      │                          │    │
│  │                           │   detection)       │                          │    │
│  │                           └─────────┬──────────┘                          │    │
│  │                                     │                                     │    │
│  │                                     ▼                                     │    │
│  │  ┌───────────────┐         ┌─────────────────┐        ┌───────────────┐   │    │
│  │  │ Conversation  │◄───────►│   LLM Manager   │◄──────►│   Tool/RAG    │   │    │
│  │  │    Store      │         │  (streaming)    │        │   Executor    │   │    │
│  │  │  (Redis)      │         └────────┬────────┘        └───────────────┘   │    │
│  │  └───────────────┘                  │                                     │    │
│  │                                     │                                     │    │
│  └─────────────────────────────────────┼─────────────────────────────────────┘    │
│                                        │                                         │
│  ┌─────────────────────────────────────┼─────────────────────────────────────┐    │
│  │                         AUDIO OUTPUT PIPELINE                              │    │
│  │                                     ▼                                      │    │
│  │                          ┌───────────────────┐                             │    │
│  │                          │  Interrupt Ctrl   │                             │    │
│  │                          │  (barge-in)       │                             │    │
│  │                          └─────────┬─────────┘                             │    │
│  │                                    │                                       │    │
│  │                                    ▼                                       │    │
│  │   ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐             │    │
│  │   │  TTS Stream  │───►│  Audio Playback  │───►│   LiveKit    │             │    │
│  │   │ (ElevenLabs) │    │     Queue        │    │   Publish    │             │    │
│  │   └──────────────┘    └──────────────────┘    └──────────────┘             │    │
│  │                                                                            │    │
│  └────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Component Deep Dives

### 3.1 Speech-to-Text Pipeline

#### Current Implementation
```python
# audio_providers.py - Synchronous, batch processing
class DeepgramSTT:
    def transcribe(self, audio_bytes: bytes, mimetype: str = "audio/wav") -> Optional[str]:
        req = request.Request(url, data=audio_bytes, headers=headers, method="POST")
        with request.urlopen(req, timeout=30) as resp:  # BLOCKING
            payload = json.loads(resp.read().decode("utf-8"))
        return _extract_deepgram_transcript(payload)
```

**Problems:**
- Synchronous HTTP call blocks event loop
- Batch mode: waits for complete audio before transcribing
- No streaming = high latency
- No partial transcripts = no early response preparation

#### Production Implementation

```python
# Streaming Deepgram with WebSocket
class StreamingDeepgramSTT:
    def __init__(self, config: DeepgramConfig):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.transcript_callback: Optional[Callable[[str, bool], None]] = None

    async def connect(self):
        url = f"wss://api.deepgram.com/v1/listen?model={self.config.model}&interim_results=true"
        self.ws = await websockets.connect(url, extra_headers={...})
        asyncio.create_task(self._receive_loop())

    async def send_audio(self, chunk: bytes):
        """Non-blocking audio send"""
        await self.ws.send(chunk)

    async def _receive_loop(self):
        async for message in self.ws:
            data = json.loads(message)
            transcript = data["channel"]["alternatives"][0]["transcript"]
            is_final = data["is_final"]
            if self.transcript_callback:
                self.transcript_callback(transcript, is_final)
```

**Tradeoffs:**

| Approach | Latency | Complexity | Cost |
|----------|---------|------------|------|
| Batch REST API | 2-5s | Low | Lower (per-request) |
| Streaming WebSocket | 200-500ms | Medium | Higher (connection time) |
| On-device (Whisper) | 100-300ms | High | Free (but CPU/GPU) |

**Recommendation:** Streaming WebSocket for production. The 10x latency improvement is worth the complexity.

---

### 3.2 Voice Activity Detection (VAD)

#### Current Implementation
```python
# livekit_agent.py - Timestamp-based, no actual audio analysis
class TurnTakingManager:
    def should_respond(self, state: ParticipantState, now: float) -> bool:
        if state.speaking:
            return False
        if state.last_speech_end_ts is None:
            return False
        return (now - state.last_speech_end_ts) >= self.silence_timeout_s
```

**Problems:**
- Relies on external "speaking" signal (not clear where this comes from)
- Fixed silence timeout (1s) - doesn't adapt to conversation dynamics
- No audio-level VAD = no real speech detection

#### Production Implementation

```python
import torch
from silero_vad import VADIterator, load_silero_vad

class SileroVAD:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=300,  # Configurable
            speech_pad_ms=100
        )

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """Returns speech segments with timestamps"""
        speech_dict = self.vad_iterator(audio_chunk)
        return speech_dict  # {'start': ms, 'end': ms} or None
```

**VAD Options Comparison:**

| Engine | Accuracy | Latency | Resource | License |
|--------|----------|---------|----------|---------|
| Silero VAD | 95%+ | <10ms | CPU (light) | MIT |
| WebRTC VAD | 85% | <5ms | CPU (minimal) | BSD |
| Deepgram (server) | 98% | Network RTT | None | Commercial |
| Custom RNN | Variable | Variable | CPU/GPU | - |

**Recommendation:** Silero VAD. Best accuracy-to-resource ratio, runs locally, MIT licensed.

---

### 3.3 LLM Integration

#### Current Implementation
```python
# llm_provider.py - Synchronous, non-streaming
class OpenAILLM:
    def generate(self, messages: List[dict[str, str]]) -> Optional[str]:
        body = json.dumps({
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 160,  # Very limited
        }).encode("utf-8")
        req = request.Request(...)
        with request.urlopen(req, timeout=30) as resp:  # BLOCKING
            payload = json.loads(resp.read().decode("utf-8"))
        return _extract_content(payload)
```

**Problems:**
- Synchronous = blocks event loop
- Non-streaming = waits for complete response
- Not wired into main flow (lead_qualification.py uses hardcoded logic)
- No retry logic for transient failures

#### Production Implementation

```python
import openai
from typing import AsyncGenerator

class StreamingLLM:
    def __init__(self, config: LLMConfig):
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
        self.model = config.model

    async def generate_stream(
        self,
        messages: list[dict],
        on_chunk: Callable[[str], None]
    ) -> AsyncGenerator[str, None]:
        """Stream tokens as they arrive"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.3,
            max_tokens=300,
        )

        buffer = ""
        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            buffer += token

            # Emit sentence-by-sentence for TTS
            if any(p in buffer for p in ".!?"):
                sentence, buffer = self._split_sentence(buffer)
                yield sentence
                on_chunk(sentence)

        if buffer.strip():
            yield buffer
            on_chunk(buffer)
```

**LLM Streaming Strategy:**

```
Token Stream:  "Hello," "I'd" "be" "happy" "to" "help" "you" "today" "."
                  │       │     │     │      │     │      │      │     │
                  └───────┴─────┴─────┴──────┴─────┴──────┴──────┴─────┘
                                              │
                                              ▼
Sentence Buffer:            "Hello, I'd be happy to help you today."
                                              │
                                              ▼
                                    TTS.synthesize() ───► Audio Stream
```

**Why sentence-level batching?**
- Word-by-word TTS sounds choppy
- Full-response TTS is too slow
- Sentences are natural speech units

---

### 3.4 Text-to-Speech Pipeline

#### Current Implementation
```python
# audio_providers.py - Blocking, full synthesis
class ElevenLabsTTS:
    def synthesize(self, text: str) -> bytes:
        # ... builds request ...
        with request.urlopen(req, timeout=30) as resp:  # BLOCKING
            return resp.read()  # Waits for COMPLETE audio
```

```python
# main.py - Called synchronously in async context
def publish_text(...):
    if tts is not None:
        audio_bytes = tts.synthesize(response)  # BLOCKS EVENT LOOP
```

**Problems:**
- Synchronous HTTP blocks entire agent
- Returns complete audio (300ms text = 2s+ wait)
- No streaming = terrible time-to-first-audio
- No interruption handling

#### Production Implementation

```python
import httpx
from typing import AsyncGenerator

class StreamingElevenLabsTTS:
    def __init__(self, config: ElevenLabsConfig):
        self.config = config
        self.client = httpx.AsyncClient()
        self._cancel_event: Optional[asyncio.Event] = None

    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks as they're generated"""
        self._cancel_event = asyncio.Event()

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.voice_id}/stream"

        async with self.client.stream(
            "POST",
            url,
            json={"text": text, "model_id": self.config.model_id},
            headers={"xi-api-key": self.config.api_key},
        ) as response:
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if self._cancel_event.is_set():
                    break  # Barge-in: stop synthesis
                yield chunk

    def cancel(self):
        """Called when user interrupts"""
        if self._cancel_event:
            self._cancel_event.set()
```

**TTS Provider Comparison:**

| Provider | Latency (first byte) | Quality | Streaming | Cost/1M chars |
|----------|---------------------|---------|-----------|---------------|
| ElevenLabs | 200-400ms | Excellent | Yes | $30-330 |
| OpenAI TTS | 300-500ms | Good | Yes | $15 |
| Google Cloud | 100-200ms | Good | Yes | $4-16 |
| Azure | 100-200ms | Good | Yes | $4-16 |
| Deepgram Aura | 150-300ms | Good | Yes | $15 |

**Recommendation:** ElevenLabs for quality, Google/Azure for cost. Support multiple providers for fallback.

---

### 3.5 Barge-In / Interruption Handling

#### Current Implementation
None. If user speaks while agent is talking, both audio streams overlap.

#### Production Implementation

```python
class InterruptionController:
    def __init__(self):
        self.is_speaking = False
        self.playback_task: Optional[asyncio.Task] = None
        self.tts: StreamingElevenLabsTTS = None

    async def start_speaking(self, text: str, publish_fn: Callable):
        """Start TTS playback with interruption support"""
        self.is_speaking = True

        try:
            async for audio_chunk in self.tts.synthesize_stream(text):
                if not self.is_speaking:  # Interrupted
                    break
                await publish_fn(audio_chunk)
        finally:
            self.is_speaking = False

    def handle_user_speech_start(self):
        """Called when VAD detects user starting to speak"""
        if self.is_speaking:
            # User is interrupting - stop playback
            self.is_speaking = False
            self.tts.cancel()
            # Optionally: save position for resumption
```

**Interruption Strategies:**

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| **Immediate Stop** | Stop TTS instantly | Default, most natural |
| **Finish Sentence** | Complete current sentence | Formal contexts |
| **Ignore Short** | Ignore <500ms speech | Avoid false triggers |
| **Smart Resume** | Resume where stopped | Long explanations |

---

### 3.6 State Management & Persistence

#### Current Implementation
```python
# In-memory dictionaries
class LeadQualificationFlow:
    def __init__(self):
        self.sessions: Dict[str, QualificationSession] = {}  # Lost on restart
```

#### Production Implementation

```python
import redis.asyncio as redis
from dataclasses import asdict
import json

class RedisSessionStore:
    def __init__(self, redis_url: str, ttl_seconds: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl_seconds

    async def get_session(self, participant_id: str) -> Optional[ConversationSession]:
        data = await self.redis.get(f"session:{participant_id}")
        if data:
            return ConversationSession(**json.loads(data))
        return None

    async def save_session(self, session: ConversationSession):
        key = f"session:{session.participant_id}"
        await self.redis.setex(key, self.ttl, json.dumps(asdict(session)))

    async def delete_session(self, participant_id: str):
        await self.redis.delete(f"session:{participant_id}")
```

**Storage Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Redis (Hot Data)              Postgres (Cold Data)        │
│   ┌─────────────────┐           ┌─────────────────┐         │
│   │ Active Sessions │           │ Conversation    │         │
│   │ TTL: 1 hour     │──archive─►│ History         │         │
│   │                 │           │                 │         │
│   │ Turn State      │           │ Lead Records    │         │
│   │ TTL: 5 minutes  │           │                 │         │
│   │                 │           │ Analytics       │         │
│   │ Rate Limits     │           │                 │         │
│   └─────────────────┘           └─────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.7 Error Handling & Resilience

#### Current Implementation
```python
# Bare try/except with logging
try:
    payload = json.dumps({...})
    room.local_participant.publish_data(payload)
except Exception:
    logger.exception("Failed to publish response.")
    return False
```

#### Production Implementation

```python
from tenacity import retry, stop_after_attempt, wait_exponential
from circuitbreaker import circuit

class ResilientLLM:
    def __init__(self, primary: StreamingLLM, fallback: StreamingLLM):
        self.primary = primary
        self.fallback = fallback

    @circuit(failure_threshold=5, recovery_timeout=60)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def generate(self, messages: list[dict]) -> str:
        try:
            return await self.primary.generate_stream(messages)
        except Exception as e:
            # Circuit breaker will track failures
            raise

    async def generate_with_fallback(self, messages: list[dict]) -> str:
        try:
            return await self.generate(messages)
        except Exception:
            # Primary failed, try fallback (e.g., smaller/cheaper model)
            return await self.fallback.generate_stream(messages)
```

**Resilience Patterns:**

| Pattern | Purpose | Implementation |
|---------|---------|----------------|
| **Retry** | Transient failures | tenacity library |
| **Circuit Breaker** | Prevent cascade | circuitbreaker library |
| **Fallback** | Graceful degradation | Secondary provider |
| **Timeout** | Prevent hangs | asyncio.timeout() |
| **Bulkhead** | Isolate failures | Semaphores per service |

---

## Part 4: Implementation Phases

### Phase 1: Core Pipeline (Foundation)
**Goal:** Working voice-to-voice conversation

| Task | Priority | Complexity | Dependencies |
|------|----------|------------|--------------|
| Async HTTP client (httpx/aiohttp) | P0 | Low | None |
| Streaming Deepgram STT | P0 | Medium | Async HTTP |
| Silero VAD integration | P0 | Medium | None |
| Wire LLM into conversation flow | P0 | Low | None |
| Streaming ElevenLabs TTS | P0 | Medium | Async HTTP |
| Async publish to LiveKit | P0 | Low | None |

### Phase 2: Conversation Quality
**Goal:** Natural, low-latency interactions

| Task | Priority | Complexity | Dependencies |
|------|----------|------------|--------------|
| LLM streaming with sentence batching | P0 | Medium | Phase 1 |
| Barge-in / interruption handling | P0 | High | VAD, TTS streaming |
| End-of-turn detection tuning | P1 | Medium | VAD |
| Partial transcript handling | P1 | Medium | STT streaming |
| Response latency optimization | P1 | Medium | All streaming |

### Phase 3: Reliability
**Goal:** Production-grade error handling

| Task | Priority | Complexity | Dependencies |
|------|----------|------------|--------------|
| Retry logic for all external APIs | P0 | Low | None |
| Circuit breakers | P1 | Medium | None |
| Fallback providers (LLM, TTS) | P1 | Medium | None |
| Redis session persistence | P0 | Medium | None |
| Graceful degradation modes | P2 | Medium | Fallbacks |

### Phase 4: Observability
**Goal:** Know what's happening in production

| Task | Priority | Complexity | Dependencies |
|------|----------|------------|--------------|
| Structured logging (structlog) | P0 | Low | None |
| Prometheus metrics | P1 | Medium | None |
| OpenTelemetry tracing | P2 | Medium | None |
| Latency dashboards | P1 | Low | Metrics |
| Error alerting | P1 | Low | Logging |

### Phase 5: Scale & Operations
**Goal:** Handle real traffic

| Task | Priority | Complexity | Dependencies |
|------|----------|------------|--------------|
| Multi-worker deployment | P1 | Medium | Redis sessions |
| Rate limiting | P1 | Low | Redis |
| Health checks | P0 | Low | None |
| Kubernetes manifests | P2 | Medium | Containerization |
| Auto-scaling policies | P2 | Medium | K8s |

### Phase 6: Testing
**Goal:** Confidence in changes

| Task | Priority | Complexity | Dependencies |
|------|----------|------------|--------------|
| Unit tests for core logic | P0 | Medium | None |
| Integration tests (mocked APIs) | P1 | Medium | Unit tests |
| End-to-end conversation tests | P2 | High | Integration tests |
| Load testing | P2 | Medium | Deployment |
| Chaos testing | P3 | High | All |

---

## Part 5: Key Architectural Decisions

### Decision 1: Streaming vs Batch Processing

**Context:** Every external API (STT, LLM, TTS) can operate in batch or streaming mode.

**Decision:** Stream everything.

**Rationale:**
- Batch: User speaks → wait for STT → wait for LLM → wait for TTS → respond
- Stream: User speaks → STT streams text → LLM streams response → TTS streams audio → respond

**Latency comparison:**
```
Batch Mode:
[───── STT 2s ─────][───── LLM 3s ─────][───── TTS 2s ─────] = 7s total

Stream Mode:
[─ STT ─]
    [─ LLM stream ─]
         [─ TTS stream ─]
              [─ Audio ─] = ~1s to first audio
```

**Trade-off:** Complexity increases significantly. Streaming requires WebSockets, chunked processing, and careful buffer management.

---

### Decision 2: VAD Location (Client vs Server)

**Context:** Voice Activity Detection can run on the client (browser/app) or server (agent).

**Decision:** Server-side VAD (Silero).

**Rationale:**
- Client VAD requires SDK changes for each platform
- Server VAD gives consistent behavior
- LiveKit provides audio frames directly to agent
- Silero is lightweight enough for real-time

**Trade-off:** Slightly more server CPU, but eliminates client dependency.

---

### Decision 3: Session State Storage

**Context:** Where to store conversation state between turns?

**Options:**
1. In-memory (current)
2. Redis only
3. Redis + Postgres

**Decision:** Redis for active sessions, Postgres for history.

**Rationale:**
- Redis: Fast reads/writes for active conversations
- Postgres: Durable storage for analytics, compliance, ML training
- In-memory: Lost on restart, no horizontal scaling

**Trade-off:** Operational complexity of two databases.

---

### Decision 4: Single Agent vs Agent Pool

**Context:** Should one agent handle one room, or pool agents across rooms?

**Decision:** One agent process per room, with worker pool for scaling.

**Rationale:**
- LiveKit's model: agent joins room as participant
- Simpler state management per-room
- Horizontal scaling via worker processes
- LiveKit has built-in agent dispatch

**Trade-off:** More processes, but cleaner isolation.

---

### Decision 5: LLM Provider Strategy

**Context:** Which LLM? Single provider or multi-provider?

**Decision:** Primary (GPT-4o-mini) + Fallback (Claude Haiku or local).

**Rationale:**
- GPT-4o-mini: Good balance of speed/quality/cost for voice
- Fallback prevents outages from killing the service
- Local fallback (Ollama) for disaster recovery

**Trade-off:** Need to maintain prompts for multiple models.

---

## Part 6: Latency Budget

For natural conversation, total response latency should be <1 second.

```
┌─────────────────────────────────────────────────────────────┐
│                    LATENCY BUDGET                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Component              Target      Current    Status      │
│   ─────────────────────────────────────────────────────     │
│   Network RTT            50ms        50ms       ✓           │
│   VAD processing         10ms        N/A        ⚠ Missing   │
│   STT (first word)       200ms       2000ms+    ✗ Batch     │
│   End-of-turn detection  300ms       1000ms     ✗ Fixed     │
│   LLM (first token)      200ms       3000ms+    ✗ Batch     │
│   TTS (first chunk)      200ms       2000ms+    ✗ Batch     │
│   Audio encoding         10ms        10ms       ✓           │
│   ─────────────────────────────────────────────────────     │
│   TOTAL                  ~970ms      8000ms+                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 7: Cost Estimation

Per-conversation cost (assuming 3-minute average call):

| Service | Usage | Unit Cost | Per-Call Cost |
|---------|-------|-----------|---------------|
| LiveKit Cloud | 3 min × 2 participants | $0.004/min | $0.024 |
| Deepgram STT | ~500 words | $0.0043/min | $0.013 |
| OpenAI GPT-4o-mini | ~2000 tokens | $0.15/1M in, $0.60/1M out | $0.002 |
| ElevenLabs TTS | ~300 chars | $0.30/1K chars | $0.09 |
| **Total** | | | **~$0.13/call** |

At 10,000 calls/month: ~$1,300/month in API costs.

---

## Part 8: File Structure (Target)

```
Livekitagent/
├── agent/
│   ├── __init__.py
│   ├── main.py                    # Entry point
│   ├── worker.py                  # Agent worker lifecycle
│   │
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── vad.py                 # Silero VAD wrapper
│   │   ├── resampler.py           # Audio format conversion
│   │   └── playback.py            # Audio queue management
│   │
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── base.py                # STT protocol
│   │   ├── deepgram.py            # Streaming Deepgram
│   │   └── whisper.py             # Local Whisper fallback
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                # LLM protocol
│   │   ├── openai.py              # Streaming OpenAI
│   │   ├── anthropic.py           # Streaming Claude
│   │   └── manager.py             # Retry, fallback, circuit breaker
│   │
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── base.py                # TTS protocol
│   │   ├── elevenlabs.py          # Streaming ElevenLabs
│   │   └── google.py              # Google Cloud TTS fallback
│   │
│   ├── conversation/
│   │   ├── __init__.py
│   │   ├── session.py             # Session state
│   │   ├── store.py               # Redis persistence
│   │   ├── turn_taking.py         # End-of-turn detection
│   │   └── interruption.py        # Barge-in handling
│   │
│   ├── flows/
│   │   ├── __init__.py
│   │   ├── base.py                # Flow protocol
│   │   └── lead_qualification.py  # Lead qual flow (LLM-powered)
│   │
│   └── observability/
│       ├── __init__.py
│       ├── logging.py             # Structured logging
│       ├── metrics.py             # Prometheus metrics
│       └── tracing.py             # OpenTelemetry
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── k8s/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Part 9: Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time-to-first-audio | <1s | P95 latency |
| Conversation completion rate | >90% | Calls ending normally |
| Interruption success rate | >95% | User barge-ins handled |
| API error rate | <1% | Failed external calls |
| Session recovery rate | 100% | Sessions surviving restart |
| Uptime | 99.9% | Monthly availability |

---

## Summary

This project has a solid foundation but needs significant work across:

1. **Streaming everything** - STT, LLM, TTS must all stream
2. **Async everything** - No blocking calls in async context
3. **Resilience** - Retries, circuit breakers, fallbacks
4. **Persistence** - Redis + Postgres for state
5. **Observability** - Metrics, logging, tracing
6. **Testing** - Unit, integration, E2E

The good news: the abstractions (Protocol classes in `interfaces.py`) are well-designed. The work is filling in production-ready implementations.
