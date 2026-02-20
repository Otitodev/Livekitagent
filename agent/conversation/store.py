"""Redis-backed conversation session persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


LEAD_QUALIFICATION_SYSTEM_PROMPT = """You are a lead qualification assistant for a service business.

Your goal is to qualify potential customers by gathering key information through natural conversation.

Information to gather:
1. Service intent - What service are they looking for?
2. Location - What city/area are they in?
3. Timeline - When do they need this done?
4. Budget - Do they have a budget range in mind?
5. Contact - Best way to reach them (phone/email)

Guidelines:
- Ask ONE question at a time
- Keep responses brief (1-2 sentences) - this is a phone call
- Be friendly but professional
- If they seem qualified (clear need, reasonable timeline, budget awareness), offer to book a consultation
- If not a fit, politely explain and offer alternatives

Once you have enough info:
- Qualified lead: Offer two specific time slots for a consultation
- Not qualified: Politely explain why and offer to connect them with resources"""


@dataclass
class ConversationSession:
    participant_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    # Qualification state
    service_intent: Optional[str] = None
    location: Optional[str] = None
    timeline: Optional[str] = None
    budget: Optional[str] = None
    contact: Optional[str] = None
    is_qualified: Optional[bool] = None

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append({"role": "assistant", "content": content})

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get conversation history formatted for LLM."""
        return [
            {"role": "system", "content": LEAD_QUALIFICATION_SYSTEM_PROMPT}
        ] + self.messages

    def to_dict(self) -> Dict:
        """Serialize session to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationSession":
        """Deserialize session from dictionary."""
        return cls(**data)


class RedisSessionStore:
    """
    Redis-backed session storage for conversation state.

    Sessions are persisted with TTL and survive server restarts.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        session_ttl_seconds: int = 3600,
        key_prefix: str = "session:",
    ):
        self.redis_url = redis_url
        self.session_ttl = session_ttl_seconds
        self.key_prefix = key_prefix
        self._redis: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            await self._redis.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis")

    def _key(self, participant_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.key_prefix}{participant_id}"

    async def get_session(self, participant_id: str) -> Optional[ConversationSession]:
        """
        Retrieve a session from Redis.

        Returns None if session doesn't exist.
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        data = await self._redis.get(self._key(participant_id))
        if data:
            try:
                return ConversationSession.from_dict(json.loads(data))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to deserialize session: {e}")
                return None
        return None

    async def save_session(self, session: ConversationSession) -> None:
        """
        Save a session to Redis with TTL.

        TTL is refreshed on each save.
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        await self._redis.setex(
            self._key(session.participant_id),
            self.session_ttl,
            json.dumps(session.to_dict()),
        )

    async def delete_session(self, participant_id: str) -> None:
        """Delete a session from Redis."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        await self._redis.delete(self._key(participant_id))

    async def get_or_create_session(self, participant_id: str) -> ConversationSession:
        """
        Get existing session or create a new one.

        New sessions are automatically persisted.
        """
        session = await self.get_session(participant_id)
        if session is None:
            session = ConversationSession(participant_id=participant_id)
            await self.save_session(session)
            logger.info(f"Created new session for {participant_id}")
        return session

    async def add_user_message(self, participant_id: str, content: str) -> ConversationSession:
        """
        Add a user message and persist.

        Returns the updated session.
        """
        session = await self.get_or_create_session(participant_id)
        session.add_user_message(content)
        await self.save_session(session)
        return session

    async def add_assistant_message(
        self, participant_id: str, content: str
    ) -> ConversationSession:
        """
        Add an assistant message and persist.

        Returns the updated session.
        """
        session = await self.get_or_create_session(participant_id)
        session.add_assistant_message(content)
        await self.save_session(session)
        return session


class InMemorySessionStore:
    """
    In-memory session store for testing/development.

    Same interface as RedisSessionStore but no persistence.
    """

    def __init__(self):
        self._sessions: Dict[str, ConversationSession] = {}

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def get_session(self, participant_id: str) -> Optional[ConversationSession]:
        return self._sessions.get(participant_id)

    async def save_session(self, session: ConversationSession) -> None:
        self._sessions[session.participant_id] = session

    async def delete_session(self, participant_id: str) -> None:
        self._sessions.pop(participant_id, None)

    async def get_or_create_session(self, participant_id: str) -> ConversationSession:
        if participant_id not in self._sessions:
            self._sessions[participant_id] = ConversationSession(
                participant_id=participant_id
            )
        return self._sessions[participant_id]

    async def add_user_message(self, participant_id: str, content: str) -> ConversationSession:
        session = await self.get_or_create_session(participant_id)
        session.add_user_message(content)
        return session

    async def add_assistant_message(
        self, participant_id: str, content: str
    ) -> ConversationSession:
        session = await self.get_or_create_session(participant_id)
        session.add_assistant_message(content)
        return session
