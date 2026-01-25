from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConversationSession:
    participant_id: str
    messages: List[dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


class ConversationStore:
    def __init__(self) -> None:
        self.sessions: Dict[str, ConversationSession] = {}

    def start(self, participant_id: str) -> None:
        if participant_id not in self.sessions:
            self.sessions[participant_id] = ConversationSession(participant_id=participant_id)

    def end(self, participant_id: str) -> None:
        self.sessions.pop(participant_id, None)

    def append_user(self, participant_id: str, content: str) -> None:
        self._get_session(participant_id).add("user", content)

    def append_assistant(self, participant_id: str, content: str) -> None:
        self._get_session(participant_id).add("assistant", content)

    def build_messages(self, participant_id: str) -> List[dict[str, str]]:
        session = self._get_session(participant_id)
        return [_system_prompt(), *session.messages]

    def _get_session(self, participant_id: str) -> ConversationSession:
        if participant_id not in self.sessions:
            self.start(participant_id)
        return self.sessions[participant_id]


def _system_prompt() -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are a lead qualification assistant for a service business. "
            "Ask one concise question at a time to gather: service intent, location, timeline, "
            "budget range, and contact info. Once you have enough info, either offer to book "
            "with two time options or politely hand off if not a fit. Keep responses brief "
            "and phone-friendly."
        ),
    }
