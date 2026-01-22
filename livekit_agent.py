from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParticipantState:
    participant_id: str
    speaking: bool = False
    last_utterance_ts: Optional[float] = None
    last_speech_end_ts: Optional[float] = None
    last_response_ts: Optional[float] = None
    conversation_context: List[str] = field(default_factory=list)
    connected: bool = True

    def mark_disconnect(self) -> None:
        self.connected = False


class TurnTakingManager:
    def __init__(self, silence_timeout_s: float = 1.0) -> None:
        self.silence_timeout_s = silence_timeout_s

    def update_speaking(self, state: ParticipantState, is_speaking: bool, timestamp: float) -> None:
        if state.speaking and not is_speaking:
            state.last_speech_end_ts = timestamp
        state.speaking = is_speaking

    def should_respond(self, state: ParticipantState, now: float) -> bool:
        if state.speaking:
            return False
        if state.last_speech_end_ts is None:
            return False
        return (now - state.last_speech_end_ts) >= self.silence_timeout_s


class LiveKitAgent:
    def __init__(
        self,
        silence_timeout_s: float = 1.0,
        max_response_chars: int = 500,
        response_cooldown_s: float = 0.5,
        publish_response: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        self.participants: Dict[str, ParticipantState] = {}
        self.turn_taking = TurnTakingManager(silence_timeout_s)
        self.max_response_chars = max_response_chars
        self.response_cooldown_s = response_cooldown_s
        self.publish_response = publish_response or self._noop_publish

    def _now(self) -> float:
        return time.time()

    def participant_join(self, participant_id: str) -> None:
        state = self.participants.get(participant_id)
        if state is None:
            self.participants[participant_id] = ParticipantState(participant_id=participant_id)
        else:
            state.connected = True
        logger.info("participant_join", extra={"participant_id": participant_id})

    def participant_leave(self, participant_id: str) -> None:
        state = self.participants.get(participant_id)
        if state:
            state.mark_disconnect()
        logger.info("participant_leave", extra={"participant_id": participant_id})

    def handle_vad(self, participant_id: str, is_speaking: bool, timestamp: Optional[float] = None) -> None:
        state = self._get_state(participant_id)
        now = timestamp or self._now()
        self.turn_taking.update_speaking(state, is_speaking, now)
        logger.info(
            "participant_speaking",
            extra={"participant_id": participant_id, "is_speaking": is_speaking},
        )

    def handle_transcript(self, participant_id: str, text: str, timestamp: Optional[float] = None) -> None:
        state = self._get_state(participant_id)
        now = timestamp or self._now()
        state.last_utterance_ts = now
        if not state.speaking:
            state.last_speech_end_ts = now
        state.conversation_context.append(text)
        logger.info(
            "transcript_received",
            extra={"participant_id": participant_id, "text": text},
        )

    def maybe_respond(self, participant_id: str, model_response: str) -> Optional[str]:
        state = self._get_state(participant_id)
        now = self._now()

        if not state.connected:
            logger.warning(
                "response_cancelled_disconnect",
                extra={"participant_id": participant_id},
            )
            return None

        if not self.turn_taking.should_respond(state, now):
            return None

        if not self._cooldown_elapsed(state, now):
            logger.warning(
                "response_cooldown_active",
                extra={"participant_id": participant_id},
            )
            return None

        response = self._apply_max_length(model_response)
        state.last_response_ts = now
        logger.info(
            "model_response",
            extra={"participant_id": participant_id, "response": response},
        )

        if self.publish_response(participant_id, response):
            logger.info(
                "publish_success",
                extra={"participant_id": participant_id},
            )
        else:
            logger.error(
                "publish_failure",
                extra={"participant_id": participant_id},
            )
        return response

    @staticmethod
    def _noop_publish(participant_id: str, response: str) -> bool:
        _ = participant_id
        _ = response
        return True

    def _apply_max_length(self, response: str) -> str:
        if len(response) <= self.max_response_chars:
            return response
        truncated = response[: self.max_response_chars].rstrip()
        logger.warning(
            "response_truncated",
            extra={"max_response_chars": self.max_response_chars},
        )
        return truncated

    def _cooldown_elapsed(self, state: ParticipantState, now: float) -> bool:
        if state.last_response_ts is None:
            return True
        return (now - state.last_response_ts) >= self.response_cooldown_s

    def _get_state(self, participant_id: str) -> ParticipantState:
        if participant_id not in self.participants:
            self.participants[participant_id] = ParticipantState(participant_id=participant_id)
        return self.participants[participant_id]
