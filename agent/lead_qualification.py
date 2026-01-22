from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class QualificationSession:
    participant_id: str
    step_index: int = 0
    answers: Dict[str, str] = field(default_factory=dict)
    qualified: Optional[bool] = None

    def record_answer(self, key: str, value: str) -> None:
        self.answers[key] = value


class LeadQualificationFlow:
    def __init__(self) -> None:
        self.sessions: Dict[str, QualificationSession] = {}
        self.questions = [
            ("intent", "Thanks for calling! What service are you looking for today?"),
            ("location", "What city are you located in?"),
            ("timeline", "When are you hoping to get this done?"),
            ("budget", "Do you have an approximate budget range in mind?"),
            ("contact", "What’s the best phone number and email for confirmation?"),
        ]

    def start_session(self, participant_id: str) -> QualificationSession:
        session = QualificationSession(participant_id=participant_id)
        self.sessions[participant_id] = session
        return session

    def end_session(self, participant_id: str) -> None:
        self.sessions.pop(participant_id, None)

    def handle_message(self, participant_id: str, message: str) -> str:
        session = self.sessions.get(participant_id)
        if session is None:
            session = self.start_session(participant_id)
            return self.questions[0][1]

        if session.step_index < len(self.questions):
            key, _ = self.questions[session.step_index]
            session.record_answer(key, message)
            session.step_index += 1

        if session.step_index < len(self.questions):
            return self.questions[session.step_index][1]

        if session.qualified is None:
            session.qualified = self._evaluate_qualification(session)

        if session.qualified:
            return (
                "Great, you’re a fit. I can book you for tomorrow at 2 PM or Friday at 11 AM. "
                "Which works best?"
            )
        return (
            "Thanks for the details. We’re not the best fit for this request, "
            "but we can connect you with a specialist if you’d like."
        )

    @staticmethod
    def _evaluate_qualification(session: QualificationSession) -> bool:
        timeline = session.answers.get("timeline", "").lower()
        budget = session.answers.get("budget", "").lower()
        if any(word in timeline for word in ("today", "this week", "asap")):
            return True
        if any(word in budget for word in ("2k", "5k", "10k", "thousand", "$")):
            return True
        return False
