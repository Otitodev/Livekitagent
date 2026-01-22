from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional
from urllib import request


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    endpoint: str = "https://api.openai.com/v1/chat/completions"


class OpenAILLM:
    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config

    def generate(self, messages: List[dict[str, str]]) -> Optional[str]:
        body = json.dumps(
            {
                "model": self.config.model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 160,
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        req = request.Request(self.config.endpoint, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return _extract_content(payload)


def _extract_content(payload: dict) -> Optional[str]:
    choices = payload.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content")
    return content.strip() if content else None
