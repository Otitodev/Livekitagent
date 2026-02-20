"""Streaming OpenAI LLM implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, List, Optional

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 300


SentenceCallback = Callable[[str], None]


class StreamingOpenAI:
    """
    Streaming OpenAI LLM with sentence-level batching for TTS.

    Streams tokens from OpenAI and emits complete sentences,
    which can be immediately sent to TTS for synthesis.
    """

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=config.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.APIConnectionError, openai.APITimeoutError)),
    )
    async def generate_stream(
        self,
        messages: List[dict[str, str]],
        on_sentence: Optional[SentenceCallback] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response, yielding complete sentences.

        Args:
            messages: Conversation history in OpenAI format
            on_sentence: Optional callback called for each sentence

        Yields:
            Complete sentences as they become available
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                stream=True,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            buffer = ""
            sentence_terminators = ".!?"

            async for chunk in response:
                token = chunk.choices[0].delta.content or ""
                buffer += token

                # Check for complete sentences
                while any(term in buffer for term in sentence_terminators):
                    # Find the first terminator
                    indices = [
                        buffer.find(term)
                        for term in sentence_terminators
                        if term in buffer
                    ]
                    first_term_idx = min(i for i in indices if i >= 0)

                    # Extract sentence (include terminator)
                    sentence = buffer[: first_term_idx + 1].strip()
                    buffer = buffer[first_term_idx + 1 :].lstrip()

                    if sentence:
                        logger.debug(f"LLM sentence: {sentence}")
                        if on_sentence:
                            on_sentence(sentence)
                        yield sentence

            # Yield any remaining text
            if buffer.strip():
                logger.debug(f"LLM final: {buffer.strip()}")
                if on_sentence:
                    on_sentence(buffer.strip())
                yield buffer.strip()

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def generate(self, messages: List[dict[str, str]]) -> str:
        """
        Generate a complete response (non-streaming).

        Useful for fallback or simple use cases.
        """
        sentences = []
        async for sentence in self.generate_stream(messages):
            sentences.append(sentence)
        return " ".join(sentences)

    async def generate_with_system(
        self,
        system_prompt: str,
        conversation: List[dict[str, str]],
        on_sentence: Optional[SentenceCallback] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate with a system prompt prepended.

        Args:
            system_prompt: System instructions
            conversation: User/assistant message history
            on_sentence: Optional sentence callback

        Yields:
            Complete sentences
        """
        messages = [{"role": "system", "content": system_prompt}] + conversation
        async for sentence in self.generate_stream(messages, on_sentence):
            yield sentence
