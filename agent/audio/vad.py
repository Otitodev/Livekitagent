"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VADEvent(Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VADConfig:
    sample_rate: int = 16000
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 100
    window_size_samples: int = 512  # 32ms at 16kHz


VADCallback = Callable[[VADEvent], None]


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD model.

    Processes audio chunks and detects speech start/end events.
    Runs efficiently on CPU with minimal latency.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._model = None
        self._callback: Optional[VADCallback] = None

        # State tracking
        self._is_speaking = False
        self._speech_samples = 0
        self._silence_samples = 0

        # Samples needed for thresholds
        self._min_speech_samples = int(
            self.config.min_speech_duration_ms * self.config.sample_rate / 1000
        )
        self._min_silence_samples = int(
            self.config.min_silence_duration_ms * self.config.sample_rate / 1000
        )

    def load_model(self) -> None:
        """Load the Silero VAD model. Call once at startup."""
        if self._model is not None:
            return

        logger.info("Loading Silero VAD model...")
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model.eval()
        logger.info("Silero VAD model loaded")

    def set_callback(self, callback: VADCallback) -> None:
        """Set callback for VAD events."""
        self._callback = callback

    def reset(self) -> None:
        """Reset VAD state for a new session."""
        self._is_speaking = False
        self._speech_samples = 0
        self._silence_samples = 0
        if self._model is not None:
            self._model.reset_states()

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently detected."""
        return self._is_speaking

    def process_chunk(self, audio_chunk: bytes) -> Optional[VADEvent]:
        """
        Process an audio chunk and detect speech activity.

        Args:
            audio_chunk: Raw PCM 16-bit audio bytes

        Returns:
            VADEvent if a state change occurred, None otherwise
        """
        if self._model is None:
            raise RuntimeError("VAD model not loaded. Call load_model() first.")

        # Convert bytes to float32 tensor
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float32)

        # Process in windows
        event = None
        window_size = self.config.window_size_samples

        for i in range(0, len(audio_tensor), window_size):
            window = audio_tensor[i : i + window_size]
            if len(window) < window_size:
                # Pad last window if needed
                window = torch.nn.functional.pad(window, (0, window_size - len(window)))

            # Get speech probability
            speech_prob = self._model(window, self.config.sample_rate).item()

            # Update state based on probability
            window_event = self._update_state(speech_prob, len(window))
            if window_event is not None:
                event = window_event

        return event

    def _update_state(self, speech_prob: float, num_samples: int) -> Optional[VADEvent]:
        """Update internal state based on speech probability."""
        is_speech = speech_prob >= self.config.threshold

        if is_speech:
            self._speech_samples += num_samples
            self._silence_samples = 0

            # Trigger speech start after enough speech detected
            if not self._is_speaking and self._speech_samples >= self._min_speech_samples:
                self._is_speaking = True
                logger.debug(f"VAD: Speech start (prob={speech_prob:.2f})")
                if self._callback:
                    self._callback(VADEvent.SPEECH_START)
                return VADEvent.SPEECH_START
        else:
            self._silence_samples += num_samples

            # Trigger speech end after enough silence detected
            if self._is_speaking and self._silence_samples >= self._min_silence_samples:
                self._is_speaking = False
                self._speech_samples = 0
                logger.debug(f"VAD: Speech end (silence={self._silence_samples})")
                if self._callback:
                    self._callback(VADEvent.SPEECH_END)
                return VADEvent.SPEECH_END

        return None

    def process_chunk_simple(self, audio_chunk: bytes) -> float:
        """
        Process audio and return speech probability (simpler interface).

        Args:
            audio_chunk: Raw PCM 16-bit audio bytes

        Returns:
            Speech probability between 0.0 and 1.0
        """
        if self._model is None:
            raise RuntimeError("VAD model not loaded. Call load_model() first.")

        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float32)

        # Ensure correct window size
        window_size = self.config.window_size_samples
        if len(audio_tensor) < window_size:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor, (0, window_size - len(audio_tensor))
            )
        elif len(audio_tensor) > window_size:
            audio_tensor = audio_tensor[:window_size]

        return self._model(audio_tensor, self.config.sample_rate).item()
