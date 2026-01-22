from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from agent.pipeline.interfaces import AudioFrame, AudioTrack, LiveKitRoom
from agent.pipeline.pipeline import LiveKitPipeline
from agent.pipeline.providers import EchoLanguageModel, EchoSpeechToText, ToneTextToSpeech


@dataclass
class FakeAudioTrack(AudioTrack):
    frames: List[AudioFrame]
    frame_handlers: List[Callable[[AudioFrame], None]] = field(default_factory=list)
    end_handlers: List[Callable[[], None]] = field(default_factory=list)

    def on_frame(self, handler: Callable[[AudioFrame], None]) -> None:
        self.frame_handlers.append(handler)

    def on_end(self, handler: Callable[[], None]) -> None:
        self.end_handlers.append(handler)

    def play(self) -> None:
        for frame in self.frames:
            for handler in self.frame_handlers:
                handler(frame)
        for handler in self.end_handlers:
            handler()


@dataclass
class FakeLiveKitRoom(LiveKitRoom):
    audio_track_handlers: List[Callable[[AudioTrack], None]] = field(default_factory=list)
    published_audio: List[AudioFrame] = field(default_factory=list)

    def on_audio_track(self, handler: Callable[[AudioTrack], None]) -> None:
        self.audio_track_handlers.append(handler)

    def publish_audio(self, frame: AudioFrame) -> None:
        self.published_audio.append(frame)

    def emit_audio_track(self, track: AudioTrack) -> None:
        for handler in self.audio_track_handlers:
            handler(track)


if __name__ == "__main__":
    room = FakeLiveKitRoom()
    pipeline = LiveKitPipeline(
        room=room,
        stt=EchoSpeechToText(),
        llm=EchoLanguageModel(),
        tts=ToneTextToSpeech(),
    )
    pipeline.start()

    track = FakeAudioTrack(
        frames=[AudioFrame(data=b"audio", sample_rate_hz=16000)],
    )
    room.emit_audio_track(track)
    track.play()

    print(room.published_audio)
