from __future__ import annotations

from dataclasses import dataclass

from agent.pipeline.interfaces import AudioFrame, AudioTrack, LanguageModel, LiveKitRoom, SpeechToText, TextToSpeech


@dataclass
class LiveKitPipeline:
    room: LiveKitRoom
    stt: SpeechToText
    llm: LanguageModel
    tts: TextToSpeech

    def start(self) -> None:
        self.room.on_audio_track(self._handle_track)

    def _handle_track(self, track: AudioTrack) -> None:
        track.on_frame(self._handle_frame)
        track.on_end(self._handle_end)

    def _handle_frame(self, frame: AudioFrame) -> None:
        self.stt.feed(frame)

    def _handle_end(self) -> None:
        transcript = self.stt.finalize()
        response = self.llm.generate(transcript)
        audio = self.tts.synthesize(response)
        self.room.publish_audio(audio)
