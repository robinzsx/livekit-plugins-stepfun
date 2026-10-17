"""Decode WAV inputs and deliver a continuous, paced microphone replacement."""

from __future__ import annotations

import asyncio
import io as bytes_io
import time
from dataclasses import dataclass

import av
from _eval_metrics import TurnRecord
from livekit.agents.voice import io

from livekit import rtc

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
MAX_AUDIO_SECONDS = 120


def decode_audio(data: bytes) -> bytes:
    """Normalize audio to mono PCM16/16 kHz, bounding decoded duration."""
    pcm = bytearray()
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    try:
        with av.open(bytes_io.BytesIO(data)) as container:
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    pcm.extend(converted.to_ndarray().tobytes())
                if len(pcm) > MAX_AUDIO_SECONDS * SAMPLE_RATE * 2:
                    raise ValueError(f"Audio exceeds {MAX_AUDIO_SECONDS} seconds")
            for converted in resampler.resample(None):
                pcm.extend(converted.to_ndarray().tobytes())
    except (av.error.FFmpegError, IndexError) as exc:
        raise ValueError("Audio could not be decoded") from exc
    if not pcm:
        raise ValueError("Audio contains no samples")
    if len(pcm) > MAX_AUDIO_SECONDS * SAMPLE_RATE * 2:
        raise ValueError(f"Audio exceeds {MAX_AUDIO_SECONDS} seconds")
    return bytes(pcm)


@dataclass
class Playback:
    pcm: bytes
    record: TurnRecord
    finished: asyncio.Event
    speech_start: float | None = None
    speech_end: float | None = None
    offset: int = 0


class ReplayAudioInput(io.AudioInput):
    """Continue sending silence between clips so server VAD can finish each turn."""

    def __init__(self) -> None:
        super().__init__(label="Evaluation dataset")
        self.playback: Playback | None = None
        self._due = 0.0
        self._closed = False

    async def __anext__(self) -> rtc.AudioFrame:
        if self._closed:
            raise StopAsyncIteration
        await asyncio.sleep(max(0.0, self._due - time.perf_counter()))
        now = time.perf_counter()
        if self._closed:
            raise StopAsyncIteration
        current = self.playback
        if current and current.offset >= len(current.pcm):
            current.record.note("input_end", at=self._due)
            current.finished.set()
            self.playback = current = None
        if current is None:
            data = bytes(FRAME_SAMPLES * 2)
        else:
            record = current.record
            record.max_schedule_lag_ms = max(
                record.max_schedule_lag_ms, max(0, now - self._due) * 1000 if self._due else 0
            )
            if current.offset == 0:
                record.note("input_start")
            data = current.pcm[current.offset : current.offset + FRAME_SAMPLES * 2]
            start = current.offset / (SAMPLE_RATE * 2)
            end = start + len(data) / (SAMPLE_RATE * 2)
            for name, boundary in (
                ("speech_start", current.speech_start),
                ("speech_end", current.speech_end),
            ):
                if boundary is not None and name not in record.marks and start <= boundary <= end:
                    record.note(name, at=now + boundary - start)
            current.offset += len(data)
            record.frames_submitted += 1
        samples = len(data) // 2
        self._due = now + samples / SAMPLE_RATE
        return rtc.AudioFrame(data, SAMPLE_RATE, 1, samples)

    def clear(self) -> None:
        self.playback = None

    def close(self) -> None:
        self._closed = True
        self.clear()


class ObservedAudioOutput(io.AudioOutput):
    def __init__(self, output: io.AudioOutput, *, get_record) -> None:
        super().__init__(
            label="Evaluation output",
            next_in_chain=output,
            capabilities=io.AudioOutputCapabilities(pause=output.can_pause),
            sample_rate=output.sample_rate,
        )
        self._get_record = get_record

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        record = self._get_record()
        if record is not None and "output_first_audio" not in record.marks:
            record.note("output_first_audio")
        await self.next_in_chain.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        super().flush()
        self.next_in_chain.clear_buffer()
