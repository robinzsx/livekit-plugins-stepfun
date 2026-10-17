"""Verify audio framing and input lifecycle without downloading a native model."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from livekit.agents.voice import io

from livekit import rtc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from _noise_cancellation import FilteredAudioInput, NoiseSuppressor  # noqa: E402


class Backend:
    def __init__(self):
        self.blocks = []
        self.closed = 0

    def process(self, samples):
        self.blocks.append(samples.copy())
        return samples.copy()

    def close(self):
        self.closed += 1


def frame(samples, *, rate=48000, channels=1):
    data = np.asarray(samples, dtype=np.int16).tobytes()
    return rtc.AudioFrame(data, rate, channels, len(data) // 2 // channels)


def samples(frames):
    return np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in frames])


@pytest.mark.parametrize("rate", [16000, 24000, 44100, 48000])
def test_resampling_preserves_silence_duration(rate):
    backend = Backend()
    processor = NoiseSuppressor(backend=backend)
    output = []
    for _ in range(10):
        output.extend(processor.push(frame(np.zeros(rate // 10, dtype=np.int16), rate=rate)))
    output.extend(processor.flush())
    assert sum(f.samples_per_channel for f in output) == 48000
    assert all(f.sample_rate == 48000 and f.num_channels == 1 for f in output)
    # The SDK's int16 resampler dithers silence by approximately one LSB.
    assert np.abs(samples(output).astype(np.int32)).max() <= 2
    assert len(backend.blocks) == 100
    processor.close()
    processor.close()
    assert backend.closed == 1


def test_partial_blocks_are_buffered_and_tail_is_trimmed():
    backend = Backend()
    processor = NoiseSuppressor(backend=backend)
    original = np.arange(1001, dtype=np.int16)
    assert processor.push(frame(original[:200])) == []
    output = processor.push(frame(original[200:]))
    assert sum(f.samples_per_channel for f in output) == 960
    output.extend(processor.flush())
    np.testing.assert_array_equal(samples(output), original)
    assert all(len(block) == 480 for block in backend.blocks)
    assert not backend.blocks[-1][41:].any()
    processor.close()


def test_stereo_is_mixed_without_overflow():
    processor = NoiseSuppressor(backend=Backend())
    output = processor.push(frame(np.tile([30000, 10000], 480), channels=2))
    np.testing.assert_array_equal(samples(output), np.full(480, 20000, dtype=np.int16))
    processor.close()


def test_format_change_drains_previous_resampler():
    processor = NoiseSuppressor(backend=Backend())
    output = processor.push(frame(np.zeros(1600, dtype=np.int16), rate=16000))
    output += processor.push(frame(np.zeros(2400, dtype=np.int16), rate=24000))
    output += processor.flush()
    assert sum(f.samples_per_channel for f in output) == 9600
    processor.close()


def test_missing_optional_dependency_has_install_instructions(monkeypatch):
    original = importlib.import_module

    def missing(name):
        if name == "pyrnnoise.rnnoise":
            raise ModuleNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="uv sync --extra examples --extra noise-cancellation"):
        NoiseSuppressor()


class Feed(io.AudioInput):
    def __init__(self, frames):
        super().__init__(label="fixture input")
        self._frames = iter(frames)
        self.attached = False

    async def __anext__(self):
        try:
            return next(self._frames)
        except StopIteration:
            raise StopAsyncIteration from None

    def on_attached(self):
        self.attached = True

    def on_detached(self):
        self.attached = False


async def test_input_flushes_tail_and_forwards_lifecycle():
    backend = Backend()
    original = np.arange(1001, dtype=np.int16)
    source = Feed([frame(original)])
    filtered = FilteredAudioInput(source, processor=NoiseSuppressor(backend=backend))
    filtered.on_attached()
    assert source.attached
    output = [f async for f in filtered]
    np.testing.assert_array_equal(samples(output), original)
    assert backend.closed == 1
    filtered.on_detached()
    assert not source.attached
    await filtered.aclose()
    assert backend.closed == 1


async def test_cancellation_waits_for_native_processing_before_destroy():
    started, release = threading.Event(), threading.Event()

    class SlowBackend(Backend):
        def process(self, samples):
            started.set()
            assert release.wait(3)
            assert self.closed == 0
            return super().process(samples)

    backend = SlowBackend()
    filtered = FilteredAudioInput(
        Feed([frame(np.zeros(480, dtype=np.int16))]), processor=NoiseSuppressor(backend=backend)
    )
    reader = asyncio.create_task(filtered.__anext__())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        closing = asyncio.create_task(filtered.aclose())
        await asyncio.sleep(0)
        assert backend.closed == 0
        release.set()
        await asyncio.wait_for(closing, 2)
        assert backend.closed == 1
    finally:
        release.set()
        await filtered.aclose()


async def test_native_rnnoise_when_installed():
    pytest.importorskip("pyrnnoise")
    processor = NoiseSuppressor()
    filtered = FilteredAudioInput(
        Feed([frame(np.zeros(4800, dtype=np.int16))]), processor=processor
    )
    try:
        output = [f async for f in filtered]
        assert sum(f.samples_per_channel for f in output) == 4800
        assert np.isfinite(samples(output)).all()
    finally:
        await filtered.aclose()


@pytest.mark.parametrize("streaming", [True, False])
async def test_shared_filter_feeds_cleaned_audio_to_stt_vad_and_turns(streaming, monkeypatch):
    import base64

    from livekit.agents import AgentSession, inference
    from livekit.agents.inference.eot import transports

    from livekit.plugins import stepfun

    from .fake_ws import FakePool, FakeWebSocket
    from .test_stepfun_session import AudioFeed, ScriptedVAD, TurnProbe
    from .test_stepfun_stt import _FakeSSESession

    seen_vad, seen_turns = [], []

    class Attenuator(Backend):
        def process(self, values):
            return values // 2

    class RecordingVAD(ScriptedVAD):
        def stream(self):
            stream = super().stream()
            original_push = stream.push_frame

            def push(value):
                seen_vad.append(np.frombuffer(value.data, dtype=np.int16).copy())
                original_push(value)

            stream.push_frame = push
            return stream

    class LocalModel:
        def predict(self, audio):
            seen_turns.append(audio.copy())
            return 1.0

    monkeypatch.setattr(transports, "_EOT", LocalModel)
    transport = _FakeSSESession(['data: {"type":"transcript.text.done","text":"hello"}'])
    stt = stepfun.STT(api_key="test-key", use_streaming=streaming, http_session=transport)

    class ObservedWebSocket(FakeWebSocket):
        def __init__(self):
            super().__init__()
            self.audio_sent = asyncio.Event()

        async def send_str(self, data):
            await super().send_str(data)
            if self.frames_of_type("input_audio_buffer.append"):
                self.audio_sent.set()

    ws = ObservedWebSocket()
    ws.queue_server_event({"type": "session.updated"})
    stt._pool = FakePool(ws)
    feed, probe = AudioFeed(), TurnProbe()
    session = AgentSession(
        stt=stt,
        vad=RecordingVAD(),
        turn_handling={
            "turn_detection": "stt" if streaming else inference.TurnDetector(version="v1-mini"),
            "interruption": {"mode": "vad"},
            "endpointing": {"min_delay": 0.0},
        },
    )
    session.input.audio = feed
    filtered = FilteredAudioInput(feed, processor=NoiseSuppressor(backend=Attenuator()))
    try:
        await session.start(agent=probe)
        session.input.audio = filtered
        feed.frames.send_nowait(frame(np.full(4800, 4000, dtype=np.int16)))
        feed.frames.send_nowait(frame(np.zeros(4800, dtype=np.int16)))
        if streaming:
            await asyncio.wait_for(ws.audio_sent.wait(), 3)
            for event in [
                {"type": "input_audio_buffer.speech_started"},
                {"type": "input_audio_buffer.speech_stopped"},
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "hello",
                    "language": "en",
                },
            ]:
                ws.queue_server_event(event)
        assert await asyncio.wait_for(probe.turns.get(), 3) == "hello"
        assert seen_vad and max(int(v.max()) for v in seen_vad) == 2000
        if streaming:
            audio = b"".join(
                base64.b64decode(e["audio"]) for e in ws.frames_of_type("input_audio_buffer.append")
            )
        else:
            audio = base64.b64decode(transport.json["audio"]["data"])
            assert seen_turns and max(v.max() for v in seen_turns) < 2500
        assert 1000 < np.frombuffer(audio, dtype=np.int16).max() < 2500
    finally:
        await session.aclose()
        await filtered.aclose()
        await stt.aclose()
        feed.frames.close()


async def test_processing_failure_is_reported_and_releases_native_state():
    class BrokenBackend(Backend):
        def process(self, values):
            raise RuntimeError("native processing failed")

    backend = BrokenBackend()
    filtered = FilteredAudioInput(
        Feed([frame(np.zeros(480, dtype=np.int16))]), processor=NoiseSuppressor(backend=backend)
    )
    with pytest.raises(RuntimeError, match="native processing failed"):
        await filtered.__anext__()
    assert backend.closed == 1


def test_processor_can_resume_resampling_after_flush():
    processor = NoiseSuppressor(backend=Backend())
    output = []
    for _ in range(2):
        output.extend(processor.push(frame(np.zeros(1600, dtype=np.int16), rate=16000)))
        output.extend(processor.flush())
    assert sum(f.samples_per_channel for f in output) == 9600
    processor.close()


async def test_native_webrtc_suppresses_stationary_noise_and_preserves_duration():
    rng = np.random.default_rng(1)
    noisy = np.clip(rng.normal(0, 1000, 48000), -32768, 32767).astype(np.int16)
    processor = NoiseSuppressor(method="webrtc")
    filtered = FilteredAudioInput(Feed([frame(noisy)]), processor=processor)
    try:
        output = [value async for value in filtered]
        cleaned = samples(output)
        assert len(cleaned) == len(noisy)
        # Allow model warmup, then check actual native processing reduced noise.
        before = np.sqrt(np.mean(noisy[24000:].astype(float) ** 2))
        after = np.sqrt(np.mean(cleaned[24000:].astype(float) ** 2))
        assert after < before * 0.7
    finally:
        await filtered.aclose()
