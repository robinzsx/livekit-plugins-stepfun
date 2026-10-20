"""Offline noise suppression shared by console and room sessions.

RNNoise operates on 10 ms blocks of mono 48 kHz PCM. Filter before the session
fans input out to STT, VAD and turn detection; never gate on speech probability.
"""

from __future__ import annotations

import asyncio
import importlib
from collections import deque
from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np
from livekit.agents.voice import io
from numpy.typing import NDArray

from livekit import rtc

_RATE = 48000
_SAMPLES = 480
_BYTES = _SAMPLES * 2
_T = TypeVar("_T")


class _Backend(Protocol):
    def process(self, samples: NDArray[np.int16]) -> NDArray[np.int16]: ...
    def close(self) -> None: ...


class _RNNoiseBackend:
    def __init__(self) -> None:
        try:
            self._native = importlib.import_module("pyrnnoise.rnnoise")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Local RNNoise is unavailable. Install the optional dependency with "
                "`uv sync --extra examples --extra noise-cancellation`, or set "
                "STEPFUN_NOISE_CANCELLATION=off."
            ) from exc
        if self._native.FRAME_SIZE != _SAMPLES:
            raise RuntimeError("Unsupported RNNoise frame size; expected 480 samples")
        self._state = self._native.create()
        if not self._state:
            raise RuntimeError("RNNoise could not allocate a denoising state")

    def process(self, samples: NDArray[np.int16]) -> NDArray[np.int16]:
        if self._state is None:
            raise RuntimeError("RNNoise is closed")
        # Ignore speech probability: even silence must reach server VAD.
        cleaned, _probability = self._native.process_mono_frame(self._state, samples)
        return np.asarray(cleaned, dtype=np.int16)

    def close(self) -> None:
        if self._state is not None:
            self._native.destroy(self._state)
            self._state = None


class _WebRTCBackend:
    def __init__(self) -> None:
        self._apm: rtc.AudioProcessingModule | None = rtc.AudioProcessingModule(
            noise_suppression=True,
        )

    def process(self, samples: NDArray[np.int16]) -> NDArray[np.int16]:
        if self._apm is None:
            raise RuntimeError("WebRTC noise suppression is closed")
        frame = rtc.AudioFrame(samples.tobytes(), _RATE, 1, _SAMPLES)
        self._apm.process_stream(frame)
        cleaned: NDArray[np.int16] = np.frombuffer(frame.data, dtype=np.int16).copy()
        return cleaned

    def close(self) -> None:
        self._apm = None  # Release the SDK-owned native handle.


class NoiseSuppressor:
    """Stateful processor, owned by exactly one session and called serially."""

    def __init__(self, *, method: str = "rnnoise", backend: _Backend | None = None) -> None:
        if method not in ("rnnoise", "webrtc"):
            raise ValueError("Noise suppression method must be 'rnnoise' or 'webrtc'")
        self.method = method
        self._backend = (
            backend
            if backend is not None
            else (_RNNoiseBackend() if method == "rnnoise" else _WebRTCBackend())
        )
        self._resampler: rtc.AudioResampler | None = None
        self._format: tuple[int, int] | None = None
        self._buffer = bytearray()
        self._closed = False

    def push(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        if self._closed:
            raise RuntimeError("Noise suppression is closed")
        if frame.sample_rate <= 0 or frame.num_channels <= 0:
            raise ValueError("Audio sample rate and channel count must be positive")
        result: list[rtc.AudioFrame] = []
        fmt = (frame.sample_rate, frame.num_channels)
        if fmt != self._format:
            result.extend(self.flush())
            self._format = fmt
            self._resampler = (
                rtc.AudioResampler(
                    frame.sample_rate,
                    _RATE,
                    num_channels=1,
                    quality=rtc.AudioResamplerQuality.QUICK,
                )
                if frame.sample_rate != _RATE
                else None
            )
        if frame.num_channels != 1:
            samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
            mono = samples.mean(axis=1).astype(np.int16)
            frame = rtc.AudioFrame(mono.tobytes(), frame.sample_rate, 1, frame.samples_per_channel)
        frames = self._resampler.push(frame) if self._resampler else [frame]
        for resampled in frames:
            self._buffer.extend(resampled.data.cast("B"))
        result.extend(self._drain())
        return result

    def _drain(self, *, final: bool = False) -> list[rtc.AudioFrame]:
        frames = []
        while len(self._buffer) >= _BYTES or (final and self._buffer):
            count = min(len(self._buffer), _BYTES)
            block = bytes(self._buffer[:count])
            del self._buffer[:count]
            # Only pad the last incomplete block, then trim the padding again.
            samples = np.frombuffer(block.ljust(_BYTES, b"\0"), dtype=np.int16)
            cleaned = self._backend.process(samples)
            if cleaned.shape != (_SAMPLES,):
                raise RuntimeError("Noise suppression returned an invalid audio block")
            frames.append(rtc.AudioFrame(cleaned.tobytes()[:count], _RATE, 1, count // 2))
        return frames

    def flush(self) -> list[rtc.AudioFrame]:
        if self._resampler is not None:
            for frame in self._resampler.flush():
                self._buffer.extend(frame.data.cast("B"))
            self._resampler = None
        self._format = None
        return self._drain(final=True)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._backend.close()
            self._buffer.clear()
            self._resampler = None


class FilteredAudioInput(io.AudioInput):
    """Filter any LiveKit audio input, including the local console's input."""

    def __init__(self, source: io.AudioInput, *, processor: NoiseSuppressor) -> None:
        super().__init__(label=f"Local {processor.method} suppression", source=source)
        self._upstream = source
        self._processor = processor
        self._pending: deque[rtc.AudioFrame] = deque()
        self._ended = False
        self._closed = False
        self._inflight: asyncio.Task | None = None
        self._close_task: asyncio.Task[None] | None = None

    async def _work(self, fn: Callable[..., _T], *args: object) -> _T:
        # Shield keeps a cancelled reader from abandoning a native call; aclose
        # waits for it before releasing the native state. One worker at a time.
        task = asyncio.create_task(asyncio.to_thread(fn, *args))
        self._inflight = task
        return await asyncio.shield(task)

    async def __anext__(self) -> rtc.AudioFrame:
        try:
            return await self._read_frame()
        except Exception:
            await self.aclose()
            raise

    async def _read_frame(self) -> rtc.AudioFrame:
        while not self._pending:
            if self._ended or self._closed:
                raise StopAsyncIteration
            try:
                frame = await self._upstream.__anext__()
            except StopAsyncIteration:
                self._ended = True
                self._pending.extend(await self._work(self._processor.flush))
                await self.aclose()
            else:
                self._pending.extend(await self._work(self._processor.push, frame))
        if self._closed and not self._ended:
            raise StopAsyncIteration
        return self._pending.popleft()

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        if self._inflight is not None:
            await asyncio.gather(self._inflight, return_exceptions=True)
        self._processor.close()
        if not self._ended:
            self._pending.clear()


class ResamplingAudioInput(io.AudioInput):
    """Downsample any LiveKit audio input to a target rate.

    Placed after noise cancellation (which must run at 48 kHz) so the VAD, turn
    detector, and ASR downstream all operate at one lower rate (16 kHz): smaller
    segment buffers, and the batch ASR upload already matches. Passthrough when
    the source is already at the target rate.
    """

    def __init__(self, source: io.AudioInput, *, sample_rate: int) -> None:
        super().__init__(label=f"Resample to {sample_rate} Hz", source=source)
        self._upstream = source
        self._target = sample_rate
        self._resampler: rtc.AudioResampler | None = None
        self._pending: deque[rtc.AudioFrame] = deque()
        self._ended = False
        self._closed = False

    def _resample(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        if frame.sample_rate == self._target:
            return [frame]
        if self._resampler is None:
            self._resampler = rtc.AudioResampler(
                frame.sample_rate, self._target, num_channels=frame.num_channels
            )
        return list(self._resampler.push(frame))

    async def __anext__(self) -> rtc.AudioFrame:
        try:
            return await self._read_frame()
        except Exception:
            await self.aclose()
            raise

    async def _read_frame(self) -> rtc.AudioFrame:
        while not self._pending:
            if self._ended or self._closed:
                raise StopAsyncIteration
            try:
                frame = await self._upstream.__anext__()
            except StopAsyncIteration:
                self._ended = True
                if self._resampler is not None:
                    self._pending.extend(self._resampler.flush())
            else:
                self._pending.extend(self._resample(frame))
        return self._pending.popleft()

    async def aclose(self) -> None:
        self._closed = True
