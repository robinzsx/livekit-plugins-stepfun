"""Regressions for option updates, transport errors, and resource cleanup."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import aiohttp
import pytest
from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.agents.types import NOT_GIVEN

from livekit.plugins import stepfun
from livekit.plugins.stepfun.tts import _await_connection_done, _Connection

from .fake_ws import FakePool, FakeWebSocket
from .test_stepfun_stt import _frame
from .test_stepfun_tts import _AUDIO_DONE, _CREATED, _SESSION, _audio_delta, _FakeHTTPSession

NO_RETRY = APIConnectOptions(max_retry=0)


@pytest.mark.parametrize(
    "options",
    [
        {"speed": 0.1},
        {"volume": 3.0},
        {"instruction": "x" * 501},
        {"model": "stepaudio-2.5-tts", "instruction": "x" * 201},
    ],
)
def test_invalid_tts_update_is_atomic(options):
    tts = stepfun.TTS()
    previous = tts._opts
    with pytest.raises(ValueError):
        tts.update_options(voice="lively-girl", **options)
    assert tts._opts == previous


def test_model_update_checks_retained_instruction():
    tts = stepfun.TTS(instruction="x" * 300)
    with pytest.raises(ValueError, match="200 characters"):
        tts.update_options(model="stepaudio-2.5-tts")
    assert tts.model == "stepaudio-3-tts"
    tts.update_options(model="step-tts-mini")
    assert tts._opts.instruction is NOT_GIVEN


@pytest.mark.parametrize("component", [stepfun.STT, stepfun.TTS])
def test_explicit_cluster_is_validated(component):
    with pytest.raises(ValueError, match="STEPFUN_CLUSTER"):
        component(cluster="invalid")


async def test_tts_handshake_surfaces_transport_error():
    ws = FakeWebSocket()
    ws.queue_error()
    with pytest.raises(APIConnectionError, match="websocket error"):
        await asyncio.wait_for(_await_connection_done(ws), 1)


@pytest.mark.parametrize("configured", [False, True])
async def test_stt_transport_error_closes_socket(configured):
    ws = FakeWebSocket()
    if configured:
        ws.queue_server_event({"type": "session.updated"})
    ws.queue_error()
    stt = stepfun.STT()
    stt._pool = FakePool(ws)
    stream = stt.stream(conn_options=NO_RETRY)
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="websocket error"):
            await asyncio.wait_for(anext(stream), 1)
        assert ws.closed
    finally:
        await stt.aclose()


@pytest.mark.parametrize("configured", [False, True])
async def test_tts_transport_error_closes_socket(configured):
    ws = FakeWebSocket()
    if configured:
        ws.queue_server_event(_CREATED)
    ws.queue_error()
    tts = stepfun.TTS()
    tts._pool = FakePool(_Connection(ws=ws, session_id=_SESSION))
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("hello")
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="websocket error"):
            await asyncio.wait_for(anext(stream), 1)
        assert ws.closed
    finally:
        await tts.aclose()


async def test_model_update_does_not_change_existing_tts_stream(monkeypatch):
    old_model = "stepaudio-3-tts"
    tts = stepfun.TTS(model=old_model)
    ws = FakeWebSocket()
    conn = _Connection(ws=ws, session_id=_SESSION, model=old_model)
    for event in (_CREATED, _audio_delta(), _AUDIO_DONE):
        ws.queue_server_event(event)
    stale = _Connection(ws=FakeWebSocket(), session_id="stale", model="stepaudio-2.5-tts")
    tts._pool = FakePool(stale)
    connect = AsyncMock(return_value=conn)
    monkeypatch.setattr(tts, "_connect_ws", connect)
    stream = tts.stream(conn_options=NO_RETRY)
    tts.update_options(model="stepaudio-2.5-tts")
    stream.push_text("hello")
    stream.end_input()
    try:
        frames = [frame async for frame in stream]
        assert frames
        connect.assert_awaited_once_with(NO_RETRY.timeout, model=old_model)
        assert stale.ws.closed
        assert ws.closed
    finally:
        await tts.aclose()


@pytest.mark.parametrize("component", [stepfun.STT, stepfun.TTS])
async def test_interrupted_stream_closes_socket_immediately(component):
    started = asyncio.Event()

    class ListeningSocket(FakeWebSocket):
        async def receive(self):
            if self._incoming.empty():
                started.set()
            return await super().receive()

    ws = ListeningSocket()
    plugin = component()
    if component is stepfun.STT:
        ws.queue_server_event({"type": "session.updated"})
        plugin._pool = FakePool(ws)
    else:
        ws.queue_server_event(_CREATED)
        plugin._pool = FakePool(_Connection(ws=ws, session_id=_SESSION))
    stream = plugin.stream(conn_options=NO_RETRY)
    try:
        await asyncio.wait_for(started.wait(), 1)
        await stream.aclose()
        assert ws.closed
    finally:
        await plugin.aclose()


async def test_tts_aclose_cancels_active_http_synthesis(monkeypatch):
    started, released = asyncio.Event(), asyncio.Event()

    class BlockingResponse:
        status = 200
        headers = {}

        @property
        def content(self):
            return self

        async def iter_chunks(self):
            started.set()
            await asyncio.Event().wait()
            yield b"", True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            released.set()

    session = _FakeHTTPSession(b"")
    monkeypatch.setattr(session, "post", lambda *args, **kwargs: BlockingResponse())
    tts = stepfun.TTS(http_session=session)
    stream = tts.synthesize("hello", conn_options=NO_RETRY)
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(tts.aclose(), 1)
        assert released.is_set()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert not session.closed
    finally:
        await stream.aclose()


async def test_stream_send_error_is_mapped_for_fallback():
    class BrokenSocket(FakeWebSocket):
        async def send_str(self, data):
            if "input_audio_buffer.append" in data:
                raise aiohttp.ClientConnectionError("upload failed")
            await super().send_str(data)

    ws = BrokenSocket()
    ws.queue_server_event({"type": "session.updated"})
    stt = stepfun.STT()
    stt._pool = FakePool(ws)
    stream = stt.stream(conn_options=NO_RETRY)
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError, match="upload failed"):
            await asyncio.wait_for(anext(stream), 1)
        assert ws.closed
    finally:
        await stt.aclose()
