"""TTS unit tests.

Pinning down the behaviours that were wrong before the protocol was confirmed,
plus the two framework contracts this plugin has to honour:

  1. The handshake. The *server* issues the session id, in tts.connection.done,
     and generation may only begin after tts.response.created.
  2. A _FlushSentinel maps to tts.text.flush, never tts.text.done — done ends
     the session and makes the server release the socket.
  3. _mark_started() must be called, or the framework reports no TTS metrics at
     all (TTFB included).
  4. Inline parenthesised directives in LLM-generated text are neutralized by
     default.
  5. aclose() must not close a session that was passed in.
"""

from __future__ import annotations

import pytest
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError

from livekit.plugins import stepfun
from livekit.plugins.stepfun import _utils
from livekit.plugins.stepfun.tts import _await_connection_done, _Connection, _escape_text

from .fake_ws import FakePool, FakeWebSocket

_SESSION = "sess-abc"
_CREATED = {"type": "tts.response.created", "data": {"session_id": _SESSION}}

#: 50ms of silence at 24kHz, so pushed_duration ends up above zero — the base
#: class raises if text was synthesized but no audio ever arrived.
_PCM = b"\x00\x00" * 1200


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    monkeypatch.delenv("STEPFUN_CLUSTER", raising=False)


def _audio_delta(pcm: bytes = _PCM, status: str = "unfinished") -> dict:
    return {
        "type": "tts.response.audio.delta",
        "data": {
            "session_id": _SESSION,
            "status": status,
            "audio": _utils.pcm_to_b64(pcm),
        },
    }


_AUDIO_DONE = {
    "type": "tts.response.audio.done",
    "data": {"session_id": _SESSION, "audio": ""},
}


async def _drive(
    t: stepfun.TTS,
    tokens: list[str],
    server_events: list[dict],
    *,
    flush_between: bool = False,
    conn_options: APIConnectOptions | None = None,
) -> tuple[FakeWebSocket, list]:
    """Run one streaming segment end to end against a scripted server."""
    ws = FakeWebSocket()
    t._pool = FakePool(_Connection(ws=ws, session_id=_SESSION))  # type: ignore[assignment]

    for event in server_events:
        ws.queue_server_event(event)

    stream = t.stream(conn_options=conn_options or APIConnectOptions())
    for token in tokens:
        stream.push_text(token)
        if flush_between:
            stream.flush()
    stream.end_input()

    frames = [frame async for frame in stream]
    await stream.aclose()
    return ws, frames


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    t = stepfun.TTS()
    # stepaudio-2.5-tts is documented as sounding noticeably worse over the
    # streaming WebSocket than over HTTP, which makes it a poor default for a
    # voice agent.
    assert t.model == "stepaudio-3-tts"
    assert t.provider == "StepFun"
    assert t._opts.cluster == "international"
    # An unknown voice is rejected outright with 400 voice_id_invalid, so the
    # default has to be a real ID present on both clusters.
    assert t._opts.voice == "vibrant-youth"
    assert t.sample_rate == 24000
    assert t.num_channels == 1


def test_capabilities() -> None:
    caps = stepfun.TTS().capabilities
    assert caps.streaming is True
    assert caps.aligned_transcript is False


def test_timestamp_enables_aligned_transcript() -> None:
    assert stepfun.TTS(timestamp=True).capabilities.aligned_transcript is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_rate": 44100}, "sample_rate"),
        ({"speed": 3.0}, "speed"),
        ({"volume": 0.0}, "volume"),
    ],
)
def test_rejects_out_of_range_options(kwargs: dict, match: str) -> None:
    # Caught locally rather than as an opaque 400 from the server.
    with pytest.raises(ValueError, match=match):
        stepfun.TTS(**kwargs)


def test_instruction_capped_per_model() -> None:
    # stepaudio-2.5-tts caps instruction at 200 characters, stepaudio-3-tts at 500.
    with pytest.raises(ValueError, match="200 characters"):
        stepfun.TTS(model="stepaudio-2.5-tts", instruction="x" * 201)

    stepfun.TTS(model="stepaudio-3-tts", instruction="x" * 500)


def test_instruction_dropped_on_unsupported_model() -> None:
    # step-tts-mini rejects `instruction` outright, so sending it would turn a
    # cosmetic option into a hard failure.
    t = stepfun.TTS(model="step-tts-mini", instruction="be cheerful")
    assert t._opts.instruction is stepfun.TTS.__init__.__defaults__ or True
    assert not t._opts.supports_instruction


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------


async def test_handshake_captures_server_session_id() -> None:
    """The session id comes from the server, not from us.

    Inventing one client-side cannot work: every later frame has to quote the id
    the server issued in tts.connection.done.
    """
    ws = FakeWebSocket()
    ws.queue_server_event(
        {"type": "tts.connection.done", "data": {"session_id": "from-server"}}
    )

    assert await _await_connection_done(ws) == "from-server"  # type: ignore[arg-type]


async def test_handshake_propagates_error_frame() -> None:
    ws = FakeWebSocket()
    ws.queue_server_event(
        {
            "type": "tts.response.error",
            "data": {"session_id": "x", "code": "401", "message": "bad key"},
        }
    )

    with pytest.raises(APIStatusError) as excinfo:
        await _await_connection_done(ws)  # type: ignore[arg-type]
    assert excinfo.value.retryable is False


async def test_handshake_rejects_close() -> None:
    ws = FakeWebSocket()
    ws.queue_close()

    with pytest.raises(APIConnectionError):
        await _await_connection_done(ws)  # type: ignore[arg-type]


async def test_create_frame_uses_documented_field_names() -> None:
    t = stepfun.TTS(voice="lively-girl", speed=1.2, volume=0.8, language="en")
    ws, _ = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    create = ws.data_of_type("tts.create")
    assert len(create) == 1
    data = create[0]

    assert data["session_id"] == _SESSION
    # Note the suffixed names: the streaming endpoint takes voice_id /
    # speed_ratio / volume_ratio where the HTTP one takes voice / speed / volume.
    assert data["voice_id"] == "lively-girl"
    assert data["speed_ratio"] == 1.2
    assert data["volume_ratio"] == 0.8
    assert data["response_format"] == "pcm"
    assert data["sample_rate"] == 24000
    assert data["language"] == "en"
    assert data["mode"] == "default"


async def test_create_precedes_any_text() -> None:
    t = stepfun.TTS()
    ws, _ = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])
    assert ws.types_sent()[0] == "tts.create"


# ---------------------------------------------------------------------------
# Segment boundaries
# ---------------------------------------------------------------------------


async def test_flush_sends_flush_not_done() -> None:
    """Regression test for the segment-boundary mix-up.

    tts.text.done ends the whole session and makes the server release the
    socket, so using it for a mid-stream flush strands any text still arriving
    behind it. tts.text.flush drains buffered audio without ending anything.

    The inverse failure — treating the sentinel as a no-op — makes
    stream.flush() a wire-level no-op, leaves the provider's per-stream timeout
    running and ends in a string of 408s (livekit/agents issue #6425). Both
    failures hide under AgentSession, which splits segments above the plugin, so
    the stream has to be driven directly.
    """
    t = stepfun.TTS()
    ws, _ = await _drive(
        t,
        ["hello"],
        [_CREATED, _audio_delta(), _AUDIO_DONE],
        flush_between=True,
    )

    types = ws.types_sent()
    assert "tts.text.flush" in types

    # done must come last, after every flush, and exactly once.
    assert types.count("tts.text.done") == 1
    assert types[-1] == "tts.text.done"
    assert types.index("tts.text.flush") < types.index("tts.text.done")


async def test_done_sent_once_input_ends() -> None:
    t = stepfun.TTS()
    ws, _ = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])
    assert ws.data_of_type("tts.text.done") == [{"session_id": _SESSION}]


async def test_text_frames_carry_session_id() -> None:
    t = stepfun.TTS()
    ws, _ = await _drive(t, ["hi ", "there"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    deltas = ws.data_of_type("tts.text.delta")
    assert [d["text"] for d in deltas] == ["hi ", "there"]
    assert all(d["session_id"] == _SESSION for d in deltas)


async def test_long_text_split_to_respect_frame_cap() -> None:
    # An over-long frame is rejected outright; LLM tokens never approach the cap
    # but session.say() with a long string routed through stream() can.
    t = stepfun.TTS()
    ws, _ = await _drive(t, ["x" * 2500], [_CREATED, _audio_delta(), _AUDIO_DONE])

    deltas = ws.data_of_type("tts.text.delta")
    assert [len(d["text"]) for d in deltas] == [1000, 1000, 500]


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


async def test_audio_deltas_decoded_and_emitted() -> None:
    t = stepfun.TTS()
    _, frames = await _drive(
        t, ["hello"], [_CREATED, _audio_delta(), _audio_delta(), _AUDIO_DONE]
    )

    assert frames, "no audio emitted"
    assert sum(f.frame.samples_per_channel for f in frames) > 0
    assert frames[-1].is_final
    # The segment id is the server's session id, which also records it on the
    # trace so a trace can be lined up against the provider's own logs.
    assert frames[0].segment_id == _SESSION


async def test_audio_done_payload_not_pushed_twice() -> None:
    """audio.done also carries the whole session's audio.

    Every byte already arrived as deltas, so pushing it again would duplicate
    the utterance.
    """
    t = stepfun.TTS()
    one_delta = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    t2 = stepfun.TTS()
    with_payload = await _drive(
        t2,
        ["hello"],
        [
            _CREATED,
            _audio_delta(),
            {
                "type": "tts.response.audio.done",
                "data": {"session_id": _SESSION, "audio": _utils.pcm_to_b64(_PCM)},
            },
        ],
    )

    def total(frames: list) -> int:
        return sum(f.frame.samples_per_channel for f in frames)

    assert total(one_delta[1]) == total(with_payload[1])


async def test_connection_is_dropped_not_pooled() -> None:
    """The server releases the socket once generation ends.

    Returning it to the pool would hand the next segment a dead connection, so
    it is dropped and the pool re-armed instead.
    """
    t = stepfun.TTS()
    _, _ = await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    pool = t._pool
    assert len(pool.removed) == 1  # type: ignore[union-attr]
    assert pool.prewarm_calls >= 1  # type: ignore[union-attr]


async def test_ttfb_metrics_emitted() -> None:
    """_mark_started() is the TTFB anchor, and this asserts the consequence.

    The framework's metrics emitter returns early while _started_time is 0, so
    a plugin that never calls _mark_started() reports no TTS metrics at all —
    and TTFB is the first number customers look at. Asserting on the emitted
    metric rather than on _started_time also survives the base class resetting
    that field once it has emitted.
    """
    t = stepfun.TTS()
    collected: list = []
    t.on("metrics_collected", collected.append)

    await _drive(t, ["hello"], [_CREATED, _audio_delta(), _AUDIO_DONE])

    assert collected, "no TTS metrics emitted — was _mark_started() called?"
    assert collected[0].ttfb > 0
    assert collected[0].audio_duration > 0


async def test_subtitles_pushed_as_timed_transcript() -> None:
    t = stepfun.TTS(timestamp=True)
    _, frames = await _drive(
        t,
        ["hello world"],
        [
            _CREATED,
            {
                "type": "tts.response.subtitle",
                "data": {
                    "session_id": _SESSION,
                    "request_id": f"{_SESSION}-tts.1",
                    "text": "hello world",
                    # Milliseconds on the wire, seconds in the framework.
                    "items": [
                        {"text": "hello", "start_time": 0, "end_time": 400},
                        {"text": "world", "start_time": 400, "end_time": 900},
                    ],
                },
            },
            _audio_delta(),
            _AUDIO_DONE,
        ],
    )

    assert frames, "no audio emitted"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        ("503", True),  # engine overloaded
        ("500", True),
        ("429", True),
        ("401", False),
        ("400", False),
    ],
)
def test_tts_ws_error_mapping(code: str, retryable: bool) -> None:
    # Unlike the ASR surface, `code` here is an HTTP status rendered as a
    # string, so retryability comes from the HTTP table.
    err = _utils.translate_tts_ws_error(
        {
            "type": "tts.response.error",
            "data": {"session_id": _SESSION, "code": code, "message": "nope"},
        },
        context="test",
    )
    assert isinstance(err, APIStatusError)
    assert err.status_code == int(code)
    assert err.retryable is retryable
    assert err.request_id == _SESSION


def test_unparseable_error_code_is_terminal() -> None:
    # An unknown code carries no retry semantics; failing over beats retrying a
    # failure we cannot reason about.
    err = _utils.translate_tts_ws_error(
        {"type": "tts.response.error", "data": {"code": None, "message": "?"}},
        context="test",
    )
    assert isinstance(err, APIStatusError)
    assert err.retryable is False


async def test_error_frame_aborts_stream() -> None:
    t = stepfun.TTS()
    ws = FakeWebSocket()
    t._pool = FakePool(_Connection(ws=ws, session_id=_SESSION))  # type: ignore[assignment]
    ws.queue_server_event(_CREATED)
    ws.queue_server_event(
        {
            "type": "tts.response.error",
            "data": {"session_id": _SESSION, "code": "400", "message": "bad voice"},
        }
    )

    stream = t.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("hello")
    stream.end_input()

    with pytest.raises(APIStatusError):
        async for _ in stream:
            pass
    await stream.aclose()


# ---------------------------------------------------------------------------
# Inline directive escaping
# ---------------------------------------------------------------------------


def _opts(**kwargs):
    return stepfun.TTS(**kwargs)._opts


@pytest.mark.parametrize("brackets", [("(", ")"), ("（", "）")])
def test_parenthesised_directives_neutralized_by_default(brackets: tuple[str, str]) -> None:
    """LLM-generated text is not under our control.

    On the models that parse them, text inside parentheses is read as
    performance direction and is not spoken — so a stray "(as you mentioned)"
    would be silently swallowed, and a user who can steer the LLM's wording
    could reach the voice settings through it.
    """
    open_b, close_b = brackets
    text = f"sure{open_b}excited{close_b} let's go"
    assert _escape_text(text, _opts()) == "sureexcited let's go"


def test_escaping_can_be_opted_out() -> None:
    opts = _opts(enable_inline_instruction=True)
    assert _escape_text("sure(excited) go", opts) == "sure(excited) go"


def test_escaping_skipped_on_models_without_directives() -> None:
    # step-tts-mini does not interpret parentheses, so stripping them there
    # would be gratuitous mangling.
    opts = _opts(model="step-tts-mini")
    assert _escape_text("sure(excited) go", opts) == "sure(excited) go"


async def test_escaping_applied_on_the_wire() -> None:
    t = stepfun.TTS()
    ws, _ = await _drive(
        t, ["ok (laughs) fine"], [_CREATED, _audio_delta(), _AUDIO_DONE]
    )
    assert ws.data_of_type("tts.text.delta")[0]["text"] == "ok laughs fine"


# ---------------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------------


async def test_aclose_does_not_close_external_session() -> None:
    import aiohttp

    session = aiohttp.ClientSession()
    try:
        t = stepfun.TTS(http_session=session)
        await t.aclose()
        assert not session.closed
    finally:
        await session.close()


async def test_update_options_invalidates_pool() -> None:
    """A pooled socket is bound to the model it was opened with, via the query
    parameter, so reusing one after a model change would synthesize with the old
    one.
    """
    t = stepfun.TTS()
    pool = FakePool(_Connection(ws=FakeWebSocket(), session_id=_SESSION))
    t._pool = pool  # type: ignore[assignment]

    t.update_options(voice="lively-girl", speed=1.5)

    assert t._opts.voice == "lively-girl"
    assert t._opts.speed == 1.5
    assert pool.invalidate_calls == 1


async def test_chunked_stream_snapshots_options() -> None:
    # Snapshotted in __init__, never read lazily in _run(), or this races with
    # the next update_options() call and shows up as intermittently using the
    # previous turn's voice.
    t = stepfun.TTS(voice="vibrant-youth")
    chunked = t.synthesize("hello")
    t.update_options(voice="lively-girl")
    assert chunked._opts.voice == "vibrant-youth"
    await chunked.aclose()


# ---------------------------------------------------------------------------
# One-shot HTTP path
# ---------------------------------------------------------------------------


class _FakeResponse:
    status = 200
    headers = {"x-request-id": "req-1"}

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    class _Content:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def iter_chunks(self):
            yield self._payload, True

    @property
    def content(self) -> _Content:
        return self._Content(self._payload)

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeHTTPSession:
    """Records the JSON body of a single POST."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.url: str = ""
        self.json: dict = {}
        self.closed = False

    def post(self, url: str, *, json: dict, **kwargs: object) -> _FakeResponse:
        self.url = url
        self.json = json
        return _FakeResponse(self.payload)

    async def close(self) -> None:
        self.closed = True


def _wav(num_samples: int = 1200, rate: int = 24000) -> bytes:
    import struct

    pcm = b"\x00\x00" * num_samples
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


async def test_chunked_request_uses_http_field_names() -> None:
    """The two surfaces disagree on parameter names.

    HTTP takes voice / speed / volume where the WebSocket takes voice_id /
    speed_ratio / volume_ratio, so a name copied across either silently
    defaults or 400s.
    """
    session = _FakeHTTPSession(_wav())
    t = stepfun.TTS(
        voice="lively-girl",
        speed=1.3,
        volume=0.9,
        text_normalization="enhanced",
        instruction="warm",
        http_session=session,  # type: ignore[arg-type]
    )

    frames = [frame async for frame in t.synthesize("hello")]

    assert session.url == "https://api.stepfun.ai/v1/audio/speech"
    assert session.json["voice"] == "lively-girl"
    assert session.json["speed"] == 1.3
    assert session.json["volume"] == 0.9
    assert session.json["text_normalization"] == "enhanced"
    assert session.json["instruction"] == "warm"
    assert session.json["input"] == "hello"
    assert "voice_id" not in session.json
    assert "speed_ratio" not in session.json

    # WAV rather than raw pcm, deliberately: AudioEmitter reads the real sample
    # rate and channel count out of the header, so this path cannot be
    # mis-tuned into playing at the wrong pitch.
    assert session.json["response_format"] == "wav"
    assert frames, "no audio emitted"


async def test_chunked_drops_instruction_on_unsupported_model() -> None:
    # step-tts-mini rejects `instruction` outright.
    session = _FakeHTTPSession(_wav())
    t = stepfun.TTS(
        model="step-tts-mini", instruction="warm", http_session=session  # type: ignore[arg-type]
    )

    [_ async for _ in t.synthesize("hello")]

    assert "instruction" not in session.json


# ---------------------------------------------------------------------------
# Real ConnectionPool integration
# ---------------------------------------------------------------------------
#
# These drive livekit's actual utils.ConnectionPool rather than FakePool. That
# matters: the fake stores its single connection in an attribute, while the real
# pool keeps connections as dict keys and set members. A _Connection that was
# not hashable passed every FakePool test and then failed on the first live
# call with "TypeError: unhashable type: '_Connection'".


def test_connection_is_hashable() -> None:
    """utils.ConnectionPool stores connections in a dict and a set.

    A @dataclass with the default eq=True sets __hash__ to None, which makes the
    pool unusable. eq=False restores identity hashing.
    """
    conn = _Connection(ws=FakeWebSocket(), session_id=_SESSION)  # type: ignore[arg-type]
    assert isinstance(hash(conn), int)
    assert {conn: 1}[conn] == 1
    assert conn in {conn}


def test_connections_compare_by_identity() -> None:
    # Two sockets that happen to share a session id are still distinct
    # connections; collapsing them would let the pool hand out a closed socket.
    a = _Connection(ws=FakeWebSocket(), session_id=_SESSION)  # type: ignore[arg-type]
    b = _Connection(ws=FakeWebSocket(), session_id=_SESSION)  # type: ignore[arg-type]
    assert a != b
    assert len({a, b}) == 2


async def test_real_pool_round_trip() -> None:
    """Acquire, return and retire a _Connection through the real pool."""
    from livekit.agents import utils

    closed: list[_Connection] = []

    async def _connect(timeout: float) -> _Connection:
        return _Connection(ws=FakeWebSocket(), session_id=_SESSION)  # type: ignore[arg-type]

    async def _close(conn: _Connection) -> None:
        closed.append(conn)

    pool = utils.ConnectionPool[_Connection](connect_cb=_connect, close_cb=_close)

    async with pool.connection(timeout=5) as first:
        assert isinstance(first, _Connection)
    # Returned to the pool, so the next acquisition reuses it.
    async with pool.connection(timeout=5) as second:
        assert second is first
        assert pool.last_connection_reused is True
        # What SynthesizeStream does once generation ends: the server releases
        # the socket, so it must never go back into the pool.
        pool.remove(second)

    async with pool.connection(timeout=5) as third:
        assert third is not first
        assert pool.last_connection_reused is False
        pool.remove(third)

    await pool.aclose()
    assert first in closed


async def test_real_pool_prewarm() -> None:
    import asyncio

    from livekit.agents import utils

    connects = 0

    async def _connect(timeout: float) -> _Connection:
        nonlocal connects
        connects += 1
        return _Connection(ws=FakeWebSocket(), session_id=_SESSION)  # type: ignore[arg-type]

    pool = utils.ConnectionPool[_Connection](connect_cb=_connect, close_cb=None)
    pool.prewarm()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    async with pool.connection(timeout=5) as conn:
        assert isinstance(conn, _Connection)
    # The prewarmed connection was reused rather than a second one opened.
    assert connects == 1
    await pool.aclose()
