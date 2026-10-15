"""STT unit tests.

The point is not the coverage number but pinning down the handful of behaviours
that are easy to regress, and that were in fact wrong before the protocol was
confirmed against the provider's documentation:

  1. session.update must be sent and acknowledged before any audio flows — the
     endpoint takes no query parameters, so this frame is the only way the model
     and audio format are ever configured.
  2. Transcription deltas on stepaudio-2.5-asr-stream are *cumulative*. Appending
     them compounds every partial into garbage.
  3. input_audio_buffer.commit is only valid when server VAD is off.
  4. Error code to exception type and retryability mapping, including the
     429-rate-limit vs 429-quota-ceiling split.
  5. aclose() must not close a session that was passed in.

These tests require no credentials and no network, which matters: CI matrix jobs
carrying provider secrets generally do not run on pull requests from forks, so
this suite is the only evidence of correctness a reviewer will see on our PR.
"""

from __future__ import annotations

import pytest
from livekit.agents import APIStatusError, stt

from livekit import rtc
from livekit.plugins import stepfun
from livekit.plugins.stepfun import _utils

from .fake_ws import FakePool, FakeWebSocket

_SAMPLE_RATE = 16000
_SESSION_READY = {"type": "session.updated", "meta": {"session_id": "sess-1"}}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    monkeypatch.delenv("STEPFUN_CLUSTER", raising=False)


def _frame(ms: int = 100) -> rtc.AudioFrame:
    samples = _SAMPLE_RATE * ms // 1000
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _drive(
    s: stepfun.STT,
    server_events: list[dict],
    *,
    frames: int = 1,
) -> tuple[FakeWebSocket, list[stt.SpeechEvent]]:
    """Run one stream end to end against a scripted server."""
    ws = FakeWebSocket()
    s._pool = FakePool(ws)  # type: ignore[assignment]

    for event in server_events:
        ws.queue_server_event(event)
    ws.queue_close()

    stream = s.stream()
    for _ in range(frames):
        stream.push_frame(_frame())
    stream.end_input()

    events = [ev async for ev in stream]
    await stream.aclose()
    return ws, events


def _texts(events: list[stt.SpeechEvent], kind: stt.SpeechEventType) -> list[str]:
    return [ev.alternatives[0].text for ev in events if ev.type == kind]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_reads_api_key_from_env() -> None:
    assert stepfun.STT()._api_key == "test-key"


def test_explicit_api_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "from-env")
    assert stepfun.STT(api_key="explicit")._api_key == "explicit"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        stepfun.STT()


def test_defaults_to_international_cluster() -> None:
    # LiveKit Cloud workers run mostly in North America and Europe, so
    # defaulting to the domestic cluster would give overseas developers a poor
    # first impression. This default is deliberate — do not flip it casually.
    s = stepfun.STT()
    assert s._opts.cluster == "international"
    assert s._opts.ws_base_url == "wss://api.stepfun.ai"
    assert s._opts.http_base_url == "https://api.stepfun.ai"


def test_env_cluster_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_CLUSTER", "domestic")
    s = stepfun.STT()
    assert s._opts.cluster == "domestic"
    assert s._opts.ws_base_url == "wss://api.stepfun.com"


def test_invalid_env_cluster_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_CLUSTER", "eu-west")
    with pytest.raises(ValueError, match="STEPFUN_CLUSTER"):
        stepfun.STT()


def test_model_and_provider_reported() -> None:
    # Without these the framework labels every metric "unknown".
    s = stepfun.STT()
    assert s.model == "stepaudio-2.5-asr-stream"
    assert s.provider == "StepFun"


def test_capabilities_declared_honestly() -> None:
    caps = stepfun.STT().capabilities
    assert caps.streaming is True
    assert caps.interim_results is True
    assert caps.offline_recognize is True
    assert caps.diarization is False
    # Not claimed unless timestamp alignment was actually requested: declaring
    # it without emitting word timings makes adaptive interruption fail
    # silently, which is very hard to debug.
    assert caps.aligned_transcript is False
    # The endpoint takes free-text `prompt` hints, not a keyterm list.
    assert caps.keyterms is False


def test_timestamp_align_enables_word_alignment() -> None:
    caps = stepfun.STT(enable_timestamp_align=True).capabilities
    assert caps.aligned_transcript == "word"


def test_timestamp_align_ignored_on_unsupported_model() -> None:
    # Only stepaudio-2.5-asr-stream implements it; claiming the capability on a
    # model that cannot deliver it would be over-declaring.
    caps = stepfun.STT(
        model="step-asr-1.1-stream", enable_timestamp_align=True
    ).capabilities
    assert caps.aligned_transcript is False


# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------


async def test_sends_session_update_before_audio() -> None:
    """The endpoint accepts no query parameters.

    Model, audio format and turn detection are configured by this frame alone,
    so skipping it leaves the session unconfigured and nothing is ever
    recognized.
    """
    s = stepfun.STT(language="zh", prompt="livekit, webrtc")
    ws, _ = await _drive(s, [_SESSION_READY])

    assert ws.types_sent()[0] == "session.update"

    update = ws.frames_of_type("session.update")[0]
    audio_input = update["session"]["audio"]["input"]

    assert audio_input["format"] == {
        "type": "pcm",
        "codec": "pcm_s16le",
        "rate": _SAMPLE_RATE,
        "bits": 16,
        "channel": 1,
    }
    assert audio_input["transcription"]["model"] == "stepaudio-2.5-asr-stream"
    assert audio_input["transcription"]["language"] == "zh"
    assert audio_input["transcription"]["prompt"] == "livekit, webrtc"
    assert audio_input["turn_detection"]["type"] == "server_vad"

    # Every client frame carries its own event_id.
    assert all("event_id" in frame for frame in ws.sent)


async def test_turn_detection_omitted_when_server_vad_disabled() -> None:
    s = stepfun.STT(server_vad=False)
    ws, _ = await _drive(s, [_SESSION_READY])

    audio_input = ws.frames_of_type("session.update")[0]["session"]["audio"]["input"]
    assert "turn_detection" not in audio_input


async def test_timestamp_align_requested_in_session_update() -> None:
    s = stepfun.STT(enable_timestamp_align=True)
    ws, _ = await _drive(s, [_SESSION_READY])

    transcription = ws.frames_of_type("session.update")[0]["session"]["audio"]["input"][
        "transcription"
    ]
    assert transcription["enable_timestamp_align"] is True


async def test_audio_is_sent_as_base64_pcm() -> None:
    s = stepfun.STT()
    ws, _ = await _drive(s, [_SESSION_READY], frames=1)

    appends = ws.frames_of_type("input_audio_buffer.append")
    assert appends, "no audio was sent"
    # 100ms of input at a 50ms frame size.
    assert len(appends) == 2
    assert _utils.b64_to_pcm(appends[0]["audio"]) == b"\x00\x00" * (_SAMPLE_RATE // 20)


# ---------------------------------------------------------------------------
# Commit semantics
# ---------------------------------------------------------------------------


async def test_no_commit_under_server_vad() -> None:
    """With server VAD the server owns segmentation, so no commit is sent.

    Committing here would compete with the server's own boundaries. Finals
    arrive when it hears `silence_duration_ms` of silence instead — verified
    live at 1372ms after the last word, which a real microphone supplies
    because it keeps streaming after the speaker stops.
    """
    s = stepfun.STT(server_vad=True)
    ws, _ = await _drive(s, [_SESSION_READY])
    assert ws.frames_of_type("input_audio_buffer.commit") == []


async def test_commit_sent_when_server_vad_disabled() -> None:
    """With server VAD off, commit is the documented path and it works.

    Measured against the live service: a final arrived 438ms after input ended.
    If a flush stops producing a commit, the final never arrives and a
    standalone caller goes mute — AgentSession does not exercise this path, so
    the stream has to be driven directly.
    """
    s = stepfun.STT(server_vad=False)
    ws, _ = await _drive(s, [_SESSION_READY])
    assert ws.frames_of_type("input_audio_buffer.commit")
    # The commit must be the last frame: anything after it would be audio the
    # server has already been told not to expect.
    assert ws.types_sent()[-1] == "input_audio_buffer.commit"


async def test_flush_drains_partial_frame() -> None:
    """The sub-frame tail inside AudioByteStream must be drained on flush.

    A 30ms push does not fill a 50ms frame, so without the explicit drain that
    audio is silently dropped.
    """
    s = stepfun.STT(server_vad=False)
    ws = FakeWebSocket()
    s._pool = FakePool(ws)  # type: ignore[assignment]
    ws.queue_server_event(_SESSION_READY)
    ws.queue_close()

    stream = s.stream()
    stream.push_frame(_frame(ms=30))
    stream.end_input()
    [_ async for _ in stream]
    await stream.aclose()

    assert ws.frames_of_type("input_audio_buffer.append"), "tail frame was lost"


# ---------------------------------------------------------------------------
# Transcript translation
# ---------------------------------------------------------------------------

_DELTA = "conversation.item.input_audio_transcription.delta"
_COMPLETED = "conversation.item.input_audio_transcription.completed"


async def test_cumulative_deltas_replace_rather_than_append() -> None:
    """Regression test for the most damaging protocol mistake available here.

    stepaudio-2.5-asr-stream sends the whole transcript so far on every delta,
    corrections included. Appending them turns "你好" / "你好世界" into
    "你好你好世界" and gets worse with every partial.
    """
    s = stepfun.STT(language="zh")
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "你好"},
            {"type": _DELTA, "text": "你好世界"},
        ],
    )

    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["你好", "你好世界"]


async def test_incremental_deltas_are_appended_on_legacy_model() -> None:
    # step-asr-1.1-stream sends fragments instead, so the two models genuinely
    # need different handling.
    s = stepfun.STT(model="step-asr-1.1-stream")
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "hello "},
            {"type": _DELTA, "text": "world"},
        ],
    )

    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == [
        "hello ",
        "hello world",
    ]


async def test_final_transcript_read_from_transcript_field() -> None:
    # The completed frame carries the final text under `transcript`, not `text`.
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "partial"},
            {"type": _COMPLETED, "transcript": "the final text"},
        ],
    )

    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["the final text"]


async def test_stash_produces_preflight_transcript() -> None:
    """`stash` marks the trailing characters the server may still rewrite.

    Everything before it is settled, which is what PREFLIGHT_TRANSCRIPT is for:
    stable enough for preemptive generation while the rest is still in flux.
    """
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [_SESSION_READY, {"type": _DELTA, "text": "hello world", "stash": "world"}],
    )

    assert _texts(events, stt.SpeechEventType.PREFLIGHT_TRANSCRIPT) == ["hello "]
    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["hello world"]


async def test_word_timestamps_mapped_when_enabled() -> None:
    s = stepfun.STT(enable_timestamp_align=True)
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {
                "type": _COMPLETED,
                "transcript": "hi there",
                "words": [
                    {"word": "hi", "start": 0.0, "end": 0.4, "final": True},
                    {"word": "there", "start": 0.4, "end": 0.9, "final": True},
                ],
            },
        ],
    )

    finals = [ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    words = finals[0].alternatives[0].words
    assert words is not None
    assert [str(w) for w in words] == ["hi", "there"]
    assert words[0].start_time == 0.0
    assert words[1].end_time == 0.9


async def test_non_final_words_are_skipped() -> None:
    # A word with final=false sits in the stash region and its timing is only an
    # estimate, so publishing it as settled would be a lie.
    s = stepfun.STT(enable_timestamp_align=True)
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {
                "type": _DELTA,
                "text": "hi there",
                "words": [
                    {"word": "hi", "start": 0.0, "end": 0.4, "final": True},
                    {"word": "there", "start": 0.4, "end": 0.9, "final": False},
                ],
            },
        ],
    )

    interim = [ev for ev in events if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT]
    words = interim[0].alternatives[0].words
    assert words is not None
    assert [str(w) for w in words] == ["hi"]


async def test_words_omitted_when_alignment_disabled() -> None:
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {
                "type": _COMPLETED,
                "transcript": "hi",
                "words": [{"word": "hi", "start": 0.0, "end": 0.4, "final": True}],
            },
        ],
    )

    finals = [ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals[0].alternatives[0].words is None


# ---------------------------------------------------------------------------
# VAD events
# ---------------------------------------------------------------------------


async def test_server_vad_events_are_translated() -> None:
    """These are genuine server events, so translating them is not fabrication.

    The design boundary that matters is the opposite one: nothing here may be
    inferred from client-side energy detection, which would fight silero.VAD and
    the turn detector in a mixed setup.
    """
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0},
            {"type": "input_audio_buffer.speech_stopped", "audio_start_ms": 900},
        ],
    )

    types = [ev.type for ev in events]
    assert stt.SpeechEventType.START_OF_SPEECH in types
    assert stt.SpeechEventType.END_OF_SPEECH in types


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


async def test_usage_reported_with_server_token_counts() -> None:
    """RECOGNITION_USAGE is the only trigger for streamed STTMetrics.

    The base class's metrics monitor listens for that event alone, so not
    emitting it means no STT metrics for the entire session.
    """
    s = stepfun.STT()
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {
                "type": _COMPLETED,
                "transcript": "hi",
                "usage": {"prompt_tokens": 120, "completion_tokens": 7},
            },
        ],
    )

    usage = [ev for ev in events if ev.type == stt.SpeechEventType.RECOGNITION_USAGE]
    assert usage, "no usage event emitted"
    assert usage[0].recognition_usage is not None
    assert usage[0].recognition_usage.input_tokens == 120
    assert usage[0].recognition_usage.output_tokens == 7
    assert usage[0].recognition_usage.audio_duration > 0


async def test_connection_is_dropped_not_pooled() -> None:
    """The socket carries per-utterance state with no documented reset.

    Returning it to the pool would hand the next stream a half-used session, so
    it is dropped and the pool re-armed instead.
    """
    s = stepfun.STT()
    ws, _ = await _drive(s, [_SESSION_READY])

    pool = s._pool
    assert pool.removed == [ws]  # type: ignore[union-attr]
    assert pool.prewarm_calls >= 1  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),  # bad parameters: a retry fails identically
        (401, False),  # marking auth failures retryable makes fallback chains hammer
        (402, False),  # no balance
        (404, False),
        (451, False),  # content moderation
        (429, True),  # plain rate limiting
        (500, True),
        (503, True),
    ],
)
def test_status_error_retryable_mapping(status: int, retryable: bool) -> None:
    err = _utils.status_error(status_code=status, message="x")
    assert isinstance(err, APIStatusError)
    assert err.retryable is retryable


@pytest.mark.parametrize(
    "identifier",
    [
        "insufficient_credit",
        "project_credit_limit_exceeded",
        "member_project_credit_limit_exceeded",
    ],
)
def test_quota_429_is_not_retryable(identifier: str) -> None:
    """429 is overloaded on this API.

    Ordinary rate limiting clears on retry; a Credit ceiling does not clear
    until the limit is raised or the quota resets on the 1st of next month.
    Treating the latter as retryable pins a FallbackAdapter chain to a provider
    that is guaranteed to fail for the rest of the month.
    """
    err = _utils.status_error(
        status_code=429,
        message="quota",
        body=f'{{"error": {{"code": "{identifier}"}}}}',
    )
    assert err.retryable is False


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        ("internal_error", True),
        ("max_idle_timeout", True),  # stale connection; reconnecting fixes it
        ("pong_timeout", True),
        ("invalid_value", False),
        ("missing_param", False),
        ("risk_blocked", False),  # content safety: a retry fails identically
    ],
)
def test_asr_ws_error_retryable_mapping(code: str, retryable: bool) -> None:
    err = _utils.translate_asr_ws_error(
        {
            "type": "error",
            "meta": {"session_id": "sess-1"},
            "error": {"type": "invalid_request_error", "code": code, "message": "nope"},
        },
        context="test",
    )
    assert isinstance(err, APIStatusError)
    assert err.retryable is retryable
    assert err.request_id == "sess-1"


async def test_error_frame_raises() -> None:
    s = stepfun.STT()
    ws = FakeWebSocket()
    s._pool = FakePool(ws)  # type: ignore[assignment]
    ws.queue_server_event(_SESSION_READY)
    ws.queue_server_event(
        {
            "type": "error",
            "meta": {"session_id": "sess-1"},
            "error": {"code": "invalid_value", "message": "bad rate"},
        }
    )

    stream = s.stream(conn_options=_no_retry())
    stream.push_frame(_frame())
    stream.end_input()

    with pytest.raises(APIStatusError):
        async for _ in stream:
            pass
    await stream.aclose()


def _no_retry():
    from livekit.agents import APIConnectOptions

    # A non-retryable error must surface immediately; retries would only slow
    # the test down and obscure which error arrived.
    return APIConnectOptions(max_retry=0)


# ---------------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------------


async def test_aclose_does_not_close_external_session() -> None:
    import aiohttp

    session = aiohttp.ClientSession()
    try:
        s = stepfun.STT(http_session=session)
        await s.aclose()
        # Closing a caller-supplied session is a common way for plugins to
        # break their callers.
        assert not session.closed
    finally:
        await session.close()


async def test_update_options_invalidates_pool() -> None:
    """Pooled sockets were configured by a session.update built from the old
    options, so they must stop being handed out.
    """
    s = stepfun.STT()
    pool = FakePool(FakeWebSocket())
    s._pool = pool  # type: ignore[assignment]

    s.update_options(language="ja")

    assert s._opts.language == "ja"
    assert pool.invalidate_calls == 1


async def test_stream_snapshots_options() -> None:
    """Options are snapshotted in the stream constructor, never read lazily in
    _run().

    Otherwise this races with the next update_options() call and shows up as
    intermittently using the previous turn's language — hard enough to reproduce
    that it needs a test.
    """
    s = stepfun.STT(language="zh")
    ws = FakeWebSocket()
    s._pool = FakePool(ws)  # type: ignore[assignment]

    stream = s.stream()
    s.update_options(language="ja")

    assert stream._opts.language == "zh"
    await stream.aclose()


# ---------------------------------------------------------------------------
# Batch recognition (HTTP + SSE)
# ---------------------------------------------------------------------------
#
# offline_recognize=True is declared, so FallbackAdapter and StreamAdapter will
# actually call this path. A declared capability with no coverage is the
# over-declaring trap these tests exist to prevent.


class _FakeSSEResponse:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    @property
    def content(self):
        async def _gen():
            for line in self._lines:
                yield (line + "\n").encode()

        return _gen()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeSSESession:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.url = ""
        self.json: dict = {}
        self.headers: dict[str, str] = {}

    def post(self, url: str, *, json: dict, headers: dict, **kwargs: object):
        self.url = url
        self.json = json
        self.headers = headers
        return _FakeSSEResponse(self.lines)

    async def close(self) -> None:
        pass


async def test_recognize_posts_nested_pcm_body() -> None:
    session = _FakeSSESession(
        [
            'data: {"type":"transcript.text.delta","delta":"hel"}',
            'data: {"type":"transcript.text.delta","delta":"lo"}',
            'data: {"type":"transcript.text.done","text":"hello",'
            '"meta":{"session_id":"sse-1"},'
            '"usage":{"input_tokens":90,"output_tokens":4}}',
            "data: [DONE]",
        ]
    )
    s = stepfun.STT(language="zh", http_session=session)  # type: ignore[arg-type]

    event = await s.recognize([_frame(ms=200)])

    assert s._opts.http_base_url == "https://api.stepfun.ai"
    assert session.url == "https://api.stepfun.ai/v1/audio/asr/sse"
    # SSE requires this Accept header; without it the server answers with plain
    # JSON and the parse yields nothing.
    assert session.headers["Accept"] == "text/event-stream"

    audio = session.json["audio"]
    transcription = audio["input"]["transcription"]
    assert transcription["model"] == "stepaudio-3-asr-max"
    assert transcription["language"] == "zh"
    # rate/bits/channel are mandatory for raw PCM, and come from the buffer
    # rather than from opts: this path gets whatever the caller recorded, with
    # no resampling in between.
    assert audio["input"]["format"] == {
        "type": "pcm",
        "codec": "pcm_s16le",
        "rate": _SAMPLE_RATE,
        "bits": 16,
        "channel": 1,
    }
    assert _utils.b64_to_pcm(audio["data"])

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    # The full transcript lives in `text` on the done event; deltas are
    # incremental on this route, unlike the WebSocket's cumulative ones.
    assert event.alternatives[0].text == "hello"
    assert event.request_id == "sse-1"
    assert event.recognition_usage is not None
    assert event.recognition_usage.input_tokens == 90
    assert event.recognition_usage.audio_duration == pytest.approx(0.2)


async def test_recognize_falls_back_to_accumulated_deltas() -> None:
    # A done event that omits `text` should not lose the transcript.
    session = _FakeSSESession(
        [
            'data: {"type":"transcript.text.delta","delta":"partial"}',
            'data: {"type":"transcript.text.done","meta":{"session_id":"sse-2"}}',
        ]
    )
    s = stepfun.STT(http_session=session)  # type: ignore[arg-type]

    event = await s.recognize([_frame()])
    assert event.alternatives[0].text == "partial"


async def test_recognize_raises_on_error_event() -> None:
    session = _FakeSSESession(
        ['data: {"type":"error","message":"bad audio","meta":{"session_id":"sse-3"}}']
    )
    s = stepfun.STT(http_session=session)  # type: ignore[arg-type]

    with pytest.raises(APIStatusError):
        await s.recognize([_frame()], conn_options=_no_retry())


async def test_recognize_omits_hotwords_on_international_cluster() -> None:
    # hotwords is documented for the domestic cluster only; sending it
    # internationally risks a 400 on an unknown parameter.
    session = _FakeSSESession(['data: {"type":"transcript.text.done","text":"x"}'])
    s = stepfun.STT(hotwords=["livekit"], http_session=session)  # type: ignore[arg-type]

    await s.recognize([_frame()])
    assert "hotwords" not in session.json["audio"]["input"]["transcription"]


async def test_recognize_sends_hotwords_on_domestic_cluster() -> None:
    session = _FakeSSESession(['data: {"type":"transcript.text.done","text":"x"}'])
    s = stepfun.STT(
        cluster="domestic", hotwords=["livekit"], http_session=session  # type: ignore[arg-type]
    )

    await s.recognize([_frame()])
    assert session.json["audio"]["input"]["transcription"]["hotwords"] == ["livekit"]


# ---------------------------------------------------------------------------
# HTTP-only mode (use_streaming=False)
# ---------------------------------------------------------------------------
#
# The two endpoints accept disjoint model sets: stepaudio-3-asr-max is
# documented for the batch route only, and the streaming route accepts only
# stepaudio-2.5-asr-stream. So running the batch model in a live conversation
# means declaring streaming=False and letting the framework wrap this STT in a
# stt.StreamAdapter, which buffers each utterance until the local VAD reports
# end of speech.


def test_http_only_declares_no_streaming() -> None:
    caps = stepfun.STT(use_streaming=False).capabilities
    assert caps.streaming is False
    # StreamAdapter surfaces only the final transcript, so claiming interim
    # results would be a lie — and the framework would believe it.
    assert caps.interim_results is False
    # recognize() still works; that is the whole point of this mode.
    assert caps.offline_recognize is True


def test_http_only_reports_the_batch_model() -> None:
    """StreamAdapter delegates `model` to the wrapped STT.

    Reporting the streaming model while the batch route does the work would
    mislabel every metric for the session.
    """
    s = stepfun.STT(use_streaming=False)
    assert s.model == "stepaudio-3-asr-max"

    s2 = stepfun.STT(use_streaming=False, batch_model="stepaudio-2.5-asr")
    assert s2.model == "stepaudio-2.5-asr"


def test_streaming_mode_reports_the_streaming_model() -> None:
    assert stepfun.STT().model == "stepaudio-2.5-asr-stream"


def test_http_only_refuses_to_stream() -> None:
    # Serving a stream here would send audio to the streaming endpoint with a
    # model it does not accept, and would contradict the declared capability.
    s = stepfun.STT(use_streaming=False)
    with pytest.raises(RuntimeError, match="use_streaming=False"):
        s.stream()


def test_http_only_ignores_timestamp_align() -> None:
    # Word alignment is a streaming-endpoint feature.
    caps = stepfun.STT(use_streaming=False, enable_timestamp_align=True).capabilities
    assert caps.aligned_transcript is False


async def test_http_only_still_recognizes() -> None:
    session = _FakeSSESession(
        ['data: {"type":"transcript.text.done","text":"hello","meta":{"session_id":"s"}}']
    )
    s = stepfun.STT(use_streaming=False, http_session=session)  # type: ignore[arg-type]

    event = await s.recognize([_frame()])

    assert event.alternatives[0].text == "hello"
    assert session.json["audio"]["input"]["transcription"]["model"] == "stepaudio-3-asr-max"


async def test_stream_adapter_wraps_http_only_stt() -> None:
    """The framework's own adapter must accept this STT.

    This is the configuration an AgentSession builds automatically when given a
    non-streaming STT plus a vad, so it is worth pinning that the wrapping
    works and that the adapter reports our model rather than its own.
    """
    from livekit.agents import stt as stt_module

    class _FakeVAD:
        pass

    s = stepfun.STT(use_streaming=False)
    adapter = stt_module.StreamAdapter(stt=s, vad=_FakeVAD())  # type: ignore[arg-type]

    assert adapter.capabilities.streaming is True
    assert adapter.capabilities.interim_results is False
    assert adapter.model == "stepaudio-3-asr-max"
    assert adapter.provider == "StepFun"
    assert adapter.wrapped_stt is s


async def test_stream_terminates_when_server_keeps_socket_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the stream must finish after input ends.

    _recv_task only returns when the socket closes, but with server VAD the
    server keeps the session open for further utterances rather than closing
    after a final. Waiting on both tasks therefore hung forever once input
    ended — which is exactly what a live run did, after happily emitting
    interims. The receive side now gets a bounded grace period instead.

    The fake server here never closes the socket, mirroring the real one.
    """
    import asyncio

    from livekit.plugins.stepfun import stt as stt_impl

    # Shortened so the suite does not sit out the real grace period; without the
    # fix this hangs regardless of its value.
    monkeypatch.setattr(stt_impl, "_FINALIZE_TIMEOUT", 0.2)

    s = stepfun.STT()
    ws = FakeWebSocket()
    s._pool = FakePool(ws)  # type: ignore[assignment]
    ws.queue_server_event(_SESSION_READY)
    ws.queue_server_event({"type": _DELTA, "text": "hello"})
    # No queue_close(): the socket stays open, as it does in production.

    stream = s.stream()
    stream.push_frame(_frame())
    stream.end_input()

    events = []
    async with asyncio.timeout(10):
        async for ev in stream:
            events.append(ev)
    await stream.aclose()

    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == ["hello"]


async def test_stream_still_surfaces_recv_errors() -> None:
    """The grace period must not swallow a genuine failure.

    asyncio.wait() does not raise, unlike gather(), so the task results have to
    be checked explicitly afterwards or every receive-side error would be
    silently discarded.
    """
    s = stepfun.STT()
    ws = FakeWebSocket()
    s._pool = FakePool(ws)  # type: ignore[assignment]
    ws.queue_server_event(_SESSION_READY)
    ws.queue_server_event(
        {
            "type": "error",
            "meta": {"session_id": "sess-1"},
            "error": {"code": "invalid_value", "message": "bad rate"},
        }
    )

    stream = s.stream(conn_options=_no_retry())
    stream.push_frame(_frame())
    stream.end_input()

    with pytest.raises(APIStatusError):
        async for _ in stream:
            pass
    await stream.aclose()


# ---------------------------------------------------------------------------
# New / unknown streaming models
# ---------------------------------------------------------------------------


def test_unknown_streaming_model_treated_as_cumulative() -> None:
    """The dangerous default must be the safe one.

    Delta semantics are keyed off a denylist of known-incremental models, so a
    model this plugin has never heard of is treated as cumulative. Guessing
    wrong the other way appends the full transcript to itself on every delta
    and compounds into garbage; guessing wrong this way only shows the latest
    fragment, and the final still arrives intact on the `completed` event.
    """
    from livekit.plugins.stepfun.stt import _STTOptions

    def _cumulative(model: str) -> bool:
        s = stepfun.STT(model=model)
        opts: _STTOptions = s._opts
        return opts.deltas_are_cumulative

    assert _cumulative("stepaudio-9-asr-stream") is True
    assert _cumulative("some-future-model-we-have-never-seen") is True
    # The one documented exception.
    assert _cumulative("step-asr-1.1-stream") is False


async def test_unknown_model_deltas_replace_not_append() -> None:
    # The end-to-end consequence of the above, for any model not on the
    # known-incremental denylist.
    s = stepfun.STT(model="stepaudio-9-asr-stream")
    _, events = await _drive(
        s,
        [
            _SESSION_READY,
            {"type": _DELTA, "text": "hello"},
            {"type": _DELTA, "text": "hello world"},
        ],
    )
    assert _texts(events, stt.SpeechEventType.INTERIM_TRANSCRIPT) == [
        "hello",
        "hello world",
    ]


def test_unknown_model_does_not_overdeclare_alignment() -> None:
    # Alignment stays an allowlist: claiming "word" for a model that never
    # emits `words` would break adaptive interruption silently.
    caps = stepfun.STT(
        model="some-future-model-we-have-never-seen", enable_timestamp_align=True
    ).capabilities
    assert caps.aligned_transcript is False


def test_rejected_model_not_advertised() -> None:
    """stepaudio-3-asr-stream is not offered, because the server refuses it.

    Tried against the live endpoint on 2026-10-14:
        {"code": "invalid_value", "param": "model",
         "message": "unsupported model: stepaudio-3-asr-stream"}

    It stays out of the Literal so editor completion cannot suggest it. Passing
    it explicitly still works — these types are for completion, not validation —
    and then fails at the server, which is the honest outcome.
    """
    from typing import get_args

    assert "stepaudio-3-asr-stream" not in get_args(stepfun.STTModels)


async def test_arbitrary_model_sent_in_session_update() -> None:
    # The Literal does not restrict what can be sent; whatever the caller
    # chooses reaches the server verbatim.
    s = stepfun.STT(model="stepaudio-9-asr-stream")
    ws, _ = await _drive(s, [_SESSION_READY])
    transcription = ws.frames_of_type("session.update")[0]["session"]["audio"]["input"][
        "transcription"
    ]
    assert transcription["model"] == "stepaudio-9-asr-stream"


async def test_session_created_does_not_acknowledge_configuration() -> None:
    s = stepfun.STT()
    ws = FakeWebSocket()
    s._pool = FakePool(ws)  # type: ignore[assignment]
    ws.queue_server_event({"type": "session.created"})
    ws.queue_server_event({"type": "error", "error": {"code": "invalid_value",
                                                       "message": "bad configuration"}})
    stream = s.stream(conn_options=_no_retry())
    stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIStatusError):
            [_ async for _ in stream]
        assert not ws.frames_of_type("input_audio_buffer.append")
        assert ws in s._pool.removed
    finally:
        await stream.aclose()


async def test_session_created_followed_by_updated_is_accepted() -> None:
    ws, _ = await _drive(stepfun.STT(), [{"type": "session.created"}, _SESSION_READY])
    assert ws.frames_of_type("input_audio_buffer.append")


async def test_end_input_commits_only_once() -> None:
    ws, _ = await _drive(stepfun.STT(server_vad=False), [_SESSION_READY])
    assert len(ws.frames_of_type("input_audio_buffer.commit")) == 1


async def test_empty_stream_does_not_commit() -> None:
    ws, _ = await _drive(stepfun.STT(server_vad=False), [_SESSION_READY], frames=0)
    assert not ws.frames_of_type("input_audio_buffer.commit")


async def test_streaming_reports_detected_language() -> None:
    _, events = await _drive(stepfun.STT(language="en"), [
        _SESSION_READY,
        {"type": _DELTA, "text": "你好", "language": "zh"},
        {"type": "conversation.item.input_audio_transcription.completed",
         "transcript": "你好", "language": "zh"},
    ])
    transcripts = [e for e in events if e.alternatives]
    assert transcripts
    assert all(e.alternatives[0].language == "zh" for e in transcripts)


async def test_batch_prewarm_does_not_open_streaming_socket() -> None:
    session = _FakeSSESession([])
    s = stepfun.STT(use_streaming=False, http_session=session)  # type: ignore[arg-type]
    pool = FakePool(FakeWebSocket())
    s._pool = pool  # type: ignore[assignment]
    s.prewarm()
    assert pool.prewarm_calls == 0
    await s.aclose()


@pytest.mark.parametrize("lines", [[], [
    'data: {"type":"transcript.text.delta","delta":"unfinished"}',
    'data: [DONE]',
]])
async def test_batch_rejects_truncated_response(lines: list[str]) -> None:
    from livekit.agents import APIConnectionError

    s = stepfun.STT(http_session=_FakeSSESession(lines))  # type: ignore[arg-type]
    with pytest.raises(APIConnectionError, match="before transcript.text.done"):
        await s.recognize([_frame()], conn_options=_no_retry())


async def test_batch_returns_at_done_without_waiting_for_eof() -> None:
    session = _FakeSSESession([
        'data: {"type":"transcript.text.done","text":"complete"}',
        'data: {"type":"error","message":"must not read past done"}',
    ])
    s = stepfun.STT(http_session=session)  # type: ignore[arg-type]
    event = await s.recognize([_frame()], conn_options=_no_retry())
    assert event.alternatives[0].text == "complete"


async def test_interleaved_items_keep_separate_transcripts() -> None:
    _, events = await _drive(stepfun.STT(), [
        _SESSION_READY,
        {"type": _DELTA, "item_id": "first", "text": "first turn"},
        {"type": _DELTA, "item_id": "second", "text": "second turn"},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "first"},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "second"},
    ])
    assert _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT) == ["first turn", "second turn"]


async def test_empty_final_does_not_restore_rejected_partial() -> None:
    _, events = await _drive(stepfun.STT(), [
        _SESSION_READY, {"type": _DELTA, "text": "discarded"},
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": ""},
    ])
    assert not _texts(events, stt.SpeechEventType.FINAL_TRANSCRIPT)


def test_model_update_refreshes_word_alignment_capability() -> None:
    speech = stepfun.STT(enable_timestamp_align=True)
    assert speech.capabilities.aligned_transcript == "word"
    speech.update_options(model="step-asr-1.1-stream")
    assert speech.capabilities.aligned_transcript is False
    speech.update_options(model="stepaudio-2.5-asr-stream")
    assert speech.capabilities.aligned_transcript == "word"


async def test_batch_empty_final_does_not_restore_rejected_delta() -> None:
    session = _FakeSSESession([
        'data: {"type":"transcript.text.delta","delta":"discarded"}',
        'data: {"type":"transcript.text.done","text":""}',
    ])
    speech = stepfun.STT(http_session=session)  # type: ignore[arg-type]
    event = await speech.recognize([_frame()])
    assert event.alternatives[0].text == ""
