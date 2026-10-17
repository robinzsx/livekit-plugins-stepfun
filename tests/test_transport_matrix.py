"""Pin the VAD / turn-detection / noise-filter matrix across both ASR transports.

The three interact, and the failure mode is silent: audio keeps flowing, the
agent just decides when a turn ended based on the wrong signal. These tests
drive the real `entrypoint()` so the matrix is the example's own configuration,
not a paraphrase of it.

    transport | local VAD | endpointing owner | interruption source
    ----------+-----------+-------------------+--------------------
    websocket | on        | StepFun server    | local VAD
    websocket | off       | StepFun server    | server transcript
    http      | forced on | local VAD+detector| local VAD
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("dotenv")
pytest.importorskip("livekit.plugins.silero")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "voice_agent.py"


@pytest.fixture
def load_agent(monkeypatch):
    """Import the example fresh per test, as env is read at module scope."""
    pytest.importorskip("livekit.plugins.stepfun")
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    monkeypatch.delenv("STEPFUN_EVAL_PORT", raising=False)
    monkeypatch.setenv("STEPFUN_PREWARM", "false")

    def load(*, transport="websocket", local_vad="true", noise="off"):
        monkeypatch.setenv("STEPFUN_ASR_TRANSPORT", transport)
        monkeypatch.setenv("STEPFUN_LOCAL_VAD", local_vad)
        monkeypatch.setenv("STEPFUN_NOISE_CANCELLATION", noise)
        spec = importlib.util.spec_from_file_location("voice_agent_matrix", EXAMPLE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


class _Input:
    """Stand-in for the console/room input a session selects during start()."""

    def __init__(self) -> None:
        self.reads = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.reads += 1
        raise StopAsyncIteration


def _run_entrypoint(module, monkeypatch, *, vad_present: bool):
    """Execute entrypoint() with the network-facing pieces stubbed out.

    Returns the session the example constructed, plus the shutdown callbacks it
    registered, so both the wiring and its teardown get checked.
    """
    for component in ("STT", "TTS"):
        monkeypatch.setattr(module.stepfun, component, MagicMock())
    monkeypatch.setattr(module, "build_llm", MagicMock())
    monkeypatch.setattr(module, "describe_llm", MagicMock(return_value={}))

    selected_input = _Input()
    session = MagicMock()
    session.input = SimpleNamespace(audio=None)
    session.output = SimpleNamespace(audio=None)

    async def start(**_kwargs):
        # Console and RoomIO both assign input.audio during start().
        session.input.audio = selected_input

    session.start = AsyncMock(side_effect=start)

    factory = MagicMock(return_value=session)
    monkeypatch.setattr(module, "AgentSession", factory)

    callbacks = []
    proc_userdata = {"vad": "VAD_WEIGHTS"} if vad_present else {}
    ctx = SimpleNamespace(
        proc=SimpleNamespace(userdata=proc_userdata),
        room=object(),
        add_shutdown_callback=callbacks.append,
    )
    return session, factory, callbacks, ctx, selected_input


def _run(coro):

    return asyncio.run(coro)


# -- local VAD presence -----------------------------------------------------


@pytest.mark.parametrize(
    "transport,local_vad,expect_vad_in_session",
    [
        # Streaming: local VAD is optional, it only drives interruption.
        ("websocket", "true", True),
        ("websocket", "false", False),
        # Batch: no server endpointing to fall back on, so VAD is mandatory
        # even when the operator asked for it off.
        ("http", "false", True),
        ("http", "true", True),
    ],
)
def test_local_vad_is_present_exactly_when_endpointing_needs_it(
    load_agent, monkeypatch, transport, local_vad, expect_vad_in_session
):
    module = load_agent(transport=transport, local_vad=local_vad)

    # prewarm() is what decides whether VAD weights are loaded at all.
    proc = SimpleNamespace(userdata={})
    monkeypatch.setattr(module.silero.VAD, "load", lambda: "VAD_WEIGHTS")
    module.prewarm(proc)
    assert ("vad" in proc.userdata) is expect_vad_in_session

    session, factory, _callbacks, ctx, _input = _run_entrypoint(
        module, monkeypatch, vad_present=expect_vad_in_session
    )
    ctx.proc.userdata = proc.userdata
    _run(module.entrypoint(ctx))

    got = factory.call_args.kwargs["vad"]
    if expect_vad_in_session:
        assert got == "VAD_WEIGHTS", "VAD must be handed to the session"
    else:
        assert got is None, "streaming may run without local VAD"


# -- turn detection ownership ----------------------------------------------


@pytest.mark.parametrize(
    "transport,expected",
    [
        # The server's endpointing is authoritative: no local turn model, and
        # no added endpointing delay on top of the server's decision.
        ("websocket", "stt"),
        # Batch has no server signal, so a local end-of-turn model plus the
        # default endpointing delay decides.
        ("http", "detector"),
    ],
)
def test_endpointing_is_owned_by_the_server_or_a_local_detector(load_agent, transport, expected):
    module = load_agent(transport=transport)
    options = module.turn_handling()

    if expected == "stt":
        assert options["turn_detection"] == "stt"
        assert options["endpointing"]["min_delay"] == 0.0, (
            "a local delay would stack on top of the server's endpointing"
        )
    else:
        detection = options["turn_detection"]
        assert detection is not None and detection != "stt"
        assert getattr(detection, "model", "").startswith("turn-detector"), detection


@pytest.mark.parametrize("transport", ["websocket", "http"])
def test_interruption_always_uses_local_vad(load_agent, transport):
    """Barge-in must not wait for a transcript.

    The server's speech_started event is what tells us a turn began, but
    stopping our own playback on it would add a round trip; local VAD reacts
    on the same audio the moment energy crosses the threshold.
    """
    module = load_agent(transport=transport)
    interruption = module.turn_handling()["interruption"]
    assert interruption["mode"] == "vad"
    assert interruption["min_duration"] == 0.2
    assert interruption["resume_false_interruption"] is False


# -- server VAD is only asked for when streaming ---------------------------


@pytest.mark.parametrize("transport", ["websocket", "http"])
def test_server_vad_configuration_follows_the_transport(load_agent, monkeypatch, transport):
    module = load_agent(transport=transport)
    stt_factory = MagicMock()

    _session, _factory, _callbacks, ctx, _input = _run_entrypoint(
        module, monkeypatch, vad_present=True
    )
    # After the helper, which stubs STT with its own mock.
    monkeypatch.setattr(module.stepfun, "STT", stt_factory)
    _run(module.entrypoint(ctx))

    kwargs = stt_factory.call_args.kwargs
    assert kwargs["use_streaming"] is (transport == "websocket")
    assert kwargs["server_vad"] is True
    # Silence window is only meaningful to the server VAD; harmless on batch,
    # where it is simply ignored.
    assert kwargs["vad_silence_duration_ms"] == 800


# -- noise filtering reaches everything -----------------------------------


@pytest.mark.parametrize("transport", ["websocket", "http"])
@pytest.mark.parametrize("noise", ["webrtc", "rnnoise"])
def test_noise_filter_wraps_whatever_input_start_selected(
    load_agent, monkeypatch, transport, noise
):
    """The filter must sit between the device and the session.

    STT, VAD and turn detection all consume the frames pushed by the session's
    single forward task, so replacing input.audio is what makes the filter
    apply to all three rather than only to recognition.
    """
    noise_module = pytest.importorskip("examples._noise_cancellation")
    processor = MagicMock()
    monkeypatch.setattr(noise_module, "NoiseSuppressor", lambda **_kwargs: processor)
    module = load_agent(transport=transport, noise=noise)

    session, _factory, callbacks, ctx, selected_input = _run_entrypoint(
        module, monkeypatch, vad_present=True
    )
    _run(module.entrypoint(ctx))

    wrapped = session.input.audio
    assert wrapped is not None and wrapped is not selected_input, (
        "the raw input must be replaced by the filter"
    )
    assert getattr(wrapped, "source", None) is selected_input
    assert noise in wrapped.label.lower(), wrapped.label

    # The denoiser must be released on shutdown, not leaked per call.
    assert callbacks, "a shutdown callback should own the denoiser"
    for callback in callbacks:
        result = callback()
        if hasattr(result, "__await__"):
            _run(result)


@pytest.mark.parametrize("transport", ["websocket", "http"])
def test_noise_off_leaves_the_input_untouched(load_agent, monkeypatch, transport):
    module = load_agent(transport=transport, noise="off")
    session, _factory, _callbacks, ctx, selected_input = _run_entrypoint(
        module, monkeypatch, vad_present=True
    )
    _run(module.entrypoint(ctx))
    assert session.input.audio is selected_input


def test_unknown_noise_backend_is_rejected_at_import(load_agent):
    with pytest.raises(ValueError, match="STEPFUN_NOISE_CANCELLATION"):
        load_agent(noise="bogus")
