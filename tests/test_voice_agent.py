"""Pin the example's actual transport-to-session configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def load_agent(monkeypatch):
    pytest.importorskip("dotenv")
    pytest.importorskip("livekit.plugins.silero")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import _config

    monkeypatch.setattr(_config, "load_env", lambda: [])
    monkeypatch.delenv("STEPFUN_EVAL_PORT", raising=False)
    monkeypatch.setenv("STEPFUN_PREWARM", "true")

    def load(transport="websocket", local_vad="true", noise="off", turn_detector="vad"):
        monkeypatch.setenv("STEPFUN_MIN_INTERRUPTION_DURATION", "0.2")
        monkeypatch.setenv("STEPFUN_ASR_TRANSPORT", transport)
        monkeypatch.setenv("STEPFUN_LOCAL_VAD", local_vad)
        monkeypatch.setenv("STEPFUN_NOISE_CANCELLATION", noise)
        monkeypatch.setenv("STEPFUN_TURN_DETECTOR_VERSION", turn_detector)
        path = Path(__file__).resolve().parents[1] / "examples" / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("stepfun_voice_agent", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.mark.parametrize(
    "transport,local_vad,needs_vad",
    [
        ("websocket", "true", True),
        ("websocket", "false", False),
        ("http", "false", True),
    ],
)
async def test_entrypoint_configures_turns_before_session_creation(
    load_agent, monkeypatch, transport, local_vad, needs_vad
):
    module = load_agent(transport, local_vad)
    vad = object()
    monkeypatch.setattr(module.silero.VAD, "load", lambda **kwargs: vad)
    proc = SimpleNamespace(userdata={})
    module.prewarm(proc)
    assert ("vad" in proc.userdata) == needs_vad
    for component in ("STT", "TTS", "LLM"):
        monkeypatch.setattr(module.stepfun, component, MagicMock())
    session = MagicMock()
    session.start = AsyncMock()
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(module, "AgentSession", factory)
    ctx = SimpleNamespace(proc=proc, room=object(), add_shutdown_callback=MagicMock())
    await module.entrypoint(ctx)
    opts = factory.call_args.kwargs
    assert opts["vad"] is (vad if needs_vad else None)
    assert opts["turn_handling"]["interruption"]["resume_false_interruption"] is True
    assert opts["turn_handling"]["interruption"]["min_duration"] == 0.2
    assert opts["turn_handling"]["interruption"]["min_words"] == 0
    assert opts["turn_handling"]["interruption"]["mode"] == "vad"
    assert opts["turn_handling"]["interruption"]["false_interruption_timeout"] == 2.0
    if transport == "http":
        assert opts["turn_handling"]["endpointing"]["min_delay"] == 0.0
        assert opts["turn_handling"]["endpointing"]["max_delay"] == 2.5
    turn_detector = opts["turn_handling"]["turn_detection"]
    if transport == "http":
        assert turn_detector == "vad"  # pure-VAD default (no semantic EOT model)
    else:
        assert turn_detector == "stt"
    assert module.stepfun.STT.call_args.kwargs["use_streaming"] == (transport == "websocket")
    # A single provider blip must not close the session: the SDK shuts the
    # session down once unrecoverable errors exceed this count.
    assert opts["conn_options"].max_unrecoverable_errors == 25
    session.start.assert_awaited_once()


async def test_batch_can_opt_into_semantic_turn_detector(load_agent, monkeypatch):
    """vad is the default (v1-mini does not confirm EOT on the corpus), but the
    local mini detector stays selectable for a real-time A/B."""
    module = load_agent(transport="http", local_vad="true", turn_detector="v1-mini")
    monkeypatch.setattr(module.silero.VAD, "load", lambda **kwargs: object())
    proc = SimpleNamespace(userdata={})
    module.prewarm(proc)
    for component in ("STT", "TTS", "LLM"):
        monkeypatch.setattr(module.stepfun, component, MagicMock())
    session = MagicMock()
    session.start = AsyncMock()
    factory = MagicMock(return_value=session)
    monkeypatch.setattr(module, "AgentSession", factory)
    ctx = SimpleNamespace(proc=proc, room=object(), add_shutdown_callback=MagicMock())
    await module.entrypoint(ctx)
    detector = factory.call_args.kwargs["turn_handling"]["turn_detection"]
    assert detector.model == "turn-detector-v1-mini"


async def test_unrecoverable_error_tolerance_is_configurable(load_agent, monkeypatch):
    """The example keeps the session alive through provider blips, and the
    tolerance is tunable per deployment."""
    session = MagicMock()
    session.start = AsyncMock()
    factory = MagicMock(return_value=session)
    ctx = SimpleNamespace(
        proc=SimpleNamespace(userdata={"vad": object()}),
        room=object(),
        add_shutdown_callback=MagicMock(),
    )

    async def tolerance_with(env: str | None) -> int:
        if env is None:
            monkeypatch.delenv("STEPFUN_MAX_UNRECOVERABLE_ERRORS", raising=False)
        else:
            monkeypatch.setenv("STEPFUN_MAX_UNRECOVERABLE_ERRORS", env)
        module = load_agent()
        for component in ("STT", "TTS", "LLM"):
            monkeypatch.setattr(module.stepfun, component, MagicMock())
        monkeypatch.setattr(module, "AgentSession", factory)
        await module.entrypoint(ctx)
        return factory.call_args.kwargs["conn_options"].max_unrecoverable_errors

    assert await tolerance_with(None) == 25
    assert await tolerance_with("5") == 5


def test_invalid_error_tolerance_is_rejected(load_agent, monkeypatch):
    monkeypatch.setenv("STEPFUN_MAX_UNRECOVERABLE_ERRORS", "0")
    with pytest.raises(ValueError, match="STEPFUN_MAX_UNRECOVERABLE_ERRORS"):
        load_agent()


def test_invalid_eval_failure_limit_is_rejected(load_agent, monkeypatch):
    monkeypatch.setenv("STEPFUN_EVAL_MAX_CONSECUTIVE_FAILURES", "0")
    with pytest.raises(ValueError, match="STEPFUN_EVAL_MAX_CONSECUTIVE_FAILURES"):
        load_agent()


def test_invalid_transport_fails_instead_of_silently_using_streaming(load_agent):
    with pytest.raises(ValueError, match="STEPFUN_ASR_TRANSPORT"):
        load_agent("typo")


@pytest.mark.parametrize("transport", ["websocket", "http"])
@pytest.mark.parametrize("input_kind", ["console", "room"])
@pytest.mark.parametrize("noise", ["rnnoise", "webrtc"])
async def test_noise_filter_wraps_input_selected_during_start(
    load_agent, monkeypatch, transport, input_kind, noise
):
    import _noise_cancellation as noise_module
    import numpy as np

    from .test_noise_cancellation import Backend, Feed, frame

    backend = Backend()
    processor = noise_module.NoiseSuppressor(backend=backend)
    monkeypatch.setattr(noise_module, "NoiseSuppressor", lambda **kwargs: processor)
    module = load_agent(transport, noise=noise)
    for component in ("STT", "TTS", "LLM"):
        monkeypatch.setattr(module.stepfun, component, MagicMock())
    source = Feed([frame(np.zeros(480, dtype=np.int16))])
    source.input_kind = input_kind
    session = MagicMock()
    session.input.audio = None

    async def start(**kwargs):
        # Console and RoomIO select their input during start, overriding any
        # input assigned beforehand. Only the selected input should be wrapped.
        session.input.audio = source

    session.start = AsyncMock(side_effect=start)
    monkeypatch.setattr(module, "AgentSession", MagicMock(return_value=session))
    ctx = SimpleNamespace(
        proc=SimpleNamespace(userdata={"vad": object()}),
        room=object(),
        add_shutdown_callback=MagicMock(),
    )
    await module.entrypoint(ctx)
    assert isinstance(session.input.audio, noise_module.FilteredAudioInput)
    assert session.input.audio.source is source
    for call in ctx.add_shutdown_callback.call_args_list:
        await call.args[0]()
    assert backend.closed == 1


def test_invalid_noise_setting_is_rejected(load_agent):
    with pytest.raises(ValueError, match="STEPFUN_NOISE_CANCELLATION"):
        load_agent(noise="typo")


@pytest.mark.parametrize("transport", ["websocket", "http"])
def test_livekit_cli_discovers_registered_server(load_agent, monkeypatch, transport):
    import sys

    from livekit.agents import AgentServer
    from livekit.agents.cli.discover import get_import_data

    module = load_agent(transport)
    monkeypatch.setitem(sys.modules, "voice_agent", module)
    discovered = get_import_data(
        path=Path(__file__).resolve().parents[1] / "examples" / "voice_agent.py"
    )
    assert discovered.import_string == "voice_agent:server"
    assert isinstance(module.server, AgentServer)
    assert module.server.setup_fnc is module.prewarm
    assert module.server._entrypoint_fnc is module.entrypoint


def test_python_entrypoint_runs_the_same_server(load_agent, monkeypatch):
    import runpy

    from livekit.agents import cli

    load_agent()  # Configure the isolated example environment.
    run_app = MagicMock()
    monkeypatch.setattr(cli, "run_app", run_app)
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "voice_agent.py"),
        run_name="__main__",
    )
    run_app.assert_called_once_with(namespace["server"])


async def test_eval_mode_replaces_microphone_and_keeps_console_output(load_agent, monkeypatch):
    from _eval_audio import ObservedAudioOutput

    from .test_eval_agent import DelayedOutput

    monkeypatch.setenv("STEPFUN_EVAL_PORT", "8765")
    module = load_agent()
    for component in ("STT", "TTS", "LLM"):
        monkeypatch.setattr(module.stepfun, component, MagicMock())
    evaluation = SimpleNamespace(audio=object(), active=None, start=AsyncMock(), aclose=AsyncMock())
    factory = MagicMock(return_value=evaluation)
    monkeypatch.setattr(module, "EvaluationController", factory)
    session = MagicMock()
    session.start = AsyncMock()
    session.output.audio = DelayedOutput()
    monkeypatch.setattr(module, "AgentSession", MagicMock(return_value=session))
    ctx = SimpleNamespace(
        proc=SimpleNamespace(userdata={"vad": object()}),
        room=object(),
        add_shutdown_callback=MagicMock(),
    )
    await module.entrypoint(ctx)
    evaluation.start.assert_awaited_once_with(8765)
    assert session.input.audio is evaluation.audio
    assert isinstance(session.output.audio, ObservedAudioOutput)
    assert session.start.call_args.kwargs["agent"]._evaluation is evaluation


def test_invalid_eval_port_is_rejected(load_agent, monkeypatch):
    monkeypatch.setenv("STEPFUN_EVAL_PORT", "70000")
    with pytest.raises(ValueError, match="STEPFUN_EVAL_PORT"):
        load_agent()
