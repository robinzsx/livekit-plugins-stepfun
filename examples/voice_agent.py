"""A complete voice agent running entirely on StepFun models."""

from __future__ import annotations

import logging

from _config import get, get_bool, load_env
from _eval_runtime import INSTRUCTIONS, EvaluationController, InstrumentedAgent
from livekit.agents import (
    AgentServer,
    AgentSession,
    InterruptionOptions,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    inference,
    metrics,
)
from livekit.agents.voice.agent_session import SessionConnectOptions

from livekit.plugins import silero, stepfun

# Must run before anything reads os.environ — the plugin resolves its API key
# at construction time.
load_env()

logger = logging.getLogger("stepfun-voice-agent")

_LLM_MODEL = get("STEPFUN_LLM_MODEL", "step-3.7-flash")

# low / medium / high. Documented for step-3.7-flash; lower effort means less
# silence before the reply starts.
_REASONING_EFFORT = get("STEPFUN_REASONING_EFFORT", "low")

# Language is worth being explicit about: omitting it selects English
# server-side on both the ASR and the TTS, on both clusters.
_LANGUAGE = get("STEPFUN_LANGUAGE", "en")

_VOICE = get("STEPFUN_TTS_VOICE", "vibrant-youth")
_TTS_MODEL = get("STEPFUN_TTS_MODEL", "stepaudio-3-tts")

_NOISE_CANCELLATION = get("STEPFUN_NOISE_CANCELLATION", "off")
if _NOISE_CANCELLATION not in ("off", "rnnoise", "webrtc"):
    raise ValueError("STEPFUN_NOISE_CANCELLATION must be 'off', 'rnnoise' or 'webrtc'")

_USE_STREAMING_ASR = get("STEPFUN_ASR_TRANSPORT", "websocket") != "http"
if get("STEPFUN_ASR_TRANSPORT", "websocket") not in ("websocket", "http"):
    raise ValueError("STEPFUN_ASR_TRANSPORT must be 'websocket' or 'http'")

# Local VAD improves interruption latency but never owns streaming endpointing.
# Batch recognition always needs it to produce audio segments.
_USE_LOCAL_VAD = not _USE_STREAMING_ASR or get_bool("STEPFUN_LOCAL_VAD", True)

# Streaming ASR model. The endpoint has historically accepted only
# stepaudio-2.5-asr-stream, so override deliberately and check the transcript.
_ASR_MODEL = get("STEPFUN_ASR_MODEL", "stepaudio-2.5-asr-stream")
_BATCH_ASR_MODEL = get("STEPFUN_BATCH_ASR_MODEL", "stepaudio-3-asr-max")
_EVAL_PORT = get("STEPFUN_EVAL_PORT")
if _EVAL_PORT is not None and not 1 <= int(_EVAL_PORT) <= 65535:
    raise ValueError("STEPFUN_EVAL_PORT must be between 1 and 65535")
_PREWARM = get_bool("STEPFUN_PREWARM", True)
_VAD_SILENCE_MS = int(get("STEPFUN_VAD_SILENCE_DURATION_MS", "800"))
if _VAD_SILENCE_MS <= 0:
    raise ValueError("STEPFUN_VAD_SILENCE_DURATION_MS must be positive")

_INTERRUPTION_DURATION = float(get("STEPFUN_MIN_INTERRUPTION_DURATION", "0.2"))
if not 0.05 <= _INTERRUPTION_DURATION <= 5.0:
    raise ValueError("STEPFUN_MIN_INTERRUPTION_DURATION must be between 0.05 and 5 seconds")

# The SDK closes the whole session once unrecoverable errors from stt, llm or
# tts exceed this count. A provider blip — one dropped TTS websocket, say —
# must not end the call, so the example keeps a wide margin over the SDK
# default of 3. Transient failures are retried inside the stream first; this
# only bounds what survives that.
_MAX_UNRECOVERABLE_ERRORS = int(get("STEPFUN_MAX_UNRECOVERABLE_ERRORS", "25"))
if _MAX_UNRECOVERABLE_ERRORS < 1:
    raise ValueError("STEPFUN_MAX_UNRECOVERABLE_ERRORS must be at least 1")

# Failed evaluation trials are recorded, not fatal: the runner stops only
# after this many in a row, which means the agent itself is broken.
_EVAL_MAX_CONSECUTIVE_FAILURES = int(get("STEPFUN_EVAL_MAX_CONSECUTIVE_FAILURES", "3"))
if not 1 <= _EVAL_MAX_CONSECUTIVE_FAILURES <= 100:
    raise ValueError("STEPFUN_EVAL_MAX_CONSECUTIVE_FAILURES must be between 1 and 100")


def turn_handling() -> TurnHandlingOptions:
    interruption: InterruptionOptions = {
        "mode": "vad",
        "min_duration": _INTERRUPTION_DURATION,
        "min_words": 0,
        "resume_false_interruption": False,
    }
    if _USE_STREAMING_ASR:
        return TurnHandlingOptions(
            turn_detection="stt",
            interruption=interruption,
            endpointing={"min_delay": 0.0},
        )

    return TurnHandlingOptions(
        turn_detection=inference.TurnDetector(version="v1-mini"), interruption=interruption
    )


class Assistant(InstrumentedAgent):
    def __init__(self, *, evaluation=None) -> None:
        super().__init__(
            instructions=INSTRUCTIONS,
            evaluation=evaluation,
        )


def prewarm(proc: JobProcess) -> None:
    """Load local VAD weights before the first session."""
    if _USE_LOCAL_VAD:
        proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    logger.info(
        "audio input: transport=%s local_vad=%s noise=%s interruption_min_duration=%.2fs",
        "websocket" if _USE_STREAMING_ASR else "http",
        _USE_LOCAL_VAD,
        _NOISE_CANCELLATION,
        _INTERRUPTION_DURATION,
    )
    # Per-session native state: never share a denoiser across callers.
    from _noise_cancellation import FilteredAudioInput, NoiseSuppressor

    denoiser = NoiseSuppressor(method=_NOISE_CANCELLATION) if _NOISE_CANCELLATION != "off" else None
    filtered_input: FilteredAudioInput | None = None

    async def _close_denoiser() -> None:
        if filtered_input is not None:
            await filtered_input.aclose()
        elif denoiser is not None:
            denoiser.close()

    if denoiser is not None:
        ctx.add_shutdown_callback(_close_denoiser)

    stt = stepfun.STT(
        language=_LANGUAGE,
        model=_ASR_MODEL,
        batch_model=_BATCH_ASR_MODEL,
        use_streaming=_USE_STREAMING_ASR,
        server_vad=True,
        vad_silence_duration_ms=_VAD_SILENCE_MS,
    )
    tts = stepfun.TTS(
        model=_TTS_MODEL,
        voice=_VOICE,
        language=_LANGUAGE,
        # Only honoured by stepaudio-2.5-tts and stepaudio-3-tts.
        instruction="warm, professional, unhurried",
    )

    # Pays for DNS, TLS, the WebSocket upgrade and — for TTS — the session
    # handshake before the user says anything.
    if _PREWARM:
        stt.prewarm()
        tts.prewarm()

    chat = stepfun.LLM(
        model=_LLM_MODEL,
        **({"reasoning_effort": _REASONING_EFFORT} if _REASONING_EFFORT != "none" else {}),
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"] if _USE_LOCAL_VAD else None,
        stt=stt,
        llm=chat,
        tts=tts,
        turn_handling=turn_handling(),
        conn_options=SessionConnectOptions(max_unrecoverable_errors=_MAX_UNRECOVERABLE_ERRORS),
    )

    # Where the latency numbers come from: ASR first partial and final, LLM
    # time to first token, TTS time to first byte.
    usage = metrics.ModelUsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: metrics.MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def _log_usage() -> None:
        logger.info("usage summary: %s", usage.flatten())

    ctx.add_shutdown_callback(_log_usage)

    evaluation = None
    if _EVAL_PORT is not None:
        evaluation = EvaluationController(
            session,
            providers={"stt": stt, "llm": chat, "tts": tts},
            max_consecutive_failures=_EVAL_MAX_CONSECUTIVE_FAILURES,
            config={
                "cluster": stt._opts.cluster,
                "language": _LANGUAGE,
                "asr_transport": "websocket" if _USE_STREAMING_ASR else "http",
                "asr_model": stt.model,
                "batch_asr_model": _BATCH_ASR_MODEL,
                "llm_model": chat.model,
                "reasoning_effort": _REASONING_EFFORT,
                "tts_model": tts.model,
                "voice": _VOICE,
                "local_vad": _USE_LOCAL_VAD,
                "noise_cancellation": _NOISE_CANCELLATION,
                "prewarm": _PREWARM,
                "vad_silence_duration_ms": _VAD_SILENCE_MS,
                "min_interruption_duration_s": _INTERRUPTION_DURATION,
                "turn_detection": "stt" if _USE_STREAMING_ASR else "v1-mini",
            },
        )
        ctx.add_shutdown_callback(evaluation.aclose)

    try:
        await session.start(agent=Assistant(evaluation=evaluation), room=ctx.room)
        if evaluation is not None:
            from _eval_audio import ObservedAudioOutput

            if session.output.audio is None:
                raise RuntimeError("Evaluation requires audio console mode; remove --text")
            session.input.audio = evaluation.audio
            session.output.audio = ObservedAudioOutput(
                session.output.audio, get_record=lambda: evaluation.active
            )
        # Replace input before the reader scheduled by start() can consume it.
        if denoiser is not None:
            if session.input.audio is not None:
                filtered_input = FilteredAudioInput(session.input.audio, processor=denoiser)
                session.input.audio = filtered_input
            else:
                denoiser.close()  # text-only console mode
        if evaluation is not None:
            await evaluation.start(int(_EVAL_PORT))
    except BaseException:
        if evaluation is not None:
            await evaluation.aclose()
        await _close_denoiser()
        raise


# The lk CLI imports this module and discovers this server without executing
# __main__. Register the same callbacks for both launch commands.
server = AgentServer(setup_fnc=prewarm)
server.rtc_session(entrypoint)


if __name__ == "__main__":
    cli.run_app(server)
