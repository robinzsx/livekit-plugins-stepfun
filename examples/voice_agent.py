"""A complete voice agent running entirely on StepFun models."""

from __future__ import annotations

import logging

from _config import get, get_bool, load_env
from livekit.agents import (
    Agent,
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

_INTERRUPTION_DURATION = float(get("STEPFUN_MIN_INTERRUPTION_DURATION", "0.2"))
if not 0.05 <= _INTERRUPTION_DURATION <= 5.0:
    raise ValueError("STEPFUN_MIN_INTERRUPTION_DURATION must be between 0.05 and 5 seconds")


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


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a concise, friendly voice assistant. Keep answers "
                "conversational and short enough to listen to."
            )
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
    stt.prewarm()
    tts.prewarm()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"] if _USE_LOCAL_VAD else None,
        stt=stt,
        llm=stepfun.LLM(model=_LLM_MODEL, reasoning_effort=_REASONING_EFFORT),
        tts=tts,
        turn_handling=turn_handling(),
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

    try:
        await session.start(agent=Assistant(), room=ctx.room)
        # Replace input before the reader scheduled by start() can consume it.
        if denoiser is not None:
            if session.input.audio is not None:
                filtered_input = FilteredAudioInput(session.input.audio, processor=denoiser)
                session.input.audio = filtered_input
            else:
                denoiser.close()  # text-only console mode
    except BaseException:
        await _close_denoiser()
        raise


# The lk CLI imports this module and discovers this server without executing
# __main__. Register the same callbacks for both launch commands.
server = AgentServer(setup_fnc=prewarm)
server.rtc_session(entrypoint)


if __name__ == "__main__":
    cli.run_app(server)
