"""A complete voice agent running entirely on StepFun models."""

from __future__ import annotations

import logging

from _config import get, get_bool, load_env
from _eval_runtime import INSTRUCTIONS, EvaluationController, InstrumentedAgent
from _llm_providers import build_llm, describe_llm
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

# stepfun | deepseek. Chosen once per process; per-provider model IDs are read
# by the provider module (STEPFUN_LLM_MODEL, DEEPSEEK_MODEL), not here.
_LLM_PROVIDER = (get("LLM_PROVIDER", "stepfun") or "stepfun").strip().lower()
if _LLM_PROVIDER not in ("stepfun", "deepseek"):
    raise ValueError("LLM_PROVIDER must be 'stepfun' or 'deepseek'")

# low / medium / high. Documented for step-5-preview and step-3.7-flash and
# omitted for models that do not take it. Lower effort means less silence
# before the reply starts; "none" leaves the model's default alone.
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

# Minimum length of overlapping speech (s) that registers as an interruption.
# The SDK default is 0.5; a noisy room (another talker, a TV) needs the higher
# bar so transient noise does not cut off the agent mid-reply.
_INTERRUPTION_DURATION = float(get("STEPFUN_MIN_INTERRUPTION_DURATION", "0.5"))
if not 0.05 <= _INTERRUPTION_DURATION <= 5.0:
    raise ValueError("STEPFUN_MIN_INTERRUPTION_DURATION must be between 0.05 and 5 seconds")

# Noise-robust interruption handling. These are provider-agnostic: they use only
# the local VAD and the StepFun STT, no LiveKit Inference.
#   - resume_false_interruption: if an interruption is followed by silence with
#     no confirming transcript, the agent RESUMES its reply instead of staying
#     cut off. Requires an audio output that supports pause (room/telephony;
#     console sinks ignore it and log a warning).
#   - false_interruption_timeout: seconds of post-interruption silence before the
#     interruption is classified false. "none"/"off" disables the resume path.
#   - mode "adaptive" opts into LiveKit's ML interruption classifier (needs
#     LiveKit Inference credentials + VAD + STT); "vad" (default) is the plain,
#     deployment-agnostic strategy.
_INTERRUPTION_MODE = (get("STEPFUN_INTERRUPTION_MODE", "vad") or "vad").strip().lower()
if _INTERRUPTION_MODE not in ("vad", "adaptive"):
    raise ValueError("STEPFUN_INTERRUPTION_MODE must be 'vad' or 'adaptive'")
_RESUME_FALSE_INTERRUPTION = get_bool("STEPFUN_RESUME_FALSE_INTERRUPTION", True)
_fit_raw = (get("STEPFUN_FALSE_INTERRUPTION_TIMEOUT", "2.0") or "").strip().lower()
_FALSE_INTERRUPTION_TIMEOUT = None if _fit_raw in ("", "none", "off") else float(_fit_raw)

# Local (Silero) VAD tuning. In batch/http mode this VAD is what segments audio
# for recognize() and what fires interruptions, so it is the main lever for a
# noisy room: raise the activation threshold (0.5 -> ~0.6-0.7) and min_speech so
# background noise stops holding a segment open (which stalls ASR) and stops
# triggering false interruptions. min_silence_duration here is the BATCH
# endpointing silence; STEPFUN_VAD_SILENCE_DURATION_MS is the streaming
# server-VAD's, a separate knob.
_VAD_ACTIVATION_THRESHOLD = float(get("STEPFUN_VAD_ACTIVATION_THRESHOLD", "0.5"))
if not 0.0 < _VAD_ACTIVATION_THRESHOLD < 1.0:
    raise ValueError("STEPFUN_VAD_ACTIVATION_THRESHOLD must be between 0 and 1 (exclusive)")
_VAD_MIN_SPEECH_MS = int(get("STEPFUN_VAD_MIN_SPEECH_MS", "50"))
_VAD_MIN_SILENCE_MS = int(get("STEPFUN_VAD_MIN_SILENCE_MS", "550"))
if _VAD_MIN_SPEECH_MS < 0 or _VAD_MIN_SILENCE_MS <= 0:
    raise ValueError("STEPFUN_VAD_MIN_SPEECH_MS and STEPFUN_VAD_MIN_SILENCE_MS must be positive")
_vad_deact_raw = (get("STEPFUN_VAD_DEACTIVATION_THRESHOLD", "") or "").strip()
_VAD_DEACTIVATION_THRESHOLD = float(_vad_deact_raw) if _vad_deact_raw else None

# Batch endpointing. The audio turn detector (v1-mini) is a streaming detector, so
# the batch path inherits the streaming endpointing defaults (min_delay 0.3,
# max_delay 2.5). min_delay is only a floor applied AFTER the detector's decision
# (audio_recognition.py); because the decision already takes ~2.5 s on this corpus,
# min_delay is never binding, so pinning it to 0.0 (as the streaming path does)
# changes little. The real batch lever is max_delay: the detector rarely confirms
# end-of-turn here, so each batch turn currently waits the full 2.5 s. Lower
# STEPFUN_ENDPOINTING_MAX_DELAY to cut that, at the cost of committing before the
# user has clearly finished (premature-turn risk) - benchmark WER + premature rate.
_ENDPOINTING_MIN_DELAY = float(get("STEPFUN_ENDPOINTING_MIN_DELAY", "0.0"))
_ENDPOINTING_MAX_DELAY = float(get("STEPFUN_ENDPOINTING_MAX_DELAY", "2.5"))
if not 0.0 <= _ENDPOINTING_MIN_DELAY <= _ENDPOINTING_MAX_DELAY <= 10.0:
    raise ValueError(
        "STEPFUN_ENDPOINTING_MIN_DELAY/MAX_DELAY must satisfy 0 <= min <= max <= 10 seconds"
    )

# Batch turn detection. "vad" (default) relies purely on the local VAD + endpointing
# delay - no semantic end-of-turn model. This is the default because the v1-mini
# audio EOT model does not confirm end-of-turn on the eval corpus (it just waits
# max_delay). "v1-mini" is the local model, "v1" is the LiveKit Inference cloud
# model (needs LIVEKIT_API_KEY/SECRET, billed, agent-worker only) - kept for A/B
# and to re-check whether either confirms EOT on real conversational audio.
_TURN_DETECTOR_VERSION = get("STEPFUN_TURN_DETECTOR_VERSION", "vad")
if _TURN_DETECTOR_VERSION not in ("vad", "v1-mini", "v1"):
    raise ValueError("STEPFUN_TURN_DETECTOR_VERSION must be 'vad', 'v1-mini' or 'v1'")

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
        "mode": _INTERRUPTION_MODE,
        "min_duration": _INTERRUPTION_DURATION,
        "min_words": 0,
        "resume_false_interruption": _RESUME_FALSE_INTERRUPTION,
        "false_interruption_timeout": _FALSE_INTERRUPTION_TIMEOUT,
    }
    if _USE_STREAMING_ASR:
        return TurnHandlingOptions(
            turn_detection="stt",
            interruption=interruption,
            endpointing={"min_delay": 0.0},
        )

    # "vad" = pure VAD endpointing (no semantic EOT model); otherwise a turn
    # detector object. v1-mini/v1 rarely confirm on this corpus, so vad is default.
    turn_detection = (
        "vad"
        if _TURN_DETECTOR_VERSION == "vad"
        else inference.TurnDetector(version=_TURN_DETECTOR_VERSION)
    )
    return TurnHandlingOptions(
        turn_detection=turn_detection,
        interruption=interruption,
        endpointing={"min_delay": _ENDPOINTING_MIN_DELAY, "max_delay": _ENDPOINTING_MAX_DELAY},
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
        vad_kwargs: dict[str, float] = {
            "activation_threshold": _VAD_ACTIVATION_THRESHOLD,
            "min_speech_duration": _VAD_MIN_SPEECH_MS / 1000.0,
            "min_silence_duration": _VAD_MIN_SILENCE_MS / 1000.0,
        }
        if _VAD_DEACTIVATION_THRESHOLD is not None:
            vad_kwargs["deactivation_threshold"] = _VAD_DEACTIVATION_THRESHOLD
        proc.userdata["vad"] = silero.VAD.load(**vad_kwargs)


async def entrypoint(ctx: JobContext) -> None:
    logger.info(
        "audio input: transport=%s local_vad=%s noise=%s interruption_min_duration=%.2fs",
        "websocket" if _USE_STREAMING_ASR else "http",
        _USE_LOCAL_VAD,
        _NOISE_CANCELLATION,
        _INTERRUPTION_DURATION,
    )
    logger.info(
        "turn handling: interruption_mode=%s resume_false=%s false_timeout=%s "
        "vad(threshold=%.2f min_speech=%dms min_silence=%dms)",
        _INTERRUPTION_MODE,
        _RESUME_FALSE_INTERRUPTION,
        _FALSE_INTERRUPTION_TIMEOUT,
        _VAD_ACTIVATION_THRESHOLD,
        _VAD_MIN_SPEECH_MS,
        _VAD_MIN_SILENCE_MS,
    )
    if not _USE_STREAMING_ASR:
        logger.info(
            "batch turn detection: %s | endpointing min_delay=%.2fs max_delay=%.2fs",
            _TURN_DETECTOR_VERSION,
            _ENDPOINTING_MIN_DELAY,
            _ENDPOINTING_MAX_DELAY,
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

    chat = build_llm(reasoning_effort=_REASONING_EFFORT)

    logger.info("llm: %s", describe_llm(chat))
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
                **describe_llm(chat),
                "reasoning_effort": _REASONING_EFFORT,
                "tts_model": tts.model,
                "voice": _VOICE,
                "local_vad": _USE_LOCAL_VAD,
                "noise_cancellation": _NOISE_CANCELLATION,
                "prewarm": _PREWARM,
                "vad_silence_duration_ms": _VAD_SILENCE_MS,
                "min_interruption_duration_s": _INTERRUPTION_DURATION,
                "interruption_mode": _INTERRUPTION_MODE,
                "resume_false_interruption": _RESUME_FALSE_INTERRUPTION,
                "false_interruption_timeout_s": _FALSE_INTERRUPTION_TIMEOUT,
                "vad_activation_threshold": _VAD_ACTIVATION_THRESHOLD,
                "vad_min_speech_ms": _VAD_MIN_SPEECH_MS,
                "vad_min_silence_ms": _VAD_MIN_SILENCE_MS,
                "endpointing_min_delay_s": _ENDPOINTING_MIN_DELAY,
                "endpointing_max_delay_s": _ENDPOINTING_MAX_DELAY,
                "turn_detection": "stt" if _USE_STREAMING_ASR else _TURN_DETECTOR_VERSION,
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
