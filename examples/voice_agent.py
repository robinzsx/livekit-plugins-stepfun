"""A complete voice agent running on StepFun models, ready to fork.

This is the starting point: a working speech-to-speech agent you can edit in
place. Everything provider-specific lives behind environment variables in
`.env`, so swapping a model is a config change rather than a code change.

    uv sync --extra examples
    uv run lk agent console examples/voice_agent.py   # talk to it locally
    uv run lk agent dev examples/voice_agent.py       # connect to a room

The two commands differ only in transport: `console` uses your microphone and
speakers in the terminal, `dev` joins a LiveKit room and needs LIVEKIT_URL,
LIVEKIT_API_KEY and LIVEKIT_API_SECRET.
"""

from __future__ import annotations

import logging

from _config import get, get_bool, load_env
from _llm_providers import build_llm, describe_llm
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

# Change this prompt first: it is the whole personality of the agent.
INSTRUCTIONS = (
    "You are a concise, friendly voice assistant. Keep answers conversational "
    "and short enough to listen to."
)

# What is said when the caller connects and says nothing. `none` waits for
# them to speak first, which is usually better for an inbound call. Written
# this way because `get()` treats an empty value as unset, so "" would silently
# fall back to the default greeting.
_GREETING = get("AGENT_GREETING", "Hi! How can I help you today?")
GREETING = None if (_GREETING or "").strip().lower() in ("none", "off") else _GREETING

# -- chat model --------------------------------------------------------------
# stepfun | deepseek. Chosen once per process; model IDs are read by
# _llm_providers, which owns STEPFUN_LLM_MODEL and DEEPSEEK_MODEL.
_LLM_PROVIDER = (get("LLM_PROVIDER", "stepfun") or "stepfun").strip().lower()
if _LLM_PROVIDER not in ("stepfun", "deepseek"):
    raise ValueError("LLM_PROVIDER must be 'stepfun' or 'deepseek'")

# low / medium / high. Documented for step-5-preview and step-3.7-flash and
# omitted for models that do not take it. Lower effort means less silence
# before the reply starts; "none" leaves the model's default alone.
_REASONING_EFFORT = get("STEPFUN_REASONING_EFFORT", "low")

# -- speech ------------------------------------------------------------------
# Language is worth being explicit about: omitting it selects English
# server-side on both the ASR and the TTS, on both clusters.
_LANGUAGE = get("STEPFUN_LANGUAGE", "en")

# global (api.stepfun.ai) or china (api.stepfun.com). Credentials are
# cluster-specific: a key from one will not authenticate against the other.
_STT_MODEL = get("STEPFUN_ASR_MODEL", "stepaudio-2.5-asr-stream")
_BATCH_ASR_MODEL = get("STEPFUN_BATCH_ASR_MODEL", "stepaudio-3-asr-max")
_TTS_MODEL = get("STEPFUN_TTS_MODEL", "stepaudio-3-tts")
_VOICE = get("STEPFUN_TTS_VOICE", "vibrant-youth")

# websocket streams interim transcripts as the caller speaks; http waits for
# the whole utterance and needs local VAD to find its boundaries.
_USE_STREAMING_ASR = get("STEPFUN_ASR_TRANSPORT", "websocket") != "http"
if get("STEPFUN_ASR_TRANSPORT", "websocket") not in ("websocket", "http"):
    raise ValueError("STEPFUN_ASR_TRANSPORT must be 'websocket' or 'http'")

# Local VAD also makes interruptions feel instant; it never owns endpointing
# while streaming, where the server decides when a turn has ended.
_USE_LOCAL_VAD = not _USE_STREAMING_ASR or get_bool("STEPFUN_LOCAL_VAD", True)

_VAD_SILENCE_MS = int(get("STEPFUN_VAD_SILENCE_DURATION_MS", "800"))
if _VAD_SILENCE_MS <= 0:
    raise ValueError("STEPFUN_VAD_SILENCE_DURATION_MS must be positive")

# Local (Silero) VAD tuning. Governs batch segmentation and interruptions. Raise
# the activation threshold (0.5 -> ~0.6-0.7) and min_speech in a noisy room so
# background noise stops holding a segment open (which stalls ASR) and stops
# false barge-in. min_silence here is the BATCH endpointing silence (distinct from
# STEPFUN_VAD_SILENCE_DURATION_MS, the streaming server-VAD's).
_VAD_ACTIVATION_THRESHOLD = float(get("STEPFUN_VAD_ACTIVATION_THRESHOLD", "0.5"))
if not 0.0 < _VAD_ACTIVATION_THRESHOLD < 1.0:
    raise ValueError("STEPFUN_VAD_ACTIVATION_THRESHOLD must be between 0 and 1 (exclusive)")
_VAD_MIN_SPEECH_MS = int(get("STEPFUN_VAD_MIN_SPEECH_MS", "50"))
_VAD_MIN_SILENCE_MS = int(get("STEPFUN_VAD_MIN_SILENCE_MS", "550"))
if _VAD_MIN_SPEECH_MS < 0 or _VAD_MIN_SILENCE_MS <= 0:
    raise ValueError("STEPFUN_VAD_MIN_SPEECH_MS and STEPFUN_VAD_MIN_SILENCE_MS must be positive")
_vad_deact_raw = (get("STEPFUN_VAD_DEACTIVATION_THRESHOLD", "") or "").strip()
_VAD_DEACTIVATION_THRESHOLD = float(_vad_deact_raw) if _vad_deact_raw else None

# ASR rate and the post-noise-cancellation downsample target. 16 kHz is the
# VAD/turn-detector/ASR standard and sufficient for speech; the 48 kHz input is
# resampled to it (noise cancellation stays at 48 kHz).
_STT_SAMPLE_RATE = int(get("STEPFUN_STT_SAMPLE_RATE", "16000"))
if _STT_SAMPLE_RATE <= 0:
    raise ValueError("STEPFUN_STT_SAMPLE_RATE must be positive")

# Batch endpointing delays (s). In vad mode min_delay is the effective end-of-turn
# cost (a fixed silence floor after the VAD closes the segment); max_delay only
# binds when a semantic detector is active and does not confirm.
_ENDPOINTING_MIN_DELAY = float(get("STEPFUN_ENDPOINTING_MIN_DELAY", "0.0"))
_ENDPOINTING_MAX_DELAY = float(get("STEPFUN_ENDPOINTING_MAX_DELAY", "2.5"))
if not 0.0 <= _ENDPOINTING_MIN_DELAY <= _ENDPOINTING_MAX_DELAY <= 10.0:
    raise ValueError(
        "STEPFUN_ENDPOINTING_MIN_DELAY/MAX_DELAY must satisfy 0 <= min <= max <= 10 seconds"
    )

# Batch turn detection. "vad" (default) is pure VAD + endpointing, no semantic EOT
# model (the local v1-mini model does not reliably confirm end-of-turn). "v1-mini"
# is the local model; "v1" is the LiveKit Inference cloud model (needs
# LIVEKIT_API_KEY/SECRET, billed, agent-worker only) - kept for A/B. Streaming
# mode always uses server-VAD ("stt") and ignores this.
_TURN_DETECTOR_VERSION = get("STEPFUN_TURN_DETECTOR_VERSION", "vad")
if _TURN_DETECTOR_VERSION not in ("vad", "v1-mini", "v1"):
    raise ValueError("STEPFUN_TURN_DETECTOR_VERSION must be 'vad', 'v1-mini' or 'v1'")

# Min length (s) of overlapping speech that registers as a barge-in. The SDK
# default is 0.5; a noisy room needs the higher bar so transient noise does not
# cut off the reply.
_INTERRUPTION_DURATION = float(get("STEPFUN_MIN_INTERRUPTION_DURATION", "0.5"))
if not 0.05 <= _INTERRUPTION_DURATION <= 5.0:
    raise ValueError("STEPFUN_MIN_INTERRUPTION_DURATION must be between 0.05 and 5 seconds")

# Noise-robust interruption handling (provider-agnostic: VAD + StepFun STT only).
#   - resume_false_interruption: if an interruption is followed by silence with no
#     confirming transcript, the agent RESUMES its reply. Needs a pause-capable
#     audio output (room/telephony; console sinks ignore it).
#   - false_interruption_timeout: seconds of post-interruption silence before it is
#     classified false; "none"/"off" disables the resume path.
#   - mode "adaptive" opts into LiveKit's ML interruption classifier (needs LiveKit
#     Inference credentials); "vad" (default) is deployment-agnostic.
_INTERRUPTION_MODE = (get("STEPFUN_INTERRUPTION_MODE", "vad") or "vad").strip().lower()
if _INTERRUPTION_MODE not in ("vad", "adaptive"):
    raise ValueError("STEPFUN_INTERRUPTION_MODE must be 'vad' or 'adaptive'")
_RESUME_FALSE_INTERRUPTION = get_bool("STEPFUN_RESUME_FALSE_INTERRUPTION", True)
_fit_raw = (get("STEPFUN_FALSE_INTERRUPTION_TIMEOUT", "2.0") or "").strip().lower()
_FALSE_INTERRUPTION_TIMEOUT = None if _fit_raw in ("", "none", "off") else float(_fit_raw)

# off | webrtc | rnnoise. Filtering happens before the session fans audio out to
# STT, VAD and turn detection, so all three see the same cleaned signal.
#
# webrtc is the usual choice: the WebRTC audio-processing module is already in
# the LiveKit SDK and costs nothing extra. rnnoise is a stronger model but is a
# native build, hence a separate `noise-cancellation` extra; ask for it
# explicitly rather than falling back to it and failing at import time.
_NOISE_CANCELLATION = get("STEPFUN_NOISE_CANCELLATION", "off")
if _NOISE_CANCELLATION not in ("off", "webrtc", "rnnoise"):
    raise ValueError("STEPFUN_NOISE_CANCELLATION must be 'off', 'webrtc' or 'rnnoise'")

# Pays for DNS, TLS, the WebSocket upgrade and — for TTS — the session
# handshake before the user says anything. Set false to compare startup cost.
_PREWARM = get_bool("STEPFUN_PREWARM", True)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self) -> None:
        if GREETING:
            self.session.generate_reply(instructions=f"Greet the caller: {GREETING}")


def turn_handling() -> TurnHandlingOptions:
    """How the session decides the caller has finished a turn.

    With streaming ASR the server's endpointing is authoritative, which is why
    it is preferred; batch ASR has no such signal, so a turn detector supplies
    one locally.
    """
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
    # detector object.
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


def prewarm(proc: JobProcess) -> None:
    """Load local VAD weights before the first session.

    Weights are ~2MB from disk; loading them per-call adds silence to the
    caller's first sentence.
    """
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
    stt = stepfun.STT(
        language=_LANGUAGE,
        model=_STT_MODEL,
        batch_model=_BATCH_ASR_MODEL,
        use_streaming=_USE_STREAMING_ASR,
        # Server-side endpointing: the model decides when a sentence ended,
        # which beats a fixed silence threshold.
        server_vad=True,
        vad_silence_duration_ms=_VAD_SILENCE_MS,
        sample_rate=_STT_SAMPLE_RATE,
    )
    tts = stepfun.TTS(
        model=_TTS_MODEL,
        voice=_VOICE,
        language=_LANGUAGE,
        # Performance direction. Only honoured by stepaudio-2.5-tts and
        # stepaudio-3-tts; drop it for the other models.
        instruction="warm, professional, unhurried",
    )
    chat = build_llm(reasoning_effort=_REASONING_EFFORT)

    # One line per session in the logs saying which model actually answered.
    logger.info(
        "llm=%s audio input: transport=%s local_vad=%s",
        describe_llm(chat),
        "websocket" if _USE_STREAMING_ASR else "http",
        _USE_LOCAL_VAD,
    )

    if _PREWARM:
        stt.prewarm()
        tts.prewarm()

    session = AgentSession(
        vad=ctx.proc.userdata.get("vad"),
        stt=stt,
        llm=chat,
        tts=tts,
        turn_handling=turn_handling(),
    )

    # Per-session native state: a denoiser is stateful and must never be shared
    # between callers, so it is built here rather than at module scope.
    denoiser = None
    filtered_input = None
    from _noise_cancellation import FilteredAudioInput, NoiseSuppressor, ResamplingAudioInput

    if _NOISE_CANCELLATION != "off":
        denoiser = NoiseSuppressor(method=_NOISE_CANCELLATION)

    async def _close_denoiser() -> None:
        # aclose() on the filter drains the buffer and releases the native
        # handle; the bare suppressor is only reachable when no input existed.
        if filtered_input is not None:
            await filtered_input.aclose()
        elif denoiser is not None:
            denoiser.close()

    if denoiser is not None:
        ctx.add_shutdown_callback(_close_denoiser)

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

        # Install the filter after start(): console and room input are both
        # selected during start() and would overwrite anything set before it.
        if denoiser is not None:
            if session.input.audio is None:
                denoiser.close()  # text-only console mode has no input to filter
            else:
                filtered_input = FilteredAudioInput(session.input.audio, processor=denoiser)
                session.input.audio = filtered_input
        # Downsample to the ASR rate after noise cancellation (NC must run at
        # 48 kHz) so the VAD, turn detector, and ASR all work at one lower rate.
        if session.input.audio is not None:
            session.input.audio = ResamplingAudioInput(
                session.input.audio, sample_rate=_STT_SAMPLE_RATE
            )
    except BaseException:
        # start() failed: the shutdown callback may never run.
        await _close_denoiser()
        raise


# The lk CLI imports this module and discovers this server without executing
# __main__. Register the same callbacks for both launch commands.
server = AgentServer(setup_fnc=prewarm)
server.rtc_session(entrypoint)


if __name__ == "__main__":
    cli.run_app(server)
