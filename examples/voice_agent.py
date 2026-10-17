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
# stepfun | deepseek | mix. Model IDs are read by _llm_providers, which owns
# STEPFUN_LLM_MODEL and DEEPSEEK_MODEL.
_LLM_PROVIDER = (get("LLM_PROVIDER", "stepfun") or "stepfun").strip().lower()
if _LLM_PROVIDER not in ("stepfun", "deepseek", "mix"):
    raise ValueError("LLM_PROVIDER must be 'stepfun', 'deepseek' or 'mix'")

# low / medium / high. Documented for step-3.7-flash; lower effort means less
# silence before the reply starts.
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

_INTERRUPTION_DURATION = float(get("STEPFUN_MIN_INTERRUPTION_DURATION", "0.2"))
if not 0.05 <= _INTERRUPTION_DURATION <= 5.0:
    raise ValueError("STEPFUN_MIN_INTERRUPTION_DURATION must be between 0.05 and 5 seconds")

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


def prewarm(proc: JobProcess) -> None:
    """Load local VAD weights before the first session.

    Weights are ~2MB from disk; loading them per-call adds silence to the
    caller's first sentence.
    """
    if _USE_LOCAL_VAD:
        proc.userdata["vad"] = silero.VAD.load()


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
    if _NOISE_CANCELLATION != "off":
        from _noise_cancellation import FilteredAudioInput, NoiseSuppressor

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
