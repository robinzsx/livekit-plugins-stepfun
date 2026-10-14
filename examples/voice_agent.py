"""A complete voice agent running entirely on StepFun models.

    STT  stepfun.STT          streaming ASR over WebSocket
    LLM  stepfun.LLM          chat, over the OpenAI-compatible endpoint
    TTS  stepfun.TTS          streaming synthesis over WebSocket

Run it locally against your own microphone and speakers — no LiveKit server,
no LiveKit account, no credentials beyond your StepFun key::

    export STEPFUN_API_KEY=...
    python voice_agent.py console

Console mode runs as an unregistered worker with a simulated job, so it never
asks for LIVEKIT_URL or an API key. That makes it the fastest honest check that
the models work end to end.

For a browser client and real WebRTC, point it at any LiveKit server — self
hosted or Cloud::

    export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
    python voice_agent.py dev

Mixing vendors is a one-line change; see the note on the llm= argument below.
Verified against livekit-agents 1.8.6 and 2.0.0b1.
"""

from __future__ import annotations

import logging
import os

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
)

from livekit.plugins import silero, stepfun

logger = logging.getLogger("stepfun-voice-agent")

# StepFun's chat endpoint is OpenAI-compatible, so stepfun.LLM is a thin
# subclass of the OpenAI plugin's client — it resolves the cluster and
# credentials the same way the STT and TTS do, so a domestic key cannot
# accidentally be pointed at the international host.
#
# step-3.7-flash is the default rather than step-5-preview: the latter is a
# 1M-context reasoning model built for agentic work, and in a voice pipeline its
# thinking time lands entirely before the agent can start speaking. Try it with
# STEPFUN_LLM_MODEL=step-5-preview and compare — but measure time to first
# *speakable* content, not time to first token.
_LLM_MODEL = os.environ.get("STEPFUN_LLM_MODEL", "step-3.7-flash")

# low / medium / high. Documented for step-3.7-flash; lower effort means less
# silence before the reply starts.
_REASONING_EFFORT = os.environ.get("STEPFUN_REASONING_EFFORT", "low")

# Language is worth being explicit about: omitting it selects English
# server-side on both the ASR and the TTS, on both clusters.
_LANGUAGE = os.environ.get("STEPFUN_LANGUAGE", "en")

_VOICE = os.environ.get("STEPFUN_TTS_VOICE", "vibrant-youth")

# STEPFUN_ASR_TRANSPORT=http runs the HTTP + SSE route instead of the
# bidirectional WebSocket. The two endpoints take disjoint model sets, so this
# is the only way to use stepaudio-3-asr-max in a live conversation. The
# framework then wraps the STT in a stt.StreamAdapter, which waits for the local
# VAD to report end of speech before sending the utterance as one request — so
# no interim transcripts, and end-of-speech latency goes up by a round trip.
_USE_STREAMING_ASR = os.environ.get("STEPFUN_ASR_TRANSPORT", "websocket") != "http"

# Streaming ASR model. The endpoint has historically accepted only
# stepaudio-2.5-asr-stream, so override deliberately and check the transcript.
_ASR_MODEL = os.environ.get("STEPFUN_ASR_MODEL", "stepaudio-2.5-asr-stream")
_BATCH_ASR_MODEL = os.environ.get("STEPFUN_BATCH_ASR_MODEL", "stepaudio-3-asr-max")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a concise, friendly voice assistant. Keep answers "
                "conversational and short enough to listen to."
            )
        )


def prewarm(proc: JobProcess) -> None:
    """Move slow one-time work out of the first conversation turn.

    Loading the VAD weights and completing the provider handshakes here is what
    keeps the first reply from being noticeably slower than the rest.
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    stt = stepfun.STT(
        language=_LANGUAGE,
        model=_ASR_MODEL,
        batch_model=_BATCH_ASR_MODEL,
        use_streaming=_USE_STREAMING_ASR,
        # Streaming only: StepFun segments utterances itself and reports speech
        # start/stop, which the plugin translates into the framework's own
        # events. In http mode the local VAD below owns segmentation instead.
        server_vad=True,
    )
    tts = stepfun.TTS(
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
        vad=ctx.proc.userdata["vad"],
        stt=stt,
        # Swap this one line for any other provider to mix vendors, e.g.
        # llm=anthropic.LLM(model="claude-sonnet-5-5"). Nothing else changes:
        # that interoperability is a design requirement, not a coincidence.
        llm=stepfun.LLM(model=_LLM_MODEL, reasoning_effort=_REASONING_EFFORT),
        tts=tts,
        # Let the transcript drive turn boundaries, with the local VAD still
        # available for fast interruption detection.
        turn_detection="stt",
    )

    # Where the latency numbers come from: ASR first partial and final, LLM
    # time to first token, TTS time to first byte.
    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: metrics.MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def _log_usage() -> None:
        logger.info("usage summary: %s", usage.get_summary())

    ctx.add_shutdown_callback(_log_usage)

    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
