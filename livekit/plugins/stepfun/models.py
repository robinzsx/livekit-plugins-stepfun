from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
#
# Both clusters speak an identical wire protocol over identical paths; only the
# model and voice catalogues differ (see the per-capability notes below).
# Credentials are NOT interchangeable between them.

Cluster = Literal["international", "domestic"]

# Default to the international cluster: LiveKit Cloud agent workers run mostly
# in North America and Europe, so defaulting to the domestic cluster would give
# overseas developers a poor first impression on latency.
DEFAULT_CLUSTER: Cluster = "international"

CLUSTER_WS_BASE_URLS: dict[Cluster, str] = {
    "international": "wss://api.stepfun.ai",
    "domestic": "wss://api.stepfun.com",
}

CLUSTER_HTTP_BASE_URLS: dict[Cluster, str] = {
    "international": "https://api.stepfun.ai",
    "domestic": "https://api.stepfun.com",
}

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

# Shared by streaming ASR, batch ASR and TTS. Everything past zh/en is marked
# preview by the provider.
#
# Note the default: omitting the language selects *English*, on both the ASR and
# the TTS side. Callers working in Chinese have to pass language="zh"
# explicitly — it is not inferred from the cluster.
Language = Literal["zh", "en", "ja", "ko", "fr", "es"]

PREVIEW_LANGUAGES = frozenset({"ja", "ko", "fr", "es"})

# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

#: Streaming ASR, over the bidirectional WebSocket.
STTModels = Literal[
    # NOTE: there is deliberately no stepaudio-3-asr-stream. It was tried
    # against the live endpoint on 2026-10-14 and rejected outright:
    #   {"code": "invalid_value", "param": "model",
    #    "message": "unsupported model: stepaudio-3-asr-stream"}
    # StepAudio 3 ASR is submitted-audio only — use batch_model with
    # use_streaming=False. Listing it here would advertise it in editor
    # completion as if it worked.
    "stepaudio-2.5-asr-stream",
    # Domestic cluster only, and documented as being phased out. The
    # international endpoint accepts stepaudio-2.5-asr-stream exclusively.
    "step-asr-1.1-stream",
]

#: Models whose transcription deltas carry an *incremental* fragment rather
#: than the cumulative transcript so far.
#:
#: Expressed as a denylist on purpose. Cumulative is the modern behaviour —
#: `stepaudio-2.5-asr-stream` and later send the whole transcript on every
#: delta, corrections included — and `step-asr-1.1-stream` is the legacy
#: exception. A denylist means a new or custom model string gets the cumulative
#: treatment by default, which is the safer wrong guess:
#:
#:   * treating cumulative deltas as incremental appends the full transcript to
#:     itself over and over, compounding into garbage;
#:   * treating incremental deltas as cumulative only shows the latest
#:     fragment, and the final still arrives intact on the `completed` event.
INCREMENTAL_DELTA_STT_MODELS = frozenset({"step-asr-1.1-stream"})

#: Models that support `enable_timestamp_align`, which populates the per-word
#: `words` array on deltas and on the completed event.
#:
#: An allowlist, unlike the above: declaring `aligned_transcript="word"` for a
#: model that never emits `words` would silently break adaptive interruption,
#: and over-declaring a capability is worse than missing one.
TIMESTAMP_ALIGN_STT_MODELS = frozenset({"stepaudio-2.5-asr-stream"})

#: Batch ASR, over HTTP + SSE. Used by `STT._recognize_impl` and by
#: StreamAdapter-based setups.
STTBatchModels = Literal[
    "stepaudio-3-asr-max",
    "stepaudio-2.5-asr",
    # Domestic cluster only.
    "stepaudio-2-asr-pro",
]

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

TTSModels = Literal[
    "stepaudio-3-tts",
    "stepaudio-2.5-tts",
    # Domestic cluster only.
    "step-tts-2",
    "step-tts-mini",
    # Deprecated by the provider; existing callers keep working.
    "step-tts-vivid",
]

#: Models that accept the `instruction` parameter (natural-language direction
#: for overall emotion and persona) and that parse parenthesised inline
#: directives inside the input text. The other models reject `instruction`
#: outright, so it must not be sent to them.
INSTRUCTION_TTS_MODELS = frozenset({"stepaudio-2.5-tts", "stepaudio-3-tts"})

#: Per-model cap on `instruction` length, in characters.
INSTRUCTION_MAX_CHARS: dict[str, int] = {
    "stepaudio-2.5-tts": 200,
    "stepaudio-3-tts": 500,
}

#: Models that accept `voice_label` (language/emotion/style presets). Mutually
#: exclusive with the `instruction` models above, which reject it.
VOICE_LABEL_TTS_MODELS = frozenset({"step-tts-2", "step-tts-mini", "step-tts-vivid"})

#: Maximum characters per `tts.text.delta` frame, and per one-shot HTTP request.
TTS_MAX_INPUT_CHARS = 1000

# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------
#
# A voice ID that the cluster does not know is rejected outright with
# `400 voice_id_invalid`, so these are worth keeping accurate. Query
# SYSTEM_VOICES_PATH (or TTS.list_voices()) for the authoritative list, since
# cloned voices are account-specific and cannot appear here.
#
# The two clusters do *not* use different naming schemes: the international
# catalogue is a strict subset of the domestic one, so no ID translation is
# needed in either direction. Only availability differs.

TTSVoices = Literal[
    # -- Chinese, male
    "cixingnansheng",
    "boyinnansheng",
    "zhengpaiqingnian",
    "zixinnansheng",
    "yuanqinansheng",
    "ruyananshi",
    "shuangkuainansheng",
    "qingniandaxuesheng",
    "shenchennanyin",
    "wenrougongzi",
    "wenrounansheng",
    # -- Chinese, female
    "elegantgentle-female",
    "linjiameimei",
    "linjiajiejie",
    "ruanmengnvsheng",
    "shuangkuaijiejie",
    "wenjingxuejie",
    "qingchunshaonv",
    "livelybreezy-female",
    "ganliannvsheng",
    "tianmeinvsheng",
    "youyanvsheng",
    "jingdiannvsheng",
    "yuanqishaonv",
    "zhixingjiejie",
    "qinqienvsheng",
    "wenroushunv",
    "qinhenvsheng",
    "huolinvsheng",
    "wenrounvsheng",
    "jilingshaonv",
    "lengyanyujie",
    # -- English
    "vibrant-youth",
    "soft-spoken-gentleman",
    "magnetic-voiced-male",
    "lively-girl",
    "energeticconfident-female",
    "yingwennansheng",
    "yingwennvsheng",
]

#: The subset published for the international cluster. Used only to produce a
#: clearer error than `400 voice_id_invalid` when a domestic-only voice is
#: requested internationally.
INTERNATIONAL_TTS_VOICES = frozenset(
    {
        "elegantgentle-female",
        "lively-girl",
        "livelybreezy-female",
        "magnetic-voiced-male",
        "soft-spoken-gentleman",
        "vibrant-youth",
        "zixinnansheng",
    }
)

#: English-language voice, available on both clusters. English is also what the
#: service falls back to when no language is given, so the two defaults agree.
DEFAULT_TTS_VOICE = "vibrant-youth"

#: `model` value the system-voices endpoint expects, per cluster. It is the one
#: place the two clusters genuinely disagree: each accepts exactly one value and
#: 400s on anything else.
SYSTEM_VOICES_MODEL: dict[Cluster, str] = {
    "international": "stepaudio-2.5-tts",
    "domestic": "step-tts-2",
}

# ---------------------------------------------------------------------------
# Audio formats
# ---------------------------------------------------------------------------

#: Sample rate sent to the ASR endpoints. The base RecognizeStream resamples
#: the 48kHz WebRTC audio down to this for us.
DEFAULT_STT_SAMPLE_RATE = 16000

#: Sample rate requested from, and declared for, the TTS endpoints. This is the
#: server's own default; any value in TTS_SAMPLE_RATES may be requested instead.
#:
#: The declared rate must match what the server actually emits. A mismatch
#: raises no error — it just plays back at the wrong pitch, which is a
#: notoriously hard bug to track down. The one-shot HTTP path sidesteps this
#: entirely by requesting WAV and letting AudioEmitter read the real rate out of
#: the header; the streaming path has no header, so this value is load-bearing.
DEFAULT_TTS_SAMPLE_RATE = 24000

TTS_SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 48000})

#: Mono. Not documented for the streaming TTS endpoint, which specifies no
#: channel parameter; mono is near-certain for a single TTS voice, and the
#: WAV header on the one-shot path confirms it at runtime.
NUM_CHANNELS = 1

#: Frame size for uploaded ASR audio. 50ms @ 16kHz mono = 800 samples =
#: 1600 bytes. Smaller frames (10ms) multiply WebSocket overhead and
#: server-side work; larger ones (200ms) push the first partial transcript out
#: by a comparable amount. Deepgram also settles on 50ms.
STT_FRAME_MS = 50

#: Raw PCM16 little-endian, the codec string the ASR `format` block expects.
PCM_CODEC = "pcm_s16le"

PCM_BITS = 16

# ---------------------------------------------------------------------------
# Server-side VAD defaults
# ---------------------------------------------------------------------------
#
# When turn detection is enabled the server segments utterances itself and
# emits speech_started / speech_stopped. When it is disabled, nothing is
# segmented automatically and the client must commit each utterance.

DEFAULT_VAD_SILENCE_DURATION_MS = 800

#: Higher is stricter.
DEFAULT_VAD_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Idle timeouts
# ---------------------------------------------------------------------------

#: The streaming TTS connection is dropped after roughly 60s of inactivity, so
#: pooled connections are retired well before that.
TTS_MAX_SESSION_DURATION = 30.0

#: The streaming ASR endpoint documents no idle timeout value, only a
#: `max_idle_timeout` error code proving one exists. Kept deliberately
#: conservative until the real number is known.
STT_MAX_SESSION_DURATION = 15.0

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

#: Bidirectional streaming ASR. Takes no query parameters: the model, audio
#: format and turn detection are all configured by a `session.update` frame
#: sent after connecting.
STT_STREAM_PATH = "/v1/realtime/asr/stream"

#: Batch ASR: one-shot upload, incremental transcripts pushed back over SSE.
#: The only batch route available on both clusters — the file-upload routes
#: (/v1/audio/asr, /v1/audio/transcriptions) are documented for the domestic
#: cluster only.
STT_RECOGNIZE_PATH = "/v1/audio/asr/sse"

#: Streaming TTS. Takes the model as a query parameter, unlike ASR.
TTS_STREAM_PATH = "/v1/realtime/audio"

#: One-shot TTS over HTTP.
TTS_SYNTHESIZE_PATH = "/v1/audio/speech"

#: Official voice catalogue; see SYSTEM_VOICES_MODEL for the required `model`.
SYSTEM_VOICES_PATH = "/v1/audio/system_voices"

#: Suffix appended to the HTTP base URL for the OpenAI-compatible client, which
#: expects the API version in the base URL rather than in each request path.
OPENAI_COMPAT_SUFFIX = "/v1"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
#
# These reach LiveKit through the OpenAI-compatible chat completions endpoint,
# so no provider-specific protocol work is involved — see llm.py.

LLMModels = Literal[
    # Reasoning model. 1M context, up to 64k output, text/image/video in.
    # Built for software engineering and multi-step agent work; its thinking
    # time lands entirely before the first speakable token, so treat it as a
    # quality-over-latency choice for voice.
    "step-5-preview",
    # Reasoning model with three documented effort levels. The provider's own
    # voice-candidate suggestion, and the default here.
    "step-3.7-flash",
    # Text only, 256K context, sparse MoE.
    "step-3.5-flash",
]

#: Default. Not step-5-preview: in a voice pipeline the user hears silence for
#: the whole reasoning phase, and this model exposes effort control to keep that
#: short. Switch deliberately, having measured both.
DEFAULT_LLM_MODEL = "step-3.7-flash"

ReasoningEffort = Literal["low", "medium", "high"]

#: Models documented as accepting `reasoning_effort`. The provider's reasoning
#: guide names only step-3.7-flash explicitly and does not state either way for
#: the others, so this is the confirmed set rather than the complete one.
REASONING_EFFORT_LLM_MODELS = frozenset({"step-3.7-flash"})

#: Models that return a `reasoning` field alongside `content`.
#:
#: The framework's chat parser reads only `delta.content`, `delta.tool_calls`
#: and `delta.extra_content`, so `reasoning` is dropped before it can reach the
#: TTS. That is the behaviour you want in a voice agent — the agent never
#: narrates its own thinking — but it also means the reasoning trace is not
#: available for logging.
REASONING_LLM_MODELS = frozenset({"step-5-preview", "step-3.7-flash"})
