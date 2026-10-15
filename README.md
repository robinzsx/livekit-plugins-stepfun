# LiveKit Plugins StepFun

Agent Framework plugin for [StepFun](https://www.stepfun.com/) (阶跃星辰) speech
services. Supports streaming speech-to-text and low-latency streaming
text-to-speech.

The current regression suite, `ruff` and strict `mypy` are verified against
`livekit-agents` **1.8.6** and **1.8.7**, with no version branching in the plugin.
Earlier versions were also checked against **2.0.0b1**; the current changes
have not been retested on that beta.

| Plugin | `livekit-agents` |
| --- | --- |
| 0.1.x | >=1.8.6; current checks cover 1.8.6 and 1.8.7 |

## Installation

```bash
pip install livekit-plugins-stepfun
```

## Configuration

The plugin itself reads only `os.environ` — set `STEPFUN_API_KEY`, or pass
`api_key=` to the constructor. It never loads a `.env` file, because a library
quietly reading files from the working directory is surprising behaviour.

The **examples** are applications, so they do load one. Copy the template and
edit it:

```bash
cp .env.example .env
```

`.env.example` documents every setting: credentials, cluster, the four model
choices, voice, language and ASR transport, plus the LiveKit variables needed
for `voice_agent.py dev`. Both `.env` and `.env.local` are gitignored;
`.env.local` wins where they overlap, and an exported variable beats both, so
one-off overrides still work:

```bash
STEPFUN_LANGUAGE=zh lk agent console examples/voice_agent.py
```

This needs the `examples` extra:

```bash
uv pip install -e '.[examples]'
```

```bash
export STEPFUN_API_KEY=<your-key>
```

StepFun runs two clusters. The plugin defaults to the **international** cluster
(`api.stepfun.ai`), because LiveKit Cloud agent workers run mostly in North
America and Europe. Override with `cluster="domestic"` or `STEPFUN_CLUSTER`.

> Credentials are **not** interchangeable between clusters: a `api.stepfun.com`
> key returns 401 against `api.stepfun.ai`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import silero, stepfun

session = AgentSession(
    vad=silero.VAD.load(),
    stt=stepfun.STT(language="zh"),
    llm=stepfun.LLM(model="step-3.7-flash", reasoning_effort="low"),
    tts=stepfun.TTS(voice="vibrant-youth", instruction="warm, professional tone"),
)
```

Any component can be swapped for another provider — `stepfun.STT` with
`openai.LLM`, or `deepgram.STT` with `stepfun.TTS`. All three follow the
standard LiveKit contracts, so they also work inside `stt.FallbackAdapter` /
`tts.FallbackAdapter`.

`stepfun.LLM` needs the `llm` extra, since StepFun's chat endpoint is
OpenAI-compatible and the wrapper builds on that plugin's client:

```bash
pip install 'livekit-plugins-stepfun[llm]'
```

The STT and TTS components do not need it. They are importable on their own, and
`stepfun.LLM` is resolved lazily so an ASR/TTS-only install keeps working.

### Two ASR transports

The two endpoints accept **disjoint model sets**, so the choice of transport is
also a choice of model:

| | transport | models | interim transcripts |
| --- | --- | --- | --- |
| `use_streaming=True` (default) | bidirectional WebSocket | `stepaudio-2.5-asr-stream` | yes |
| `use_streaming=False` | HTTP + SSE | `stepaudio-3-asr-max`, `stepaudio-2.5-asr` | no |

`stepaudio-3-asr-max` is documented for the batch route only, so running it in a
live conversation means turning native streaming off:

```python
session = AgentSession(
    vad=silero.VAD.load(),                       # required in this mode
    stt=stepfun.STT(use_streaming=False, batch_model="stepaudio-3-asr-max"),
    ...
)
```

The framework then wraps the STT in a `stt.StreamAdapter`, which buffers each
utterance until the **local** VAD reports end of speech and sends it as one
request. An `AgentSession` without a `vad` raises rather than silently
degrading.

Be clear-eyed about the cost: nothing is recognized until the user stops
talking, and `StreamAdapter` declares `interim_results=False`, so there is no
partial transcript to drive preemptive generation. Expect end-of-speech latency
to go up by roughly a full request round trip. Prefer the default unless the
batch model's accuracy is worth that trade.

`recognize()` uses the batch route in **both** modes — that is what
`offline_recognize=True` promises, and it is the route `FallbackAdapter` falls
back to.

### Language defaults to English

Omitting `language` selects **English** server-side, on both the ASR and the TTS
side. It is not inferred from the cluster, so a Chinese-first deployment must
pass `language="zh"` explicitly.

Supported: `zh`, `en`, plus `ja`, `ko`, `fr`, `es` in preview.

### Models

| Component | Default | Also available |
| --- | --- | --- |
| `STT` streaming | `stepaudio-2.5-asr-stream` | `step-asr-1.1-stream` (domestic, deprecated) |
| `STT` batch | `stepaudio-3-asr-max` | `stepaudio-2.5-asr`, `stepaudio-2-asr-pro` (domestic) |
| `LLM` | `step-3.7-flash` | `step-5-preview`, `step-3.5-flash` |
| `TTS` | `stepaudio-3-tts` | `stepaudio-2.5-tts`; `step-tts-2`, `step-tts-mini` (domestic) |

**There is no streaming StepAudio 3 ASR.**

These `Literal` types are for editor completion, not validation: any model
string is accepted and forwarded to the server verbatim, which is where an
unknown one fails.

`stepaudio-2.5-tts` is *not* the TTS default on purpose: StepFun documents its
WebSocket streaming quality as noticeably worse than its HTTP output, which
makes it a poor fit for the streaming path a voice agent lives on. It remains a
good choice via `session.say()`, which uses the HTTP path.

`instruction` (global emotional direction) is only accepted by
`stepaudio-2.5-tts` (≤200 chars) and `stepaudio-3-tts` (≤500 chars). Passing it
to another model is rejected by the server, so the plugin drops it with a
warning instead.

### Reasoning models

`step-5-preview` and `step-3.7-flash` return their thinking in a separate
`reasoning` field. The framework's chat parser reads only `delta.content`,
`delta.tool_calls` and `delta.extra_content`, so **reasoning never reaches the
TTS** — the agent does not narrate its own thinking. The trade-off is that the
reasoning trace is not available for logging either.

What reasoning does cost is latency: it happens *before* the first speakable
token, so the user hears silence for its whole duration. `step-3.7-flash` is the
default rather than `step-5-preview` for that reason — the latter is a
1M-context model built for agentic and software-engineering work, with
`max_tokens` defaulting to unlimited.

```python
llm=stepfun.LLM(model="step-3.7-flash", reasoning_effort="low")
```

`reasoning_effort` accepts `low`, `medium`, `high`. Only `step-3.7-flash` is
documented as supporting all three, so the plugin warns rather than silently
assuming it applies to other models. When benchmarking, measure time to first
*speakable* content, not time to first token.

### Voices

A voice ID the cluster does not know is rejected with `400 voice_id_invalid`, so
discover them rather than guessing:

```python
tts = stepfun.TTS()
print(await tts.list_voices())
```

The international catalogue (7 voices) is a strict subset of the domestic one
(39) — same IDs, fewer of them — so no per-cluster translation is needed. The
default, `vibrant-youth`, is an English voice present on both.

### Word-level timestamps

```python
stt = stepfun.STT(enable_timestamp_align=True)   # aligned_transcript="word"
tts = stepfun.TTS(timestamp=True)                # aligned_transcript=True
```

ASR alignment enables LiveKit's adaptive interruption. Note that
`stt.FallbackAdapter` omits `aligned_transcript` when rebuilding its own
capabilities ([livekit/agents#5436](https://github.com/livekit/agents/issues/5436)),
so wrapping this STT in one silently drops word alignment.

### Turn detection

The example selects turn handling **before** constructing `AgentSession`:

| ASR transport | End of turn | Local VAD |
| --- | --- | --- |
| `websocket` | StepFun server VAD, `turn_detection="stt"` | Enabled for faster interruptions; optional |
| `http` | Local audio turn detector (`v1-mini`) | Required for `StreamAdapter` segmentation |

Streaming skips the local turn detector. Local VAD only improves interruption
responsiveness; it does not gate audio sent to StepFun or decide when the
streaming turn ends. Set `STEPFUN_LOCAL_VAD=false` to remove it: turn completion
still uses StepFun's speech events, but interruptions must wait for an interim
or final transcript. Adaptive audio interruption detection needs local VAD;
the example explicitly selects basic interruption handling to avoid requiring
LiveKit Cloud inference for a local console session. See
[LiveKit's turn handling guide](https://docs.livekit.io/agents/logic/turns/).

Batch mode always uses local VAD, even with `STEPFUN_LOCAL_VAD=false`. VAD
segments the audio for batch STT while `inference.TurnDetector(version="v1-mini")`
assesses turn completion directly from audio. The session waits for the batch
transcript before replying. Pinning `v1-mini` keeps turn detection on the CPU
even in dev mode; no cloud turn-detector credentials are required. See
[LiveKit's audio turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/).
Install the example dependencies before launching batch mode:

```bash
uv sync --extra examples --extra dev
STEPFUN_ASR_TRANSPORT=http uv run lk agent console examples/voice_agent.py
```

Use `STEPFUN_ASR_TRANSPORT=websocket` for streaming. Do not use `"manual"`
turn detection for this example: that mode requires explicit application calls
to commit user turns.

With local VAD enabled, the example stops a reply after at least
`STEPFUN_MIN_INTERRUPTION_DURATION=0.2` seconds of detected speech, without
waiting for a transcript or an end-of-turn prediction. This is a detection
threshold, not a guarantee of total audible stopping latency. Increase it
toward `0.5` if brief sounds interrupt too often.

The example disables resumable false interruptions: detected speech cancels
the reply rather than pausing and potentially resuming it. In the installed
1.8.6/1.8.7 SDKs, the TCP console advertises pause support but does not send a
pause command; cancellation sends the actual playback-clear command instead.
The SDK's initial AEC warmup still applies to the first speaking turn.

The figures below are historical measurements from earlier live-service runs,
not latency guarantees or results of the current review.

| | `server_vad=True` (default) | `server_vad=False` |
| --- | --- | --- |
| Use for | `AgentSession` | driving `stream()` yourself |
| Who segments | StepFun, on hearing silence | you, via `flush()` |
| `START_OF_SPEECH` / `END_OF_SPEECH` | yes | no |
| Final latency | 1372ms after the last word | 438ms after `flush()` |

**`AgentSession` needs the default.** The framework never calls `flush()` on a
natively streaming STT — it flushes its own VAD stream instead — so with
`server_vad=False` nothing would ever commit and no final would arrive. The
default mode also supplies `START_OF_SPEECH` / `END_OF_SPEECH`, translated from
the server's own VAD events. The 1372ms is mostly the server's 800ms
`silence_duration_ms` window, which `vad_silence_duration_ms` can shorten.

**Driving the stream yourself needs `server_vad=False`**, so that `flush()`
commits each utterance explicitly.

One property to be aware of in the default mode: server VAD needs **audio
containing silence**, not an absence of audio. A caller that stops sending the
moment the speaker stops talking gives the detector nothing to work with and
never sees a final. A real microphone keeps streaming, so this only bites
synthetic feeds — pad the tail with silence, as `examples/smoke_test.py` does
with `--trailing-silence`.

```python
# Inside an AgentSession — the default is correct
stt=stepfun.STT()

# Driving the stream yourself
stt = stepfun.STT(server_vad=False)
stream = stt.stream()
stream.push_frame(frame)
stream.end_input()   # flush() commits; the final follows
```

### Optional local noise suppression

The voice-agent example supports two offline CPU backends:

- `webrtc`: uses the installed LiveKit SDK's `AudioProcessingModule`; no extra
  package, cloud account or runtime model download is required.
- `rnnoise`: uses [RNNoise](https://github.com/xiph/rnnoise), through the optional
  [pyrnnoise](https://github.com/pengzhendong/pyrnnoise) dependency.

For WebRTC suppression with an already installed example environment:

```bash
STEPFUN_NOISE_CANCELLATION=webrtc uv run --no-sync lk agent console examples/voice_agent.py
```

For RNNoise:

```bash
uv sync --extra examples --extra noise-cancellation
STEPFUN_NOISE_CANCELLATION=rnnoise uv run lk agent console examples/voice_agent.py
```

Set `STEPFUN_NOISE_CANCELLATION=webrtc` or `rnnoise` in `.env.local` to keep it
selected. The default is `off`. Both backends reduce background noise; this
integration does not configure acoustic echo cancellation or speaker isolation.
An explicit `rnnoise` setting fails with installation instructions if its
optional dependency or native library is unavailable. The native wheel must
support your OS and CPU; consult
[pyrnnoise's available wheels](https://pypi.org/project/pyrnnoise/#files).

The same filter works with `voice_agent.py dev` and both ASR transports. It
wraps the audio input selected by the SDK, including the console input that
bypasses room-level noise cancellation. STT, local VAD and the local audio turn
detector all receive the cleaned input. Denoiser state belongs to one session.

Input is mixed to mono, resampled to 48 kHz and processed in 10 ms blocks off
the event loop. This adds framing and processing latency. Silence is never
gated or dropped, so StepFun server VAD can still finalize streaming turns.
Incomplete final blocks are padded for processing and trimmed back to their
original duration. Local VAD remains required in batch mode.

The offline suite tests framing, resampling, cancellation-safe cleanup and
audio delivery through both STT pipelines with a scripted denoiser. It also
checks actual WebRTC noise suppression on synthetic noise. The native RNNoise
test runs when `pyrnnoise` is installed:

```bash
uv sync --extra examples --extra dev --extra noise-cancellation
uv run pytest -q tests/test_noise_cancellation.py
```

## Design notes

A few deliberate decisions worth knowing before you change things:

**The plugin does not do its own VAD.** The speech events above are translated
from genuine server events; nothing is inferred from client-side energy gating
or silence splitting. Turn detection and interruption are the framework's job,
and a plugin that second-guesses them will fight `silero.VAD` and the turn
detector in mixed setups.

**The plugin does not retry.** Retries are configured exclusively through
`APIConnectOptions.max_retry`. Adding client-side retries on top produces
multiplicative request amplification where only the framework's attempts are
visible in logs.

**`retryable` is set deliberately per status code.** Auth failures are *not*
retryable. Marking them retryable makes a customer's fallback chain hammer a
provider that can never succeed instead of failing over. One subtlety: `429`
means either ordinary rate limiting (retryable) or a Credit ceiling
(`insufficient_credit`, `project_credit_limit_exceeded`,
`member_project_credit_limit_exceeded`), which cannot clear until the limit is
raised or the quota resets — those are treated as terminal.

**One segment per `SynthesizeStream`.** The framework deprecated handling
several segments in one stream, and StepFun's session ends at `tts.text.done`
with the server releasing the socket, so a segment maps exactly onto a session.
Mid-stream boundaries use `tts.text.flush`, which drains buffered audio without
ending anything. Connection reuse is handled by `utils.ConnectionPool` with
`prewarm()`.

**The one-shot path requests WAV, the streaming path raw PCM.** WAV carries its
own sample rate and channel count, so `synthesize()` cannot be mis-tuned into
playing at the wrong pitch. The streaming path has no header, so its declared
`sample_rate` is load-bearing.

**Inline parenthesised directives are neutralized by default.** On
`stepaudio-2.5-tts` / `stepaudio-3-tts`, text inside `()` or `（）` is read as
performance direction and is not spoken. In a LiveKit pipeline that text comes
from the LLM, so a stray "(as you mentioned)" would be silently swallowed, and a
user who can steer the LLM's wording could reach the voice settings indirectly.
StepFun documents no escape sequence, so the plugin removes the delimiters and
keeps their contents as ordinary words. Set `enable_inline_instruction=True` to
pass them through.

**Option snapshots are taken in the stream constructor**, never read lazily in
`_run()`, so `update_options()` cannot race with an in-flight turn.
`update_options()` also invalidates the connection pool, since a pooled socket
is bound to the model and configuration it was opened with.

## Known limitations

| Area | Limitation |
| --- | --- |
| **Barge-in billing** | StepFun's streaming TTS protocol has **no cancel event**. On interruption the plugin can only close the socket, so the server may finish generating — and bill for — audio nobody hears. This is the one significant gap left on the provider side. |
| Browser clients | Auth is a WebSocket header, which browsers cannot set on a native WebSocket. Fine for a server-side Python agent; an `agents-js` browser setup would need query-parameter or ephemeral-token auth. |
| Diarization | Not supported by the streaming endpoint. |
| Keyterms | The streaming endpoint takes free-text `prompt` hints rather than a keyterm list, so `STTCapabilities.keyterms` is `False`. Use `prompt=`. Batch recognition accepts `hotwords=` (domestic cluster only). |
| Nonverbal markup | `tts.TTS.Markup` is not declared. `_provider_key()` only indexes the framework's own `_provider_format` tables, so a plugin-side override with no upstream `stepfun` entry declares nothing. Enabling it requires a change to `livekit-agents` itself. |
| ASR idle timeout | The endpoint exposes `max_idle_timeout` / `pong_timeout` error codes but documents no interval, so pooled connections are retired conservatively after 15s. |

## Development

```bash
uv venv && uv pip install -e '.[dev]'
uv run ruff check livekit tests
uv run mypy -p livekit.plugins.stepfun
uv run pytest
```

The test suite needs no credentials and no network: it drives both streaming
surfaces against a scripted fake WebSocket. That matters because CI jobs holding
provider secrets generally do not run on pull requests from forks.

To verify against the live service — all four code paths, plus the pitch check
that real audio settles:

```bash
export STEPFUN_API_KEY=...
uv run python examples/smoke_test.py                              # English
uv run python examples/smoke_test.py --language zh                # Chinese
uv run python examples/smoke_test.py --cluster domestic --language zh

# Compare chat models in one run
uv run python examples/smoke_test.py --llm-model step-3.7-flash,step-5-preview
```

Every model is selectable: `--stt-model`, `--batch-model`, `--tts-model`,
`--voice`, `--llm-model` (comma-separated, `none` to skip) and
`--reasoning-effort`. A model that fails is reported and the run continues, so
one rejected ID does not hide the rest.

It synthesizes a phrase, transcribes that same audio back, and writes both the
one-shot and streaming audio to `./smoke_out/` — so it needs no microphone and
no LiveKit server. Listen to the two files: a sample-rate or channel-count
mistake produces no error and only sounds like the wrong pitch.

## License

Apache-2.0
