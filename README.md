# StepFun plugin for LiveKit Agents

Streaming speech recognition, speech synthesis, and chat models from
[StepFun](https://platform.stepfun.ai/docs) (阶跃星辰), using the standard
LiveKit `STT`, `TTS`, and `LLM` interfaces. Python 3.10+ and LiveKit Agents
1.8.6+ are required.

## Run the voice agent

Install from this checkout to use this implementation:

```bash
uv sync --extra examples --extra dev
cp .env.example .env
```

Set `STEPFUN_API_KEY` in `.env`, then start a local microphone conversation:

```bash
uv run lk agent console examples/voice_agent.py
```

Install the [LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup/) if `lk`
is unavailable. Local console mode requires a StepFun key and internet access;
it does not require a LiveKit server or LiveKit credentials. To launch without
`lk`, use `uv run python examples/voice_agent.py console` (the SDK's legacy
console entry point).

For a room session, also set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and
`LIVEKIT_API_SECRET`, then connect a LiveKit client to the room:

```bash
uv run python examples/voice_agent.py dev
```

The examples load `.env.local` before `.env` in each searched directory,
starting with the working directory and then the repository root. Exported
variables take precedence over files. The plugin itself reads environment
variables only.

## Use the plugin in your application

Install with pip from the repository root if you do not use uv:

```bash
pip install -e '.[llm]'
```

```python
from livekit.agents import AgentSession
from livekit.plugins import stepfun

session = AgentSession(
    stt=stepfun.STT(language="zh"),
    llm=stepfun.LLM(reasoning_effort="low"),
    tts=stepfun.TTS(voice="vibrant-youth", language="zh"),
)
```

STT and TTS work without the `llm` extra. The chat wrapper uses
`livekit-plugins-openai` and accepts its additional options through `**kwargs`.
You can combine StepFun components with other providers and LiveKit fallback
adapters.

All components accept `api_key=`, `cluster=`, and `base_url=`. STT and TTS
use `base_url` for WebSockets and `http_base_url=` for HTTP. The chat client
appends `/v1` to its base URL when absent.

| Setting | Default | Alternative |
| --- | --- | --- |
| `STEPFUN_CLUSTER` / `cluster` | `global` (`api.stepfun.ai`) | `china` (`api.stepfun.com`, mainland China) |
| STT streaming model | `stepaudio-2.5-asr-stream` | Custom model string |
| STT batch model | `stepaudio-3-asr-max` | `stepaudio-2.5-asr` |
| TTS model | `stepaudio-3-tts` | `stepaudio-2.5-tts` |
| LLM model | `step-3.7-flash` | `step-5-preview`, `step-3.5-flash` |
| Language | Server default, English | `zh`, `en`; `ja`, `ko`, `fr`, `es` in preview |
| TTS voice | `vibrant-youth` | Discover with `await tts.list_voices()` |

Keys belong to their issuing cluster. Use `cluster="china"` or set
`STEPFUN_CLUSTER=china` for a mainland-China key. Model and voice availability
can differ between clusters; custom model and cloned voice IDs are accepted
and validated by the service.

## Speech recognition

The WebSocket route streams interim and final transcripts. `recognize()`
always uses the HTTP + SSE batch route, including on a streaming STT instance.
The two routes use different model IDs.

```python
stt = stepfun.STT(
    language="zh",
    prompt="Product names: StepFun, LiveKit",
    enable_timestamp_align=True,
    vad_silence_duration_ms=800,
    vad_threshold=0.6,
)
```

Timestamp alignment is advertised only for known supported streaming models.
Batch recognition accepts `hotwords=[...]` on the mainland-China cluster.

To use a batch model for conversation, supply local VAD so LiveKit can segment
utterances and wrap the plugin in `stt.StreamAdapter`:

```python
from livekit.plugins import silero

session = AgentSession(
    vad=silero.VAD.load(),
    stt=stepfun.STT(use_streaming=False, batch_model="stepaudio-3-asr-max"),
    llm=stepfun.LLM(reasoning_effort="low"),
    tts=stepfun.TTS(),
)
```

Batch mode produces no interim transcripts and sends a request after each
utterance ends. For the bundled agent, select it with:

```bash
STEPFUN_ASR_TRANSPORT=http uv run lk agent console examples/voice_agent.py
```

The example uses StepFun server VAD for streaming turn completion. Local
Silero VAD improves interruptions and can be disabled with
`STEPFUN_LOCAL_VAD=false`. Batch mode requires local VAD and uses the local
CPU turn detector `v1-mini`.

When feeding a streaming STT directly, use `server_vad=False` and `flush()` or
`end_input()` to commit audio. With server VAD enabled, keep sending silence
after speech so the server can detect the utterance boundary. After input
ends, the plugin allows up to three seconds for remaining transcripts.

## Speech synthesis

```python
tts = stepfun.TTS(
    model="stepaudio-3-tts",
    voice="vibrant-youth",
    language="en",
    instruction="warm, professional tone",
    speed=1.0,
    volume=1.0,
    timestamp=True,
)
```

`stream()` uses WebSockets with PCM16 audio; `synthesize()` uses HTTP with
WAV audio. Supported sample rates are 8000, 16000, 22050, 24000 (default),
and 48000 Hz. Speed accepts 0.5–2.0 and volume accepts 0.1–2.0.
`update_options()` validates changes before applying them to future requests.
Active streams keep their original options.

`instruction` supports up to 200 characters on `stepaudio-2.5-tts` and 500 on
`stepaudio-3-tts`. Unsupported models drop it with a warning. Parentheses in
LLM output are removed by default to keep their contents spoken rather than
interpreted as performance directions; set `enable_inline_instruction=True`
to pass directives through. Streaming `timestamp=True` requests aligned subtitles.

Each TTS stream handles one segment. `flush()` sends `tts.text.flush`, while
`end_input()` finishes the session. Interrupting a stream closes its socket;
StepFun exposes no explicit generation-cancel event, so closing does not
provide a server-side billing guarantee.

## Example settings

See [.env.example](.env.example) for the complete template.

| Variable | Default | Purpose |
| --- | --- | --- |
| `STEPFUN_LANGUAGE` | `en` | Language for both STT and TTS |
| `STEPFUN_ASR_TRANSPORT` | `websocket` | `websocket` or `http` |
| `STEPFUN_ASR_MODEL` | `stepaudio-2.5-asr-stream` | Streaming ASR |
| `STEPFUN_BATCH_ASR_MODEL` | `stepaudio-3-asr-max` | Batch ASR |
| `STEPFUN_LLM_MODEL` | `step-3.7-flash` | Chat model (`LLM_PROVIDER=stepfun`) |
| `STEPFUN_REASONING_EFFORT` | `low` | `low`, `medium`, or `high` |
| `STEPFUN_TTS_MODEL` | `stepaudio-3-tts` | Speech synthesis model |
| `STEPFUN_TTS_VOICE` | `vibrant-youth` | Voice ID |
| `STEPFUN_LOCAL_VAD` | `true` | Local VAD; always enabled for batch ASR |
| `STEPFUN_MIN_INTERRUPTION_DURATION` | `0.2` | Minimum detected speech, in seconds |
| `STEPFUN_NOISE_CANCELLATION` | `off` | `off`, `webrtc`, or `rnnoise` |
| `STEPFUN_MAX_UNRECOVERABLE_ERRORS` | `25` | Provider errors tolerated before the session closes |
| `STEPFUN_EVAL_MAX_CONSECUTIVE_FAILURES` | `3` | Failed trials in a row that stop an evaluation run |
| `LLM_PROVIDER` | `stepfun` | `stepfun`, `deepseek`, or `mix` |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Chat model (`LLM_PROVIDER=deepseek`) |
| `DEEPSEEK_API_KEY` | — | Required when DeepSeek is selectable |
| `LLM_MIX_PROVIDERS` | `stepfun,deepseek` | Legs of a `mix` run |
| `LLM_MIX_STRATEGY` | `round_robin` | `round_robin` or `weighted` |
| `LLM_MIX_WEIGHTS` | unweighted | e.g. `stepfun:3,deepseek:1` |

Reasoning effort is documented for `step-3.7-flash`; it is omitted for models that
do not document it.

### Switching the chat model

STT and TTS are always StepFun. Only the chat model is pluggable:

```bash
# Everything on StepFun (default, unchanged behaviour)
uv run lk agent console examples/voice_agent.py

# DeepSeek for chat, StepFun for speech
LLM_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-... \
  uv run lk agent console examples/voice_agent.py

# Alternate models turn by turn, to compare them on the same conversation
LLM_PROVIDER=mix LLM_MIX_STRATEGY=round_robin DEEPSEEK_API_KEY=sk-... \
  uv run lk agent console examples/voice_agent.py
```

A "turn" is one reply. Tool-call follow-ups are a fresh `chat()` call, so a
multi-step exchange can switch mid-conversation; per-turn metrics record which
model actually answered each turn.
Reasoning happens before speakable content and can increase reply latency.

The agent can apply local CPU noise suppression before audio reaches STT,
VAD, and turn detection:

```bash
STEPFUN_NOISE_CANCELLATION=webrtc uv run lk agent console examples/voice_agent.py
```

For RNNoise, install the optional native dependency:

```bash
uv sync --extra examples --extra noise-cancellation
STEPFUN_NOISE_CANCELLATION=rnnoise uv run lk agent console examples/voice_agent.py
```

These filters reduce background noise; they do not configure echo cancellation.

## Smoke test

This example makes live API calls and writes audio to `smoke_out/`:

```bash
uv run python examples/smoke_test.py
uv run python examples/smoke_test.py --language zh --cluster china
uv run python examples/smoke_test.py --llm-model step-3.7-flash,step-5-preview
uv run python examples/smoke_test.py --help
```

It exercises HTTP and streaming TTS, streaming and batch STT, and chat/tool
calling. Use `--llm-model none` to skip chat. CLI arguments override the example
model, language, and voice settings. Listen to the generated WAV files to
assess audio quality and pitch.

Outside a LiveKit worker job, pass your own `aiohttp.ClientSession` to STT
and TTS. The plugin leaves caller-supplied sessions open; call `aclose()` on
the plugin instances before closing the HTTP session.

## Evaluate an audio dataset

Keep the console agent running with its local evaluation endpoint enabled:

```bash
STEPFUN_EVAL_PORT=8765 uv run python examples/voice_agent.py console
# Or: STEPFUN_EVAL_PORT=8765 uv run lk agent console examples/voice_agent.py
```

In another terminal, replay your folder of WAVs:

```bash
uv run python examples/eval_agent.py run mono/ \
  --label baseline --output eval_results/baseline
```

The runner sends one recording at real-time speed, waits for the entire reply
to finish playback, then sends the next. It saves component and end-to-end
timings, raw events, transcripts, CSV/JSON summaries, and an HTML report.
Failed trials are recorded with their errors and the run continues; it stops
after three failures in a row. Evaluation mode replaces the microphone; normal
console behavior is retained when `STEPFUN_EVAL_PORT` is unset.

See [the evaluation guide](examples/EVALUATION.md) for model comparisons,
streaming versus batch ASR, repetitions, component benchmarks, and optional
reference transcripts and speech-boundary annotations.

## Development and branches

```bash
uv sync --extra dev --extra examples
./scripts/check.sh .venv
```

The script runs lint, formatting, strict type checks, and offline tests.
`./scripts/check.sh .venv1x .venv` checks multiple existing environments.
Tests use scripted responses and require no provider credentials. Native
RNNoise coverage runs only when its dependency is installed. Live API smoke
tests are separate and manual.

- `main` contains the standalone package, all examples, configuration template,
  and user documentation.
- `add-stepfun-plugin` is based on `livekit/agents` upstream history. It adds the
  plugin under `livekit-plugins/livekit-plugins-stepfun/` and provider unit tests
  under `tests/stepfun/`, with a short plugin README. Its diff contains no
  StepFun agent example, denoiser, or standalone application dependencies.

The branches use different repository histories and layouts. Port shared
plugin fixes by path; merge upstream LiveKit changes into the PR branch as
needed. Submit that branch against `livekit/agents:main` after pushing it to
a fork. See [CONTRIBUTING.md](CONTRIBUTING.md) for validation and contribution notes.

## License

[Apache-2.0](LICENSE).
