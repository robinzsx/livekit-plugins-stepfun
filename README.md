# StepFun plugin for LiveKit Agents

Streaming speech recognition, speech synthesis, and chat models from
[StepFun](https://platform.stepfun.ai/docs) (阶跃星辰), using the standard
LiveKit `STT`, `TTS`, and `LLM` interfaces. Python 3.10+ and LiveKit Agents
1.8.6+ are required.

## Run the voice agent

Install from this checkout to use this implementation:

```bash
uv sync --extra examples
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
| `STEPFUN_REASONING_EFFORT` | `low` | `low`, `medium`, `high`, or `none`; sent only to models that document it |
| `STEPFUN_TTS_MODEL` | `stepaudio-3-tts` | Speech synthesis model |
| `STEPFUN_TTS_VOICE` | `vibrant-youth` | Voice ID |
| `STEPFUN_LOCAL_VAD` | `true` | Local VAD; always enabled for batch ASR |
| `STEPFUN_MIN_INTERRUPTION_DURATION` | `0.2` | Minimum detected speech, in seconds |
| `STEPFUN_NOISE_CANCELLATION` | `off` | `off`, `webrtc`, or `rnnoise` |
| `STEPFUN_PREWARM` | `true` | Warm connections before the caller speaks |
| `LLM_PROVIDER` | `stepfun` | `stepfun` or `deepseek` |
| `DEEPSEEK_MODEL` | `deepseek-v4.1-flash` | Chat model (`LLM_PROVIDER=deepseek`) |
| `DEEPSEEK_API_KEY` | — | Required when DeepSeek is selectable |

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

```

The provider is fixed for the life of the process, so comparing two models
means two runs rather than one conversation alternating between them. That keeps
each transcript attributable to the model named in its config block.

Reasoning happens before speakable content and can increase reply latency.

## Noise cancellation

Background noise reaches STT, VAD, and turn detection, so filtering happens
before the session sees the audio and all three get the same cleaned signal:

```bash
STEPFUN_NOISE_CANCELLATION=webrtc uv run lk agent console examples/voice_agent.py
```

`webrtc` needs nothing extra — the audio-processing module ships with the
LiveKit SDK. `rnnoise` is a stronger model but a native build:

```bash
uv sync --extra examples --extra noise-cancellation
STEPFUN_NOISE_CANCELLATION=rnnoise uv run lk agent console examples/voice_agent.py
```

These are noise *suppressors*: they reduce steady background sound. They do not
provide acoustic echo cancellation, which is a property of the client device,
not of the agent.

## Build your own agent

Start from `examples/voice_agent.py`. It is a complete, working agent — the
pieces worth changing first are:

| Change | Where |
| --- | --- |
| Personality and task | `INSTRUCTIONS` at the top of `examples/voice_agent.py` |
| Opening line | `AGENT_GREETING`; set it to `none` to wait for the caller to speak first |
| Voice, language, models | `.env` |
| Add a tool | a method on `Assistant` decorated with `@function_tool` |

Tools are the usual reason to fork this: any `async def` on the `Assistant`
class decorated with `@function_tool` becomes callable by the model, and its
docstring is the description the model reads.

```python
from livekit.agents import RunContext, function_tool

ORDERS = {"A123": "packed, ships tomorrow"}

class Assistant(Agent):
    @function_tool()
    async def lookup_order(self, context: RunContext, order_id: str) -> str:
        """Look up an order by ID and return its current status.

        Call this whenever the caller asks about an order.
        """
        return ORDERS.get(order_id, "no such order")
```

The docstring is the tool description the model reads, so say when to call it,
not just what it does. Nothing else is needed: `AgentSession` discovers methods
on the agent class, and the reply is spoken through the same TTS.

Because `Assistant` subclasses `Agent`, `on_enter` and the node overrides work
as documented in the [agents guide](https://docs.livekit.io/agents/build/).

### Going past this example

This example covers the single-agent case. For handoffs, multi-step flows,
state that survives a model switch, and prompt design for speech, see the
LiveKit docs rather than growing this file.

## Development

```bash
uv sync --extra dev --extra examples
./scripts/check.sh .venv
```

The script runs lint, formatting, strict type checks, and the offline unit
tests. `./scripts/check.sh .venv1x .venv` checks multiple environments. The
tests use scripted responses and need no provider credentials or network.

This branch is the agent-oriented cut: the plugin, one runnable example, its
configuration loader, and the plugin's unit tests. The evaluation harness,
dataset replay, and smoke-test scripts live on `main`. The upstream PR lives on
`add-stepfun-plugin`, which is based on `livekit/agents` history and tracks the
plugin under `livekit-plugins/livekit-plugins-stepfun/`. Fixes to plugin
behavior belong there first; these branches are not mergeable into each other.

## License

[Apache-2.0](LICENSE).
