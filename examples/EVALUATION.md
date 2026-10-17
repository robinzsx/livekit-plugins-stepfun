# Dataset replay and latency evaluation

Run the voice agent in one terminal and the evaluator in another. The same
agent session stays running across the entire folder. The evaluator uploads
one WAV, waits for the reply's SDK playout completion, then uploads the next.
It does not use a fixed response sleep or send text in place of audio.

## Start the agent

Install the example dependencies and configure your StepFun key in `.env`:

```bash
uv sync --extra examples --extra dev
```

Enable the local evaluation endpoint when launching the existing agent:

```bash
STEPFUN_EVAL_PORT=8765 uv run python examples/voice_agent.py console
```

Or use the LiveKit CLI:

```bash
STEPFUN_EVAL_PORT=8765 uv run lk agent console examples/voice_agent.py
```

With an activated environment, omit `uv run`. Both commands run the same agent.
Wait for the log message `Dataset evaluation ready at http://127.0.0.1:8765`.
The endpoint binds to loopback only and accepts one evaluation at a time.

Evaluation mode replaces microphone input with dataset audio and silence.
Replies still play through the console's normal audio output. Keep the console
in audio mode throughout the run; do not toggle to text mode. Leaving
`STEPFUN_EVAL_PORT` unset keeps normal microphone behavior.

## Replay your mono folder

Prepare the folder whenever you are ready:

```text
mono/
  001.wav
  002.wav
  003.wav
```

In a second terminal:

```bash
uv run python examples/eval_agent.py run mono/ \
  --label streaming-baseline --output eval_results/streaming-baseline
```

The folder is read in filename order, without recursion. Use zero-padded
filenames if numeric ordering matters. WAV input is decoded and normalized to
mono PCM16 at 16 kHz. Each recording must be nonempty and at most 120 seconds.
Use recordings containing one user turn each.

Audio is delivered in real time, followed by continuous silence so server VAD
can detect the end of speech. Once the reply finishes, the runner waits an
optional 250 ms gap before sending the next file. The agent remains running
after the final file, so you can launch another evaluation against it.

Useful options:

```bash
# Screen the first five files.
uv run python examples/eval_agent.py run mono/ --limit 5

# Three repetitions with reproducible randomized trial order.
uv run python examples/eval_agent.py run mono/ --repeat 3 --shuffle --seed 42

# Treat successive recordings as a continuing conversation.
uv run python examples/eval_agent.py run mono/ --keep-history
```

By default, the LLM conversation context resets before every recording to
make independent prompts comparable. The worker, provider clients, and speech
connections remain alive. `--keep-history` preserves LLM context; its results
will depend on recording order and increasing context length.

`--timeout 120` allows 120 seconds beyond the input duration for a response to
finish. Provider errors and timeouts are saved as failed trials — the row, its
errors, and its event timeline — and the run continues. After a failed trial the
agent drops queued speech and pending audio so the next recording starts clean.
Only a streak of failures stops the runner: `--max-consecutive-failures 3`
(the default) stops after three failed trials in a row, which means the agent
itself is broken and needs a restart. The agent applies the same limit to its
evaluation endpoint (`STEPFUN_EVAL_MAX_CONSECUTIVE_FAILURES`) and then refuses
further requests until restarted. Invalid uploads are rejected before replay
starts. Completed results are flushed to disk after each recording, including
when a later trial fails.

Transient connection failures are retried before a trial is failed: the
framework retries a synthesis stream up to three times when no audio has been
sent to the user yet, and the plugin replaces a pooled socket the server
already dropped. Every retry is recorded as a recoverable error, so the
`retries` column and the `provider_error` events show what happened; the close
code in the message says why the socket died. A socket that drops mid-reply,
after audio was already played, is not retried — the user already heard it —
and fails the trial instead.

## Compare models and ASR modes

For each configuration, restart the agent with the desired settings, then
run the same folder with a distinct label/output directory. For example:

```bash
# Terminal 1: streaming ASR.
STEPFUN_EVAL_PORT=8765 STEPFUN_ASR_TRANSPORT=websocket \
  uv run python examples/voice_agent.py console

# Terminal 2:
uv run python examples/eval_agent.py run mono/ \
  --label streaming --output eval_results/streaming
```

Stop that agent and restart for the batch configuration:

```bash
# Terminal 1: batch ASR.
STEPFUN_EVAL_PORT=8765 STEPFUN_ASR_TRANSPORT=http \
  STEPFUN_BATCH_ASR_MODEL=stepaudio-3-asr-max \
  uv run python examples/voice_agent.py console

# Terminal 2:
uv run python examples/eval_agent.py run mono/ \
  --label batch --output eval_results/batch
```

Use `STEPFUN_LLM_MODEL`, `STEPFUN_REASONING_EFFORT`, `STEPFUN_TTS_MODEL`, and
`STEPFUN_TTS_VOICE` for other comparisons. Use `STEPFUN_REASONING_EFFORT=none`
to omit effort on models where the setting is undocumented. Change one
setting at a time and keep language, cluster, dataset, and history policy fixed.

Streaming and batch ASR accept different model IDs. The example also uses
different turn detectors in the two modes. Those runs compare complete
configurations; they do not isolate the causal effect of transport alone.

`STEPFUN_PREWARM=false` disables explicit provider prewarming.
This does not guarantee a fully cold call: a running agent opens its STT
stream before the first recording, and provider/SDK caches can remain warm.
`agent_evaluation_index` records the trial number since agent startup;
connection metrics record acquisition time and reuse. Treat startup and
steady-state measurements separately.

## Read the results

Every run directory contains:

| File | Contents |
| --- | --- |
| `run.json` | Label, resolved model/settings, dependency versions, platform, client Git commit |
| `dataset.json` | Dataset path and exact trial order |
| `results.jsonl` | Complete per-trial results, metadata, requests, errors, and SDK metrics |
| `events.jsonl` | Timestamped events with run/sample IDs |
| `turns.csv` | Flat latency, transcript, response, usage, and quality fields |
| `summary.json` | Sample counts, failure rate, median/p95 timings, corpus WER/CER when available |
| `report.html` | Self-contained tables and expandable per-turn event timelines |

Compare existing runs without calling any providers:

```bash
uv run python examples/eval_agent.py report \
  eval_results/streaming eval_results/batch \
  --output eval_results/comparison.html
```

The main timings are:

| Field | Definition |
| --- | --- |
| `input_to_first_partial_ms` | First delivered input frame to first interim ASR result |
| `input_end_to_final_ms` | End of delivered file to first final ASR result |
| `llm_first_content_ms` | LLM node start to first non-whitespace speakable content |
| `tts_first_audio_ms` | First text consumed by TTS node to first returned audio frame |
| `input_end_to_first_output_ms` | End of delivered file to first frame submitted to agent audio output |
| `response_playout_ms` | First output frame to completion of the entire reply/playout |
| `llm_total_ms`, `tts_total_ms` | Sum of completed node request durations for that component |
| `stt_batch_request_ms` | SDK batch request duration; absent for streaming ASR |
| `*_sdk_*`, `*_acquire_ms` | Native SDK timings, retained alongside the observed node timings |

Streaming partial latency is absent for batch ASR. Unknown timings are null,
not zero. Streaming SDK request duration is zero and is not treated as an ASR
latency measurement. All SDK payloads are retained to inspect token throughput,
reasoning tokens, connection reuse, cancellation, and request IDs.
Native token counters can default to zero when a provider does not report a
breakdown; zero reasoning tokens alone does not establish that no reasoning occurred.

End-to-end timing stops at the agent's output boundary; it does not measure
acoustic speaker onset or remote-client receive latency. Reply completion uses
the active output sink's playout signal. Measurements include local scheduling,
networking, provider queueing, and inference. Pipeline stages overlap, so do not
add their durations to estimate end-to-end latency.

Median/p95 tables include successful trials and display each metric's sample
count. Failed trials contribute to the failure rate. p95 uses nearest rank;
small runs are exploratory. Negative file-relative latencies are preserved:
they can occur when the WAV includes silence after speech or the agent responds
before the file finishes.

## Optional transcripts and speech boundaries

A plain folder is enough for file-relative latency measurements. To measure
speech-end latency and ASR accuracy, optionally create a JSONL manifest:

```json
{"id":"001","audio":"mono/001.wav","reference":"你好，请介绍一下你自己。","speech_start_s":0.2,"speech_end_s":2.8}
{"id":"002","audio":"mono/002.wav","reference":"What is the capital of France?","expected_keywords":["Paris"]}
```

Paths are relative to the manifest file. Speech annotations must specify both
start and end, in seconds within the recording. Without annotations,
`speech_end_to_*` fields remain null: file end is not assumed to be speech end.
Annotated boundaries are mapped to actual paced frame delivery times.
Delivery uses 20 ms frames. Isolated batch-ASR benchmarks have no paced
speech clock; use their request duration rather than speech-relative timings.

```bash
uv run python examples/eval_agent.py run dataset.jsonl
```

WER/CER use Unicode NFKC, case folding, punctuation replaced by spaces, and
collapsed whitespace. WER splits on whitespace; CER removes whitespace.
Corpus rates sum edit counts over reference lengths. CER is more informative
for Chinese text without spaces. `expected_keywords` supplies a simple response
keyword check; it is not a general answer-correctness judge.

## Isolated component benchmarks

These use the running agent's configured providers, without generating a full
conversation. They run sequentially and save the same result format:

```bash
uv run python examples/eval_agent.py run mono/ --mode stt --label asr-only
uv run python examples/eval_agent.py run mono/ --mode llm \
  --text "Explain what LiveKit does in one short sentence." --label llm-only
uv run python examples/eval_agent.py run mono/ --mode tts \
  --text "Hello, this is a voice latency test." --label tts-only
```

LLM/TTS modes use `--text`, or each manifest entry's `text`/`reference`; they do
not derive text from the WAV. These modes run once per selected dataset entry
per repetition. Streaming STT is paced; isolated batch STT uploads the complete
recording immediately. Isolated TTS consumes generated audio without playing
it through the console. Full-agent mode always uses real-time audio input and
waits for console reply playback to finish.
