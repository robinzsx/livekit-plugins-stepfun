# StepFun voice-agent evaluation — metrics overview

The eval harness replays a WAV dataset into the running console agent and records
per-turn timings, accuracy, usage, and reliability. `InstrumentedAgent`
(`examples/_eval_runtime.py:57`) wraps the agent's `stt_node` / `llm_node` /
`tts_node` without modifying provider implementations; every event is stamped
with `perf_counter` offsets (`examples/_eval_metrics.py:63`). Each turn produces
one row (`TurnRecord.result`, `examples/_eval_metrics.py:79`) plus raw
`events` / `requests` / `sdk_metrics` / `errors`, and the corpus is aggregated by
`summarize` (`examples/_eval_metrics.py:188`).

## 1. Turn timing

### End-to-end (what the caller perceives)

| Field | Definition |
| --- | --- |
| `input_end_to_first_output_ms` | End of delivered audio → first frame submitted to the agent's audio output. The primary responsiveness number. |
| `speech_end_to_first_output_ms` | Same, measured from the *annotated* speech end — excludes trailing silence in the WAV. Requires a JSONL manifest. |
| `response_playout_ms` | First output frame → completion of the whole reply's playout. Cost of reply length. |
| `turn_total_ms` | Whole turn wall-clock (evaluation start → response done). |
| `input_end_to_final_ms` | End of delivered audio → first final ASR transcript. |
| `speech_end_to_turn_decision_ms` | Annotated speech end → turn-commit decision. Endpointing cost (server VAD or turn detector). |

### Stage-level (where the latency comes from)

| Field | Definition |
| --- | --- |
| `input_to_first_partial_ms` / `speech_to_first_partial_ms` | First delivered frame / speech start → first interim ASR result. Null in batch mode (no interim results). |
| `llm_first_content_ms` | LLM node start → first non-whitespace speakable content. Hides reasoning that precedes speech. |
| `tts_first_audio_ms` | First text consumed by the TTS node → first returned audio frame (TTS time-to-first-audio). |
| `llm_total_ms` / `tts_total_ms` | Sum of completed node request durations for that component. |

Stages overlap; do not add per-stage numbers to estimate end-to-end latency.

## 2. SDK-reported metrics

Native provider timings, retained alongside the observed node timings so the two
can be cross-checked.

| Field | Definition |
| --- | --- |
| `llm_sdk_ttft_ms` | `LLMMetrics.ttft` — provider-reported time to first generated token. |
| `tts_sdk_ttfb_ms` | `TTSMetrics.ttfb` — provider-reported time to first audio byte. |
| `stt_batch_request_ms` | `STTMetrics.duration` for a batch `recognize()` request. **Absent for streaming ASR**, which reports 0.0 and is deliberately not treated as a latency measurement. |
| `sdk_eou_delay_ms` | `EOUMetrics.end_of_utterance_delay` — VAD speech end → turn-end decision, as measured by the SDK. |
| `stt_acquire_ms` / `tts_acquire_ms` | WebSocket pool `acquire_time` — isolates connection/handshake cost from inference cost; pair with the `connection_reused` flag in the raw metrics. |

## 3. ASR accuracy (requires `reference` in a JSONL manifest)

| Field | Definition |
| --- | --- |
| `wer` | Word error rate after NFKC normalization, case folding, punctuation → spaces, collapsed whitespace. Corpus rate = summed edit distance / summed reference words. |
| `cer` | Character error rate with the same normalization, whitespace removed. More informative than WER for Chinese, which has no word spacing. |

## 4. Response quality

| Field | Definition |
| --- | --- |
| `keyword_match` | Per-turn boolean: every `expected_keywords` entry appears (case-insensitively) in the reply. Aggregated as `keyword_match_rate`. A cheap on-topic sanity check — not a general answer-correctness judge. |

## 5. Reliability

| Field | Definition |
| --- | --- |
| `failure_rate` | Failed trials (`timeout` / `error` / `cancelled` / `client_error`) over all trials. Failures stop the runner and require an agent restart so late transcripts or queued speech cannot contaminate later trials. |
| `retries` | Recoverable errors within a turn — transient provider failures the pipeline survived. |

## 6. Usage / cost

| Field | Definition |
| --- | --- |
| `llm_prompt_tokens` | Prompt tokens summed over the turn's LLM requests. |
| `llm_completion_tokens` | Completion tokens summed over the turn's requests. |
| `llm_reasoning_tokens` | Hidden-reasoning tokens (already included in `completion_tokens`). A large share explains slow TTFT. |
| `llm_prompt_cached_tokens` | Prompt-cache reads — context for interpreting prompt-token counts. |

## 7. Harness-integrity diagnostics

| Field | Definition |
| --- | --- |
| `frames_submitted` | Audio frames delivered during the turn — the feeder kept its 20 ms pacing. |
| `max_schedule_lag_ms` | Worst pacing jitter of the audio feeder. Guards against blaming the provider for harness-side scheduling delay. |
| `agent_evaluation_index` | Trial number since agent startup — separates cold-start trials (prewarm, first connection) from steady state. |
| `warning` | Set when a turn produced multiple ASR finals, i.e. the recording violated one-utterance-per-file. |

## 8. Corpus summary

`samples`, `successful`, `failed`, `failure_rate`, `reference_samples`,
per-metric **p50 / p95 / n** latency tables, corpus `wer` / `cer`,
`keyword_match_rate`. Documented caveats:

- Failures are excluded from latency percentiles but included in failure rates.
- p95 uses nearest rank; small runs are exploratory.
- Unknown timings are null, not zero; negative file-relative latencies are preserved (they occur when the WAV has trailing silence or the agent replies before the file ends).
- End-to-end timing stops at the agent's output boundary — not acoustic speaker onset, not remote-client receive latency.
- Native token counters can default to zero when a provider reports no breakdown; zero reasoning tokens alone does not prove no reasoning occurred.

## 9. Modes

| Mode | Purpose |
| --- | --- |
| `agent` | Full real-time pipeline: paced audio in, wait for console playout to finish. |
| `stt` | ASR only, through the agent's configured provider. Streaming is paced; batch uploads the whole recording at once (the only mode where `stt_batch_request_ms` is meaningful). |
| `llm` | Chat completion for a fixed text; measures `llm_first_content_ms` / `llm_sdk_ttft_ms` and token usage without ASR or TTS. |
| `tts` | Synthesis of a fixed text; measures `tts_first_audio_ms` / `tts_sdk_ttfb_ms` without playing audio through the console. |

## 10. Reproducibility

- `run.json` — label, resolved models/settings, dependency versions, platform, client git commit.
- `dataset.json` — dataset path and the exact trial order (after `--limit` / `--shuffle` / `--seed` / `--repeat`).
- `results.jsonl` / `events.jsonl` — complete per-trial rows and the timestamped event stream.
- `turns.csv`, `summary.json`, `report.html` — flat table, corpus summary, and self-contained comparison report.
- Comparison caveat: streaming vs batch ASR also switches the turn detector (server VAD vs `v1-mini`), so such runs compare whole configurations, not transport in isolation.
