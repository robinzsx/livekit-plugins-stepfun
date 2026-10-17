"""Turn records and offline reports for the voice-agent evaluation tools."""

from __future__ import annotations

import html
import json
import math
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any


def error_counts(reference: str, hypothesis: str) -> dict[str, int | float | None]:
    """NFKC/casefold; replace punctuation with spaces, then compare words/characters."""

    def normalize(text: str) -> str:
        return " ".join(
            "".join(
                " " if unicodedata.category(c).startswith("P") else c
                for c in unicodedata.normalize("NFKC", text).casefold()
            ).split()
        )

    def distance(a: list[str], b: list[str]) -> int:
        previous = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            current = [i]
            for j, y in enumerate(b, 1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
            previous = current
        return previous[-1]

    ref, hyp = normalize(reference), normalize(hypothesis)
    words, characters = ref.split(), list(ref.replace(" ", ""))
    word_errors = distance(words, hyp.split())
    char_errors = distance(characters, list(hyp.replace(" ", "")))
    return {
        "reference_words": len(words),
        "word_errors": word_errors,
        "reference_characters": len(characters),
        "character_errors": char_errors,
        "wer": word_errors / len(words) if words else None,
        "cer": char_errors / len(characters) if characters else None,
    }


@dataclass
class TurnRecord:
    sample_id: str
    origin: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)
    requests: list[dict[str, Any]] = field(default_factory=list)
    sdk_metrics: list[dict[str, Any]] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    frames_submitted: int = 0
    max_schedule_lag_ms: float = 0.0

    def note(self, event: str, *, at: float | None = None, **data: Any) -> None:
        now = time.perf_counter() if at is None else at
        self.marks.setdefault(event, now)
        self.events.append({"event": event, "offset_ms": (now - self.origin) * 1000, **data})

    def begin_request(self, component: str) -> dict[str, Any]:
        request = {"component": component, "start": time.perf_counter(), "cancelled": True}
        self.requests.append(request)
        self.note(f"{component}_request_start", request_index=len(self.requests) - 1)
        return request

    def delta_ms(self, end: str, start: str) -> float | None:
        if end not in self.marks or start not in self.marks:
            return None
        return (self.marks[end] - self.marks[start]) * 1000

    def result(self, *, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
        def head_latency(component: str, first: str, start: str) -> float | None:
            for request in self.requests:
                if request["component"] != component or request.get("cancelled"):
                    continue
                if first in request and start in request:
                    return (request[first] - request[start]) * 1000
            return None

        def sdk_value(kind: str, name: str, *, positive: bool = False) -> float | None:
            for metric in self.sdk_metrics:
                value = metric.get(name)
                if metric.get("type") == kind and isinstance(value, (int, float)):
                    if metric.get("cancelled") or value < 0 or (positive and value == 0):
                        continue
                    return value * 1000
            return None

        def request_duration(component: str) -> float | None:
            durations = [
                (request["end"] - request["start"]) * 1000
                for request in self.requests
                if request["component"] == component
                and not request.get("cancelled")
                and "end" in request
            ]
            return sum(durations) if durations else None

        row = {
            "sample_id": self.sample_id,
            "status": status,
            "transcript": " ".join(self.transcripts),
            "response": "\n".join(self.responses),
            "asr_final_count": len(self.transcripts),
            "assistant_response_count": len(self.responses),
            "user_turn_count": sum(event["event"] == "turn_decision" for event in self.events),
            "input_to_first_partial_ms": self.delta_ms("asr_partial", "input_start"),
            "speech_to_first_partial_ms": self.delta_ms("asr_partial", "speech_start"),
            "input_end_to_final_ms": self.delta_ms("asr_final", "input_end"),
            "speech_end_to_final_ms": self.delta_ms("asr_final", "speech_end"),
            "speech_end_to_turn_decision_ms": self.delta_ms("turn_decision", "speech_end"),
            "llm_first_content_ms": head_latency("llm", "first_content", "start"),
            "tts_first_audio_ms": head_latency("tts", "first_audio", "first_text"),
            "llm_total_ms": request_duration("llm"),
            "tts_total_ms": request_duration("tts"),
            "input_end_to_first_output_ms": self.delta_ms("output_first_audio", "input_end"),
            "speech_end_to_first_output_ms": self.delta_ms("output_first_audio", "speech_end"),
            "response_playout_ms": self.delta_ms("response_done", "output_first_audio"),
            "turn_total_ms": self.delta_ms("response_done", "evaluation_start"),
            "llm_sdk_ttft_ms": sdk_value("llm_metrics", "ttft"),
            "tts_sdk_ttfb_ms": sdk_value("tts_metrics", "ttfb"),
            "stt_batch_request_ms": sdk_value("stt_metrics", "duration", positive=True),
            "sdk_eou_delay_ms": sdk_value("eou_metrics", "end_of_utterance_delay", positive=True),
            "stt_acquire_ms": sdk_value("stt_metrics", "acquire_time"),
            "tts_acquire_ms": sdk_value("tts_metrics", "acquire_time"),
            "retries": sum(bool(e.get("recoverable")) for e in self.errors),
            "frames_submitted": self.frames_submitted,
            "max_schedule_lag_ms": self.max_schedule_lag_ms,
        }
        llm_metrics = [
            metric
            for metric in self.sdk_metrics
            if metric.get("type") == "llm_metrics" and not metric.get("cancelled")
        ]
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "prompt_cached_tokens",
        ):
            row[f"llm_{key}"] = (
                sum(metric.get(key, 0) for metric in llm_metrics) if llm_metrics else None
            )
        row["agent_evaluation_index"] = metadata.get("agent_evaluation_index")
        if len(self.transcripts) > 1:
            row["warning"] = "Multiple ASR finals; use one user utterance per recording"
        reference = metadata.get("reference")
        if (
            isinstance(reference, str)
            and reference.strip()
            and metadata.get("mode", "agent") in ("agent", "stt")
        ):
            row.update(error_counts(reference, row["transcript"]))
        keywords = metadata.get("expected_keywords")
        if keywords and metadata.get("mode", "agent") in ("agent", "llm"):
            row["keyword_match"] = all(
                str(k).casefold() in row["response"].casefold() for k in keywords
            )
        requests = [
            {
                key: (
                    (value - self.origin) * 1000
                    if key in ("start", "end", "first_content", "first_text", "first_audio")
                    else value
                )
                for key, value in request.items()
            }
            for request in self.requests
        ]
        return {
            "row": row,
            "metadata": metadata,
            "events": self.events,
            "requests": requests,
            "sdk_metrics": self.sdk_metrics,
            "errors": self.errors,
        }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [result["row"] for result in results]
    successful = [row for row in rows if row["status"] == "ok"]
    summary: dict[str, Any] = {
        "samples": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "failure_rate": (len(rows) - len(successful)) / len(rows) if rows else None,
        "latencies": {},
        "reference_samples": sum(bool(row.get("reference_characters")) for row in successful),
    }
    keys = sorted({key for row in rows for key in row if key.endswith("_ms")})
    for key in keys:
        values = sorted(
            row[key]
            for row in successful
            if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
        )
        if values:
            summary["latencies"][key] = {
                "n": len(values),
                "p50": statistics.median(values),
                "p95": values[max(0, math.ceil(len(values) * 0.95) - 1)],
            }
    for rate, errors, count in (
        ("wer", "word_errors", "reference_words"),
        ("cer", "character_errors", "reference_characters"),
    ):
        total = sum(row.get(count, 0) for row in successful)
        summary[rate] = sum(row.get(errors, 0) for row in successful) / total if total else None
    keyword_checks = [row["keyword_match"] for row in successful if "keyword_match" in row]
    summary["keyword_checks"] = len(keyword_checks)
    summary["keyword_match_rate"] = (
        sum(keyword_checks) / len(keyword_checks) if keyword_checks else None
    )
    return summary


def render_report(runs: list[dict[str, Any]]) -> str:
    """Self-contained comparison tables and inspectable event timelines."""

    def escape(value):
        return html.escape(str(value))

    parts = [
        """<!doctype html><meta charset="utf-8"><title>StepFun evaluation</title>
<style>body{font:15px system-ui;margin:32px;max-width:1500px;color:#172033}
table{border-collapse:collapse;margin:20px 0;width:100%}td,th{border:1px solid #ccd4df;
padding:8px;text-align:left}th{background:#eef2f7}pre{white-space:pre-wrap}
details{margin:12px 0}small{color:#536174}</style><h1>StepFun evaluation</h1>
<p>Times are milliseconds. End-to-end timing ends at the agent's audio output boundary,
not the acoustic speaker. Speech-boundary metrics require manifest annotations.
p95 uses nearest rank; small runs are exploratory. Failures are excluded from latency
percentiles and included in failure rates. Overlapping stages must not be added.</p>"""
    ]
    keys = sorted({key for run in runs for key in run["summary"]["latencies"]})
    parts.append(
        "<table><tr><th>Measurement</th>"
        + "".join(f"<th>{escape(run['label'])}<br>p50 / p95 (n)</th>" for run in runs)
        + "</tr>"
    )
    for key in keys:
        parts.append(f"<tr><td>{escape(key)}</td>")
        for run in runs:
            metric = run["summary"]["latencies"].get(key)
            value = f"{metric['p50']:.1f} / {metric['p95']:.1f} ({metric['n']})" if metric else "—"
            parts.append(f"<td>{value}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    for run in runs:
        parts.append(
            f"<h2>{escape(run['label'])}</h2>"
            f"<pre>{escape(json.dumps(run['config'], indent=2))}</pre>"
        )
        parts.append(f"<pre>{escape(json.dumps(run['summary'], indent=2))}</pre>")
        for result in run["results"]:
            row = result["row"]
            parts.append(
                f"<details><summary>{escape(row['sample_id'])}: {escape(row['status'])}</summary>"
            )
            parts.append(f"<pre>{escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre>")
            parts.append("<table><tr><th>Offset (ms)</th><th>Event</th><th>Data</th></tr>")
            for event in sorted(result["events"], key=lambda event: event["offset_ms"]):
                data = {
                    key: value for key, value in event.items() if key not in ("offset_ms", "event")
                }
                parts.append(
                    f"<tr><td>{event['offset_ms']:.1f}</td><td>{escape(event['event'])}</td>"
                    f"<td>{escape(json.dumps(data, ensure_ascii=False))}</td></tr>"
                )
            parts.append("</table></details>")
    return "".join(parts)
