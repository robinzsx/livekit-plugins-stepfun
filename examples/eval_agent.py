"""Replay WAV datasets against a running console agent, or compare saved runs."""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from _eval_metrics import render_report, summarize


def load_dataset(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """A folder of WAVs, or JSONL entries with paths relative to the manifest."""
    if path.is_dir():
        entries = [
            {"audio": str(p.resolve()), "id": p.name}
            for p in sorted(path.iterdir())
            if p.is_file() and p.suffix.lower() == ".wav"
        ]
    elif path.is_file() and path.suffix.lower() == ".jsonl":
        entries = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict) or not isinstance(entry.get("audio"), str):
                    raise ValueError("Each entry requires an audio path")
                entry["audio"] = str((path.parent / entry["audio"]).resolve())
                entry.setdefault("id", Path(entry["audio"]).name)
                entries.append(entry)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    else:
        raise ValueError(f"Dataset must be a WAV folder or JSONL manifest: {path}")
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise ValueError(f"No WAV files found in {path}")
    ids = set()
    for entry in entries:
        audio = Path(entry["audio"])
        if not audio.is_file() or audio.suffix.lower() != ".wav":
            raise ValueError(f"Audio must be an existing WAV file: {audio}")
        if not isinstance(entry["id"], str) or not entry["id"] or entry["id"] in ids:
            raise ValueError(f"Sample IDs must be nonempty and unique: {entry['id']!r}")
        ids.add(entry["id"])
    return entries


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_artifacts(folder: Path, run: dict[str, Any]) -> None:
    run["summary"] = summarize(run["results"])
    _write_json(folder / "summary.json", run["summary"])
    rows = [result["row"] for result in run["results"]]
    fields = ["sample_id", "status"] + sorted(
        {key for row in rows for key in row} - {"sample_id", "status"}
    )
    with (folder / "turns.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (folder / "report.html").write_text(render_report([run]), encoding="utf-8")


async def wait_ready(client, endpoint: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            async with client.get(
                endpoint + "/health", timeout=aiohttp.ClientTimeout(total=3)
            ) as response:
                response.raise_for_status()
                health = await response.json()
                if health.get("error"):
                    raise RuntimeError(health["error"])
                if health.get("ready"):
                    return health
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Agent not ready at {endpoint}. "
                "Start the console agent with a matching STEPFUN_EVAL_PORT."
            )
        await asyncio.sleep(0.5)


async def run_dataset(args) -> int:
    dataset = load_dataset(args.dataset, limit=args.limit)
    if args.mode in ("llm", "tts") and not args.text:
        if any(not (entry.get("text") or entry.get("reference")) for entry in dataset):
            raise ValueError(
                "LLM/TTS benchmarks need --text or text/reference in every manifest entry"
            )
    endpoint = args.endpoint.rstrip("/")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = args.output or Path("eval_results") / run_id
    trials = [
        (sample, repetition) for repetition in range(1, args.repeat + 1) for sample in dataset
    ]
    if args.shuffle:
        random.Random(args.seed).shuffle(trials)
    git = await asyncio.to_thread(
        subprocess.run,
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=3,
    )
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout + 150)
    ) as client:
        health = await wait_ready(client, endpoint, args.wait_ready)
        if health.get("busy"):
            raise RuntimeError("Agent is already evaluating another sample")
        await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=False)
        config = {
            **health["config"],
            "mode": args.mode,
            "keep_history": args.keep_history,
            "repeat": args.repeat,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "client_git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "response_timeout_s": args.timeout,
            "gap_s": args.gap,
        }
        run = {"run_id": run_id, "label": args.label or run_id, "config": config, "results": []}
        _write_json(
            folder / "run.json", {key: value for key, value in run.items() if key != "results"}
        )
        _write_json(
            folder / "dataset.json",
            {
                "source": str(args.dataset.absolute()),
                "trials": [{**sample, "repetition": repetition} for sample, repetition in trials],
            },
        )
        print(f"Run {run_id}: {len(trials)} samples → {folder}", flush=True)
        consecutive_failures = 0
        try:
            with (
                (folder / "results.jsonl").open("a", encoding="utf-8") as results_file,
                (folder / "events.jsonl").open("a", encoding="utf-8") as events_file,
            ):
                for index, (sample, repetition) in enumerate(trials, 1):
                    sample_id = f"{sample['id']}#r{repetition}"
                    audio = await asyncio.to_thread(Path(sample["audio"]).read_bytes)
                    payload = {
                        **{
                            key: value
                            for key, value in sample.items()
                            if key not in ("audio", "id")
                        },
                        "sample_id": sample_id,
                        "trial_index": index,
                        "repetition": repetition,
                        "mode": args.mode,
                        "reset_history": not args.keep_history,
                        "response_timeout": args.timeout,
                        "audio_sha256": hashlib.sha256(audio).hexdigest(),
                    }
                    if args.mode in ("agent", "stt"):
                        payload["audio_b64"] = base64.b64encode(audio).decode("ascii")
                    if args.text:
                        payload["text"] = args.text
                    print(
                        f"[{index}/{len(trials)}] {sample_id}: sending, waiting for completion…",
                        flush=True,
                    )
                    try:
                        async with client.post(endpoint + "/evaluate", json=payload) as response:
                            if response.status >= 400:
                                raise RuntimeError(
                                    f"HTTP {response.status}: {await response.text()}"
                                )
                            result = await response.json()
                    except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
                        result = {
                            "row": {"sample_id": sample_id, "status": "client_error"},
                            "metadata": {
                                key: value for key, value in payload.items() if key != "audio_b64"
                            },
                            "events": [],
                            "requests": [],
                            "sdk_metrics": [],
                            "errors": [{"message": str(exc), "recoverable": False}],
                        }
                    run["results"].append(result)
                    results_file.write(
                        json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    results_file.flush()
                    for event in result["events"]:
                        events_file.write(
                            json.dumps(
                                {"run_id": run_id, "sample_id": sample_id, **event},
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                    events_file.flush()
                    row = result["row"]
                    latency = row.get("input_end_to_first_output_ms")
                    timing = (
                        f", input end → output: {latency:.0f} ms" if latency is not None else ""
                    )
                    print(f"  {row['status']}{timing}", flush=True)
                    if row["status"] != "ok":
                        # The failure is already saved to results.jsonl with
                        # its errors and event timeline. One bad turn — a
                        # dropped provider socket, say — does not end the run;
                        # only a streak of them does.
                        print(f"  Failure recorded: {result['errors']}", flush=True)
                        consecutive_failures += 1
                        if consecutive_failures >= args.max_consecutive_failures:
                            print(
                                f"  Stopping: {consecutive_failures} consecutive failed "
                                "trials; restart the agent before continuing.",
                                flush=True,
                            )
                            break
                    else:
                        consecutive_failures = 0
                    await asyncio.sleep(args.gap)
        finally:
            write_artifacts(folder, run)
            print(f"Report: {folder / 'report.html'}", flush=True)
    return 0 if len(run["results"]) == len(trials) and not run["summary"]["failed"] else 1


def compare_runs(paths: list[Path], output: Path) -> None:
    runs = []
    for path in paths:
        run = json.loads((path / "run.json").read_text(encoding="utf-8"))
        run["results"] = [
            json.loads(line)
            for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        run["summary"] = summarize(run["results"])
        runs.append(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(runs), encoding="utf-8")
    print(f"Comparison: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Send WAVs sequentially to a running agent")
    run.add_argument("dataset", type=Path, help="WAV folder, e.g. mono/, or JSONL manifest")
    run.add_argument("--endpoint", default="http://127.0.0.1:8765")
    run.add_argument(
        "--output", type=Path, help="New results directory; defaults to eval_results/<run-id>"
    )
    run.add_argument("--label", help="Name shown in comparison reports")
    run.add_argument("--mode", choices=("agent", "stt", "llm", "tts"), default="agent")
    run.add_argument("--text", help="Fixed prompt/text for isolated LLM/TTS benchmarks")
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--limit", type=int)
    run.add_argument("--shuffle", action="store_true", help="Shuffle trials reproducibly")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--keep-history", action="store_true", help="Keep conversation context between samples"
    )
    run.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Response budget in seconds, beyond input duration",
    )
    run.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Stop after this many failed trials in a row; single failures are recorded "
        "and the run continues (default: 3)",
    )
    run.add_argument(
        "--wait-ready", type=float, default=60, help="Startup readiness budget in seconds"
    )
    run.add_argument(
        "--gap", type=float, default=0.25, help="Silence after completed playback, in seconds"
    )
    report = commands.add_parser("report", help="Compare saved runs without calling models")
    report.add_argument("runs", type=Path, nargs="+")
    report.add_argument("--output", type=Path, default=Path("eval_results/comparison.html"))
    args = parser.parse_args()
    try:
        if args.command == "report":
            compare_runs(args.runs, args.output)
            return 0
        if args.repeat < 1 or (args.limit is not None and args.limit < 1):
            parser.error("--repeat and --limit must be positive")
        if (
            not 1 <= args.timeout <= 600
            or not 0 <= args.gap <= 60
            or not 0 < args.wait_ready <= 600
            or not 1 <= args.max_consecutive_failures <= 100
        ):
            parser.error("Invalid timeout, startup wait, gap, or failure limit")
        return asyncio.run(run_dataset(args))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Evaluation interrupted; completed samples remain on disk.\n")


if __name__ == "__main__":
    sys.exit(main())
