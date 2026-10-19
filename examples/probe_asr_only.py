"""ASR-only A/B: streaming vs batch transcription latency, WITHOUT turn detection.

Runs both StepFun ASR engines on the same real clips. Streaming uses the
WebSocket with server-VAD disabled and a manual commit, so the final transcript
reflects pure ASR finalization (no endpointing decision). Batch uploads the
whole clip. Both hit the same cluster, so the gap is ASR-engine speed, not
network or turn detection.

    uv run python examples/probe_asr_only.py --n 15
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import statistics
import time
from pathlib import Path

import aiohttp
import numpy as np
from _config import get, load_env, require

from livekit.plugins.stepfun import _utils
from livekit.plugins.stepfun.models import (
    NUM_CHANNELS,
    PCM_BITS,
    PCM_CODEC,
    STT_RECOGNIZE_PATH,
    STT_STREAM_PATH,
)

load_env()
MONO = "/Users/jyxc-dz-0101666/Downloads/asr_audio/mono"


def extract_pcm(data: bytes) -> tuple[bytes, int, int, int]:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not RIFF/WAVE")
    fmt = pcm = None
    off, n = 12, len(data)
    while off + 8 <= n:
        cid = data[off : off + 4]
        sz = int.from_bytes(data[off + 4 : off + 8], "little")
        body = data[off + 8 : off + 8 + sz]
        if cid == b"fmt ":
            fmt = (
                int.from_bytes(body[2:4], "little"),
                int.from_bytes(body[4:8], "little"),
                int.from_bytes(body[14:16], "little"),
            )
        elif cid == b"data":
            pcm = body
        off += 8 + sz + (sz & 1)
    if fmt is None or pcm is None:
        raise ValueError("missing fmt/data")
    ch, rate, bits = fmt
    return pcm, rate, ch, bits


def to_16k_mono16(pcm: bytes, rate: int, ch: int, bits: int) -> bytes:
    a = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1).astype("<i2")
    if rate != 16000:
        if rate % 16000 == 0:
            a = a[:: rate // 16000]
        else:
            idx = np.linspace(0, len(a) - 1, int(len(a) * 16000 / rate)).astype(np.int64)
            a = a[idx]
    return a.astype("<i2").tobytes()


async def sse_json(resp):
    buf = b""
    async for chunk in resp.content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                pass


async def stream_asr(session, args, pcm16k: bytes) -> tuple[float, str]:
    """server_vad OFF + manual commit -> pure ASR finalization (commit->completed)."""
    url = f"{_utils.resolve_ws_base_url(None, args.cluster)}{STT_STREAM_PATH}"
    async with session.ws_connect(
        url,
        headers=_utils.auth_headers(args.api_key),
        timeout=aiohttp.ClientTimeout(total=args.timeout),
    ) as ws:
        await ws.send_str(
            json.dumps(
                {
                    "event_id": _utils.event_id(),
                    "type": "session.update",
                    "session": {
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "pcm",
                                    "codec": PCM_CODEC,
                                    "rate": 16000,
                                    "bits": PCM_BITS,
                                    "channel": NUM_CHANNELS,
                                },
                                # no turn_detection block -> server VAD disabled
                                "transcription": {
                                    "model": args.stt_model,
                                    "enable_itn": True,
                                    **({"language": args.language} if args.language else {}),
                                },
                            }
                        }
                    },
                }
            )
        )
        while True:
            ev = json.loads((await ws.receive()).data)
            if ev.get("type") == "session.updated":
                break
            if ev.get("type") == "error":
                raise RuntimeError(f"session.update error: {ev}")

        step = 16000 // 10 * 2  # 100 ms
        for i in range(0, len(pcm16k), step):
            await ws.send_str(
                json.dumps(
                    {
                        "event_id": _utils.event_id(),
                        "type": "input_audio_buffer.append",
                        "audio": _utils.pcm_to_b64(pcm16k[i : i + step]),
                    }
                )
            )
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)  # let ingest settle so commit->final is ASR compute
        t_commit = time.perf_counter()
        await ws.send_str(
            json.dumps({"event_id": _utils.event_id(), "type": "input_audio_buffer.commit"})
        )

        text = ""
        while True:
            msg = await asyncio.wait_for(ws.receive(), timeout=args.timeout)
            if msg.type is not aiohttp.WSMsgType.TEXT:
                break
            ev = json.loads(msg.data)
            if ev.get("type") == "conversation.item.input_audio_transcription.completed":
                return (time.perf_counter() - t_commit) * 1000, ev.get("transcript", "")
            if ev.get("type") == "error":
                raise RuntimeError(f"stream error: {ev}")
    return -1.0, text


async def batch_asr(session, args, pcm16k: bytes) -> tuple[float, str]:
    url = f"{_utils.resolve_http_base_url(None, args.cluster)}{STT_RECOGNIZE_PATH}"
    headers = {
        **_utils.auth_headers(args.api_key),
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "audio": {
            "data": _utils.pcm_to_b64(pcm16k),
            "input": {
                "transcription": {
                    "model": args.batch_model,
                    "enable_itn": True,
                    **({"language": args.language} if args.language else {}),
                },
                "format": {
                    "type": "pcm",
                    "codec": PCM_CODEC,
                    "rate": 16000,
                    "bits": PCM_BITS,
                    "channel": NUM_CHANNELS,
                },
            },
        }
    }
    t0 = time.perf_counter()
    text = ""
    async with session.post(
        url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"batch HTTP {resp.status}: {(await resp.text())[:200]}")
        async for ev in sse_json(resp):
            if ev.get("type") == "transcript.text.done":
                text = ev.get("text", "")
                break
    return (time.perf_counter() - t0) * 1000, text


def pct(vals):
    vals = sorted(vals)
    if not vals:
        return (None, None, 0)
    return (statistics.median(vals), vals[max(0, math.ceil(len(vals) * 0.95) - 1)], len(vals))


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--cluster", default=get("STEPFUN_CLUSTER"))
    p.add_argument("--language", default=get("STEPFUN_LANGUAGE", "en"))
    p.add_argument("--stt-model", default=get("STEPFUN_ASR_MODEL", "stepaudio-2.5-asr-stream"))
    p.add_argument("--batch-model", default=get("STEPFUN_BATCH_ASR_MODEL", "stepaudio-3-asr-max"))
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()
    args.api_key = require("STEPFUN_API_KEY")

    files = sorted(glob.glob(f"{MONO}/*.wav"))[: args.n]
    print(
        f"cluster={args.cluster}  stream={args.stt_model}  batch={args.batch_model}  n={len(files)}"
    )
    print(
        "streaming = server-VAD OFF + manual commit (commit->final);  batch = whole-clip request\n"
    )

    s_ms, b_ms = [], []
    print(f"{'clip':22s} {'dur':>6s} {'stream_final':>13s} {'batch_req':>10s}   transcript")
    async with aiohttp.ClientSession() as session:
        for f in files:
            pcm, rate, ch, bits = extract_pcm(await asyncio.to_thread(Path(f).read_bytes))
            pcm16k = to_16k_mono16(pcm, rate, ch, bits)
            dur = len(pcm16k) / 2 / 16000
            name = f.split("/")[-1]
            try:
                sm, st = await stream_asr(session, args, pcm16k)
                bm, bt = await batch_asr(session, args, pcm16k)
            except Exception as exc:
                print(f"{name:22s} ERROR {type(exc).__name__}: {exc}")
                continue
            s_ms.append(sm)
            b_ms.append(bm)
            print(f"{name:22s} {dur:5.1f}s {sm:12.0f}ms {bm:9.0f}ms   {st[:42]!r}")

    sp, bp = pct(s_ms), pct(b_ms)
    print("\n=== ASR-only finalization (no turn detection), ms ===")
    print(
        f"  streaming (commit->final): p50={sp[0]:.0f}  p95={sp[1]:.0f}  (n={sp[2]})"
        if sp[0]
        else "  streaming: no data"
    )
    print(
        f"  batch (whole-clip req):    p50={bp[0]:.0f}  p95={bp[1]:.0f}  (n={bp[2]})"
        if bp[0]
        else "  batch: no data"
    )
    if sp[0] and bp[0]:
        d = statistics.median(b_ms) - statistics.median(s_ms)
        verdict = f"streaming {d:.0f} ms faster" if d > 0 else f"batch {-d:.0f} ms faster"
        print(f"  delta (batch - stream) p50: {d:+.0f} ms  ->  {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
