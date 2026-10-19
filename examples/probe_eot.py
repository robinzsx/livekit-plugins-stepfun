"""Why doesn't v1-mini confirm end-of-turn on this English corpus?

Feeds each real clip (plus ~1s trailing silence, as the agent does at a turn
boundary) into the LOCAL v1-mini audio EOT detector and prints
end_of_turn_probability / backchannel_probability against the calibrated
unlikely_threshold. If probability < unlikely_threshold the turn is NOT
confirmed and the session waits max_delay (the ~2.5s you see).

    uv run python examples/probe_eot.py --clips 8
"""

from __future__ import annotations

import argparse
import asyncio
import glob
from pathlib import Path

from _config import load_env
from livekit.agents import LanguageCode, inference
from livekit.agents.utils import http_context

from livekit import rtc

load_env()

MONO = "/Users/jyxc-dz-0101666/Downloads/asr_audio/mono"


def read_wav_16k(path: str) -> tuple[bytes, float]:
    data = Path(path).read_bytes()
    off, n = 12, len(data)
    fmt = pcm = None
    while off + 8 <= n:
        cid = data[off : off + 4]
        sz = int.from_bytes(data[off + 4 : off + 8], "little")
        body = data[off + 8 : off + 8 + sz]
        if cid == b"fmt ":
            fmt = (int.from_bytes(body[2:4], "little"), int.from_bytes(body[4:8], "little"))
        elif cid == b"data":
            pcm = body
        off += 8 + sz + (sz & 1)
    _ch, rate = fmt
    # simple 48k->16k decimation (clips are mono 16-bit)
    step = max(1, rate // 16000)
    import numpy as np

    a = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")[::step]
    return a.astype("<i2").tobytes(), len(a) / 16000.0


async def predict_clip(detector, pcm16k: bytes, tail_s: float) -> tuple[float, float | None]:
    import numpy as np

    stream = detector.stream()
    try:
        a = np.frombuffer(pcm16k, dtype="<i2")
        if tail_s > 0:
            a = np.concatenate([a, np.zeros(int(16000 * tail_s), dtype="<i2")])
        # The local detector has a fixed 19200-sample (~1.2s) buffer; feed small
        # frames so it keeps a sliding window instead of rejecting an oversized one.
        step = 1600  # 100 ms
        for i in range(0, len(a), step):
            chunk = a[i : i + step]
            stream.push_audio(rtc.AudioFrame(chunk.tobytes(), 16000, 1, len(chunk)))
        await asyncio.sleep(0.2)  # let the main task drain the channel
        fut = stream.predict()
        ev = await asyncio.wait_for(fut, timeout=3.0)
        return ev.end_of_turn_probability, ev.backchannel_probability
    finally:
        await stream.aclose()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clips", type=int, default=8)
    p.add_argument("--tail", type=float, default=1.0, help="trailing silence seconds")
    p.add_argument("--language", default="en")
    p.add_argument(
        "--version",
        default="v1-mini",
        choices=["v1-mini", "v1"],
        help="v1-mini runs locally; v1 is the LiveKit Inference cloud detector "
        "(needs LIVEKIT_API_KEY/SECRET, billed per audio duration)",
    )
    args = p.parse_args()

    # The cloud v1 detector needs a job-scoped HTTP session; a bare script has
    # none, so open one (a no-op for the local v1-mini).
    async with http_context.open():
        try:
            detector = inference.TurnDetector(version=args.version)
        except ValueError as exc:
            raise SystemExit(
                f"could not build the {args.version} detector: {exc}\n"
                "For v1 set LIVEKIT_API_KEY and LIVEKIT_API_SECRET (LiveKit Cloud project "
                "keys) in the environment/.env, and ensure egress to "
                "https://agent-gateway.livekit.cloud."
            ) from exc
        await _run(detector, args)


async def _run(detector: inference.TurnDetector, args: argparse.Namespace) -> None:
    lang = LanguageCode(args.language)
    probe = detector.stream()
    unlikely = await probe.unlikely_threshold(lang)
    bc_thr = await probe.backchannel_threshold(lang)
    supports = await probe.supports_language(lang)
    await probe.aclose()
    print(f"{args.version}  language={args.language}  supports_language={supports}")
    print(f"  unlikely_threshold={unlikely}  backchannel_threshold={bc_thr}")
    print(f"  (commit fast when end_of_turn_probability >= {unlikely}; else wait max_delay)\n")

    files = sorted(glob.glob(f"{MONO}/*.wav"))[: args.clips]
    confirmed = 0
    print(f"{'clip':22s} {'dur':>6s} {'eot_prob':>9s} {'bc_prob':>8s}  verdict")
    for f in files:
        pcm, dur = read_wav_16k(f)
        try:
            prob, bc = await predict_clip(detector, pcm, args.tail)
        except Exception as exc:
            print(f"{f.split('/')[-1]:22s} ERROR {type(exc).__name__}: {exc}")
            continue
        ok = unlikely is not None and prob >= unlikely
        confirmed += int(ok)
        bcs = f"{bc:.2f}" if bc is not None else "-"
        print(
            f"{f.split('/')[-1]:22s} {dur:5.1f}s {prob:9.3f} {bcs:>8s}  "
            f"{'CONFIRM (min_delay)' if ok else 'not-confirmed -> max_delay 2.5s'}"
        )
    print(f"\nconfirmed {confirmed}/{len(files)} clips at unlikely_threshold={unlikely}")


if __name__ == "__main__":
    asyncio.run(main())
