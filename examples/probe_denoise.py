"""A/B the local denoisers (rnnoise vs webrtc) for ASR robustness in noise.

For each clean clip we take its clean batch-ASR transcript as the reference,
inject noise (white noise, and a competing talker = another clip), then run the
audio through each denoiser and batch-ASR the result. Lower WER/CER vs the
clean reference = the denoiser better preserves the target speech. Also reports
the denoiser's CPU cost (real-time factor).

    uv run python examples/probe_denoise.py --clips 3
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
from _config import get, load_env, require
from _eval_metrics import error_counts
from _noise_cancellation import _RNNoiseBackend, _WebRTCBackend

from livekit.plugins.stepfun import _utils
from livekit.plugins.stepfun.models import (
    NUM_CHANNELS,
    PCM_BITS,
    PCM_CODEC,
    STT_RECOGNIZE_PATH,
)

load_env()
MONO = "/Users/jyxc-dz-0101666/Downloads/asr_audio/mono"


def read_wav(path: str) -> np.ndarray:
    data = Path(path).read_bytes()
    off, n = 12, len(data)
    pcm = None
    while off + 8 <= n:
        cid = data[off : off + 4]
        sz = int.from_bytes(data[off + 4 : off + 8], "little")
        if cid == b"data":
            pcm = data[off + 8 : off + 8 + sz]
        off += 8 + sz + (sz & 1)
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").copy()


def decimate(sig: np.ndarray, factor: int = 3) -> np.ndarray:
    return np.ascontiguousarray(sig[::factor]).astype("<i2")


def add_white(sig: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(sig.astype(np.float64) ** 2)) + 1e-9
    noise = rng.standard_normal(len(sig))
    noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-9) * rms * 10 ** (-snr_db / 20)
    return np.clip(sig.astype(np.float64) + noise, -32768, 32767).astype(np.int16)


def add_talker(sig: np.ndarray, interferer: np.ndarray, snr_db: float) -> np.ndarray:
    n = len(sig)
    if len(interferer) < n:
        interferer = np.tile(interferer, int(np.ceil(n / len(interferer)) + 1))[:n]
    else:
        interferer = interferer[:n]
    s_rms = np.sqrt(np.mean(sig.astype(np.float64) ** 2)) + 1e-9
    i_rms = np.sqrt(np.mean(interferer.astype(np.float64) ** 2)) + 1e-9
    scaled = interferer.astype(np.float64) / i_rms * s_rms * 10 ** (-snr_db / 20)
    return np.clip(sig.astype(np.float64) + scaled, -32768, 32767).astype(np.int16)


def run_denoise(kind: str, sig: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    if kind == "none":
        out = sig
    else:
        backend = _RNNoiseBackend() if kind == "rnnoise" else _WebRTCBackend()
        chunks = [backend.process(sig[i : i + 480]) for i in range(0, len(sig) - 479, 480)]
        out = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
        backend.close()
    return out, (time.perf_counter() - t0) * 1000


async def sse_json(resp):
    buf = b""
    async for chunk in resp.content.iter_any():
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                return
            try:
                yield json.loads(p)
            except json.JSONDecodeError:
                pass


async def batch_asr(session, args, pcm16k: bytes) -> str:
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
    async with session.post(
        url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}")
        async for ev in sse_json(resp):
            if ev.get("type") == "transcript.text.done":
                return ev.get("text", "")
    return ""


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clips", type=int, default=3)
    p.add_argument("--cluster", default=get("STEPFUN_CLUSTER"))
    p.add_argument("--language", default=get("STEPFUN_LANGUAGE", "en"))
    p.add_argument("--batch-model", default=get("STEPFUN_BATCH_ASR_MODEL", "stepaudio-3-asr-max"))
    p.add_argument("--white-snr", type=float, default=10.0)
    p.add_argument("--talker-snr", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()
    args.api_key = require("STEPFUN_API_KEY")

    files = sorted(glob.glob(f"{MONO}/*.wav"))
    targets = files[: args.clips]
    interferer = read_wav(files[args.clips]) if len(files) > args.clips else read_wav(files[0])

    conds = ["none", "rnnoise", "webrtc"]
    noises = [("white", args.white_snr), ("talker", args.talker_snr)]
    agg: dict[tuple[str, str], list[float]] = {(nz, c): [] for nz, _ in noises for c in conds}
    dn_ms: dict[str, list[float]] = {c: [] for c in conds}

    print(
        f"cluster={args.cluster} model={args.batch_model} clips={len(targets)} "
        f"white@{args.white_snr}dB talker@{args.talker_snr}dB"
    )
    async with aiohttp.ClientSession() as session:
        for idx, tf in enumerate(targets):
            clean = read_wav(tf)
            ref = await batch_asr(session, args, decimate(clean).tobytes())
            dur_ms = len(clean) / 48.0
            print(f"\n[{tf.split('/')[-1]}] {dur_ms / 1000:.1f}s  ref={ref[:60]!r}")
            for nz, snr in noises:
                noisy = (
                    add_white(clean, snr, seed=idx)
                    if nz == "white"
                    else add_talker(clean, interferer, snr)
                )
                for c in conds:
                    proc, ms = run_denoise(c, noisy)
                    hyp = await batch_asr(session, args, decimate(proc).tobytes())
                    wer = error_counts(ref, hyp)["wer"]
                    agg[(nz, c)].append(wer if wer is not None else 0.0)
                    if c != "none":
                        dn_ms[c].append(ms / dur_ms)  # real-time factor
                    print(
                        f"   {nz:6s} {c:7s} WER={wer * 100:5.1f}%  "
                        f"denoise={ms:6.1f}ms  {hyp[:44]!r}"
                    )

    print("\n=== mean WER% vs clean reference (lower is better) ===")
    print(f"{'noise':8s} " + " ".join(f"{c:>8s}" for c in conds))
    for nz, _ in noises:
        row = " ".join(
            f"{100 * sum(agg[(nz, c)]) / max(1, len(agg[(nz, c)])):7.1f}%" for c in conds
        )
        print(f"{nz:8s} {row}")
    for c in ("rnnoise", "webrtc"):
        if dn_ms[c]:
            print(
                f"  {c}: mean real-time factor {sum(dn_ms[c]) / len(dn_ms[c]):.3f} "
                f"(<1 = faster than realtime)"
            )


if __name__ == "__main__":
    asyncio.run(main())
