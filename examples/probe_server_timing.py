"""Probe raw StepFun responses to discover server-side timing fields.

The LiveKit SDK metrics (LLM ttft, TTS ttfb, STT duration) are measured on the
client clock, so they already include network round-trip time. To separate
network from server processing you need a timestamp or duration the *server*
reports. Standard OpenAI-compatible streaming returns none — but StepFun may add
extra fields, and the speech WebSocket/SSE protocols may carry their own. This
tool talks to each endpoint with raw aiohttp and dumps every server frame
verbatim (large audio/base64 payloads redacted), so you can see exactly which
timing fields the service returns and compare them with the client-side numbers.

Network RTT ~= client_metric - server_metric, whenever a server_metric exists.

Examples:
    uv run python examples/probe_server_timing.py                 # llm + tts + stt
    uv run python examples/probe_server_timing.py --only llm
    uv run python examples/probe_server_timing.py --wav clip.wav  # transcribe your own audio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from array import array
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiohttp
from _config import get, load_env, require

from livekit.plugins.stepfun import _utils
from livekit.plugins.stepfun.models import (
    NUM_CHANNELS,
    PCM_BITS,
    PCM_CODEC,
    STT_RECOGNIZE_PATH,
    STT_STREAM_PATH,
    TTS_STREAM_PATH,
    TTS_SYNTHESIZE_PATH,
)

load_env()

# Standard OpenAI chat-completion streaming schema. Anything outside these sets
# is StepFun-specific and is exactly what we are hunting for.
_OPENAI_TOP = {
    "id",
    "object",
    "created",
    "model",
    "choices",
    "usage",
    "system_fingerprint",
    "service_tier",
}
_OPENAI_USAGE = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "completion_tokens_details",
    "prompt_tokens_details",
}
_OPENAI_DELTA = {
    "role",
    "content",
    "reasoning_content",
    "reasoning",
    "tool_calls",
    "function_call",
    "refusal",
}

DEFAULT_TEXT = "List the numbers 1 through 40 separated by commas, then say done."


def stamp(t0: float) -> str:
    return f"[+{(time.perf_counter() - t0) * 1000:8.1f}ms]"


def redact(obj: Any, limit: int = 120) -> Any:
    """Deep-copy with long strings (base64 audio, etc.) shortened."""
    if isinstance(obj, str):
        return f"<str len={len(obj)}: {obj[:40]!r}...>" if len(obj) > limit else obj
    if isinstance(obj, list):
        return [redact(v, limit) for v in obj]
    if isinstance(obj, dict):
        return {k: redact(v, limit) for k, v in obj.items()}
    return obj


def pretty(obj: Any) -> str:
    return json.dumps(redact(obj), indent=2, ensure_ascii=False)


def hdr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def sse_json(resp: aiohttp.ClientResponse):
    """Yield decoded JSON objects from an SSE body, robust to partial reads."""
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
                print("  (undecodable SSE payload)", payload[:120])


# --- audio helpers ---------------------------------------------------------


def extract_pcm_from_wav(data: bytes) -> tuple[bytes, int, int, int]:
    """Return (pcm_bytes, rate, channels, bits) by walking RIFF chunks."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("payload is not RIFF/WAVE")
    fmt = pcm = None
    off, n = 12, len(data)
    while off + 8 <= n:
        cid = data[off : off + 4]
        size = int.from_bytes(data[off + 4 : off + 8], "little")
        body = data[off + 8 : off + 8 + size]
        if cid == b"fmt ":
            fmt = (
                int.from_bytes(body[0:2], "little"),  # audio format
                int.from_bytes(body[2:4], "little"),  # channels
                int.from_bytes(body[4:8], "little"),  # rate
                int.from_bytes(body[14:16], "little"),
            )  # bits
        elif cid == b"data":
            pcm = body
        off += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("WAV missing fmt or data chunk")
    return pcm, fmt[2], fmt[1], fmt[3]


def downmix_mono16(pcm: bytes, channels: int) -> bytes:
    if channels <= 1:
        return pcm
    a = array("h")
    a.frombytes(pcm[: len(pcm) // 2 * 2])
    out = array(
        "h",
        [sum(a[i : i + channels]) // channels for i in range(0, len(a) - channels + 1, channels)],
    )
    return out.tobytes()


# --- LLM -------------------------------------------------------------------


async def probe_llm(session: aiohttp.ClientSession, args) -> None:
    hdr(f"LLM  ({args.llm_model})  raw streaming chat-completions")
    url = f"{_utils.resolve_llm_base_url(None, args.cluster)}/chat/completions"
    body: dict[str, Any] = {
        "model": args.llm_model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": args.text}],
    }
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort

    headers = {**_utils.auth_headers(args.api_key), "Content-Type": "application/json"}
    t0 = time.perf_counter()
    t_first = t_last = None
    usage: dict[str, Any] | None = None
    n_chunks = 0
    text = ""
    extra_seen: set[str] = set()
    first_content_chunk = None

    async def run(payload: dict[str, Any]) -> None:
        nonlocal t_first, t_last, usage, n_chunks, text, first_content_chunk, extra_seen
        async with session.post(
            url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=args.timeout)
        ) as resp:
            print(
                f"{stamp(t0)} HTTP {resp.status}  content-type={resp.headers.get('content-type')}"
            )
            if resp.status >= 400:
                print("  error body:", (await resp.text())[:500])
                resp.raise_for_status()
            async for ch in sse_json(resp):
                n_chunks += 1
                extra = set(ch) - _OPENAI_TOP
                if extra:
                    extra_seen |= {f"top:{k}" for k in extra}
                choices = ch.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                dextra = set(delta) - _OPENAI_DELTA
                if dextra:
                    extra_seen |= {f"delta:{k}" for k in dextra}
                content = delta.get("content") or ""
                finish = choices[0].get("finish_reason") if choices else None
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                        first_content_chunk = ch
                    t_last = time.perf_counter()
                    text += content
                if ch.get("usage"):
                    usage = ch["usage"]
                    uextra = set(usage) - _OPENAI_USAGE
                    if uextra:
                        extra_seen |= {f"usage:{k}" for k in uextra}
                print(
                    f"{stamp(t0)} #{n_chunks:3d} content={len(content):3d} "
                    f"finish={finish} extra={sorted(extra | dextra) or '-'}"
                )

    try:
        await run(body)
    except aiohttp.ClientResponseError:
        if "reasoning_effort" in body:
            print("  retrying without reasoning_effort (server may reject it)")
            body.pop("reasoning_effort", None)
            await run(body)
        else:
            raise

    ttft_ms = (t_first - t0) * 1000 if t_first else None
    span_ms = (t_last - t_first) * 1000 if t_first and t_last else None
    completion = (usage or {}).get("completion_tokens")
    otps = (completion / (span_ms / 1000)) if completion and span_ms else None

    hdr("LLM summary")
    if ttft_ms:
        print(f"  client TTFT (request -> first content): {ttft_ms:.0f} ms")
    else:
        print("  no content received")
    if span_ms is not None:
        print(f"  generation span (first -> last content): {span_ms:.0f} ms")
    if otps:
        print(
            f"  completion_tokens={completion}  =>  OTPS = {otps:.1f} tok/s "
            f"(client-side; network-independent)"
        )
    if usage:
        print("  usage chunk (raw):")
        print("   ", json.dumps(usage, ensure_ascii=False))
    if first_content_chunk is not None:
        print("  first content chunk (raw):")
        print("   ", json.dumps(redact(first_content_chunk), ensure_ascii=False))
    print(f"  reply: {text.strip()[:200]!r}")
    if extra_seen:
        print(f"  >>> NON-STANDARD fields the server sent: {sorted(extra_seen)}")
        print("      ^ candidate server-side timing; compare against client TTFT above.")
    else:
        print("  server-side timing fields: NONE (standard OpenAI schema only)")
        print(
            "  => client TTFT already includes network; "
            "no server split available from this endpoint."
        )


# --- TTS -------------------------------------------------------------------


async def probe_tts(session: aiohttp.ClientSession, args) -> None:
    hdr(f"TTS  ({args.tts_model} / {args.voice})  raw streaming WebSocket")
    base = _utils.resolve_ws_base_url(None, args.cluster)
    url = f"{base}{TTS_STREAM_PATH}?{urlencode({'model': args.tts_model})}"
    t0 = time.perf_counter()
    extra_seen: set[str] = set()

    async with session.ws_connect(
        url,
        headers=_utils.auth_headers(args.api_key),
        timeout=aiohttp.ClientTimeout(total=args.timeout),
    ) as ws:
        # handshake: wait for tts.connection.done -> session_id
        session_id = None
        while session_id is None:
            ev = json.loads((await ws.receive()).data)
            print(f"{stamp(t0)} <= {ev.get('type')}")
            if ev.get("type") == "tts.connection.done":
                session_id = (ev.get("data") or {}).get("session_id")
                print("   ", pretty(ev))
        # ask for word-level subtitles: these carry SERVER-side audio timings
        await ws.send_str(
            json.dumps(
                {
                    "type": "tts.create",
                    "data": {
                        "session_id": session_id,
                        "voice_id": args.voice,
                        "response_format": "pcm",
                        "sample_rate": 24000,
                        "speed_ratio": 1.0,
                        "volume_ratio": 1.0,
                        "mode": "default",
                        "timestamp": True,
                    },
                }
            )
        )
        t_create = time.perf_counter()
        # wait for tts.response.created
        ts_created = None
        while True:
            ev = json.loads((await ws.receive()).data)
            print(f"{stamp(t0)} <= {ev.get('type')}")
            if ev.get("type") == "tts.response.created":
                ts_created = ev.get("timestamp")
                break
            if ev.get("type") == "tts.response.error":
                print("   ", pretty(ev))
                return
        # push text, then done
        await ws.send_str(
            json.dumps(
                {"type": "tts.text.delta", "data": {"session_id": session_id, "text": args.text}}
            )
        )
        await ws.send_str(json.dumps({"type": "tts.text.done", "data": {"session_id": session_id}}))

        t_first_audio = None
        ts_first_audio = None
        ts_done = None
        n_audio = 0
        audio_bytes = 0
        try:
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=args.timeout)
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    print(f"{stamp(t0)} <= <{msg.type.name}>")
                    break
                ev = json.loads(msg.data)
                etype = ev.get("type")
                data = ev.get("data") or {}
                extra_top = set(ev) - {"type", "event_id", "timestamp", "data"}
                if extra_top:
                    extra_seen |= {f"top:{k}" for k in extra_top}
                extra = set(data) - {
                    "session_id",
                    "audio",
                    "items",
                    "request_id",
                    "text",
                    "start_time",
                    "end_time",
                    "index",
                    "duration",
                    "status",
                    "timestamp",
                }
                if extra:
                    extra_seen |= {f"data:{k}" for k in extra}
                if etype == "tts.response.audio.delta" and data.get("audio"):
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                        ts_first_audio = ev.get("timestamp")
                    n_audio += 1
                    audio_bytes += len(_utils.b64_to_pcm(data["audio"]))
                if etype == "tts.response.audio.done":
                    ts_done = ev.get("timestamp")
                if etype in (
                    "tts.response.subtitle",
                    "tts.response.sentence.start",
                    "tts.response.sentence.end",
                    "tts.response.audio.done",
                ):
                    print(f"{stamp(t0)} <= {etype}")
                    print("   ", pretty(ev))
                if etype == "tts.response.audio.done":
                    break
        except asyncio.TimeoutError:
            print("  (read timeout before audio.done)")

    hdr("TTS summary")
    ttfb_ms = (t_first_audio - t_create) * 1000 if t_first_audio else None
    if ttfb_ms:
        print(f"  client TTFB (tts.create -> first audio.delta): {ttfb_ms:.0f} ms")
    else:
        print("  no audio")
    if ts_created and ts_first_audio:
        print(
            f"  SERVER synthesis (response.created -> first audio, server clock): "
            f"{ts_first_audio - ts_created} ms   <-- network-free server TTFA"
        )
    if ts_created and ts_done:
        print(
            f"  SERVER total (response.created -> audio.done, server clock): "
            f"{ts_done - ts_created} ms"
        )
    if ts_first_audio:
        print(
            f"  first-audio server timestamp: {ts_first_audio} (epoch ms); "
            f"client wall now ~{time.time() * 1000:.0f}"
        )
        print("  => one-way network ~ client_recv_ms - server_ts (needs clock alignment)")
    print(
        f"  audio frames={n_audio}  pcm bytes={audio_bytes}  (~{audio_bytes / 2 / 24000:.2f}s @24k)"
    )
    print("  subtitle events carry SERVER-side word start/end times (ms into the audio).")
    if extra_seen:
        print(f"  >>> other NON-STANDARD fields: {sorted(extra_seen)}")
    else:
        print(
            "  timing source: per-event top-level `timestamp` (server epoch ms) "
            "+ subtitle word times."
        )


# --- STT audio source ------------------------------------------------------


async def get_stt_audio(session, args) -> tuple[bytes, int, int, int] | None:
    if args.wav:
        data = await asyncio.to_thread(Path(args.wav).read_bytes)
        pcm, rate, ch, bits = extract_pcm_from_wav(data)
        print(f"  loaded {args.wav}: rate={rate} ch={ch} bits={bits} pcm={len(pcm)}B")
        return downmix_mono16(pcm, ch) if ch > 1 else pcm, rate, 1, bits
    # synthesize a short clip via the one-shot TTS HTTP route at 16 kHz mono,
    # so STT runs with no external file (mirrors smoke_test.py).
    print("  no --wav given; synthesizing a 16 kHz clip via TTS HTTP for STT input...")
    payload = {
        "model": args.tts_model,
        "voice": args.voice,
        "input": args.text,
        "response_format": "wav",
        "sample_rate": 16000,
        "speed": 1.0,
        "volume": 1.0,
        "text_normalization": "standard",
    }
    url = f"{_utils.resolve_http_base_url(None, args.cluster)}{TTS_SYNTHESIZE_PATH}"
    async with session.post(
        url,
        json=payload,
        headers=_utils.auth_headers(args.api_key),
        timeout=aiohttp.ClientTimeout(total=args.timeout),
    ) as resp:
        if resp.status >= 400:
            print("  TTS synthesize failed:", (await resp.text())[:300])
            return None
        wav = await resp.read()
    pcm, rate, ch, bits = extract_pcm_from_wav(wav)
    print(f"  synthesized clip: rate={rate} ch={ch} bits={bits} pcm={len(pcm)}B")
    return pcm, rate, 1, bits


async def probe_stt_stream(session, args, audio) -> None:
    hdr(f"STT streaming  ({args.stt_model})  raw WebSocket")
    pcm, rate, _ch, bits = audio
    if bits != 16:
        print(f"  expected 16-bit PCM, got {bits}; aborting")
        return
    url = f"{_utils.resolve_ws_base_url(None, args.cluster)}{STT_STREAM_PATH}"
    t0 = time.perf_counter()
    extra_seen: set[str] = set()

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
                                    "rate": rate,
                                    "bits": PCM_BITS,
                                    "channel": NUM_CHANNELS,
                                },
                                "transcription": {
                                    "model": args.stt_model,
                                    "enable_itn": True,
                                    **({"language": args.language} if args.language else {}),
                                },
                                "turn_detection": {
                                    "type": "server_vad",
                                    "silence_duration_ms": 800,
                                    "threshold": 0.6,
                                },
                            }
                        }
                    },
                }
            )
        )
        # wait for session.updated
        while True:
            ev = json.loads((await ws.receive()).data)
            print(f"{stamp(t0)} <= {ev.get('type')}")
            if ev.get("type") == "session.updated":
                break
            if ev.get("type") == "error":
                print("   ", pretty(ev))
                return

        times = {"speech_end": None, "feed_end": None}

        async def feed():
            step = rate // 10 * 2  # 100 ms of mono 16-bit
            for i in range(0, len(pcm), step):
                await ws.send_str(
                    json.dumps(
                        {
                            "event_id": _utils.event_id(),
                            "type": "input_audio_buffer.append",
                            "audio": _utils.pcm_to_b64(pcm[i : i + step]),
                        }
                    )
                )
                await asyncio.sleep(0.1)
            times["speech_end"] = time.perf_counter()
            # trailing silence so server VAD closes the utterance
            silence = b"\x00\x00" * (rate * 2)  # ~1s
            for i in range(0, len(silence), step):
                await ws.send_str(
                    json.dumps(
                        {
                            "event_id": _utils.event_id(),
                            "type": "input_audio_buffer.append",
                            "audio": _utils.pcm_to_b64(silence[i : i + step]),
                        }
                    )
                )
                await asyncio.sleep(0.1)
            times["feed_end"] = time.perf_counter()

        feeder = asyncio.create_task(feed())
        t_completed = None
        try:
            while True:
                msg = await asyncio.wait_for(ws.receive(), timeout=args.timeout)
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    print(f"{stamp(t0)} <= <{msg.type.name}>")
                    break
                ev = json.loads(msg.data)
                etype = ev.get("type")
                extra = set(ev) - {
                    "type",
                    "event_id",
                    "text",
                    "transcript",
                    "item_id",
                    "audio_start_ms",
                    "audio_end_ms",
                    "usage",
                    "meta",
                    "words",
                    "stash",
                    "language",
                }
                if extra:
                    extra_seen |= extra
                if etype in ("conversation.item.input_audio_transcription.completed",):
                    t_completed = time.perf_counter()
                    print(f"{stamp(t0)} <= {etype}")
                    print("   ", pretty(ev))
                    break
                if etype in (
                    "input_audio_buffer.speech_started",
                    "input_audio_buffer.speech_stopped",
                    "conversation.item.input_audio_transcription.delta",
                ):
                    label = (ev.get("transcript") or ev.get("text") or "")[:60]
                    print(f"{stamp(t0)} <= {etype} {label!r}")
                else:
                    print(f"{stamp(t0)} <= {etype}")
        except asyncio.TimeoutError:
            print("  (read timeout before a final transcript)")
        finally:
            await feeder

    hdr("STT streaming summary")
    if t_completed and times["speech_end"]:
        final_ms = (t_completed - times["speech_end"]) * 1000
        print(f"  finalization (speech end -> completed): {final_ms:.0f} ms")
    if extra_seen:
        print(f"  >>> NON-STANDARD event fields: {sorted(extra_seen)}")
    else:
        print("  no non-standard timing fields beyond the documented event schema.")


async def probe_stt_batch(session, args, audio) -> None:
    hdr(f"STT batch  ({args.batch_model})  raw HTTP + SSE")
    pcm, rate, _ch, bits = audio
    if bits != 16:
        print(f"  expected 16-bit PCM, got {bits}; aborting")
        return
    url = f"{_utils.resolve_http_base_url(None, args.cluster)}{STT_RECOGNIZE_PATH}"
    headers = {
        **_utils.auth_headers(args.api_key),
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    transcription = {"model": args.batch_model, "enable_itn": True}
    if args.language:
        transcription["language"] = args.language
    payload = {
        "audio": {
            "data": _utils.pcm_to_b64(pcm),
            "input": {
                "transcription": transcription,
                "format": {
                    "type": "pcm",
                    "codec": PCM_CODEC,
                    "rate": rate,
                    "bits": PCM_BITS,
                    "channel": NUM_CHANNELS,
                },
            },
        }
    }
    t0 = time.perf_counter()
    extra_seen: set[str] = set()
    async with session.post(
        url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as resp:
        print(f"{stamp(t0)} HTTP {resp.status}")
        if resp.status >= 400:
            print("  error body:", (await resp.text())[:400])
            return
        async for ev in sse_json(resp):
            etype = ev.get("type")
            extra = set(ev) - {"type", "delta", "text", "usage", "meta"}
            if extra:
                extra_seen |= extra
            print(f"{stamp(t0)} <= {etype}")
            if etype in ("transcript.text.done", "error"):
                print("   ", pretty(ev))
    hdr("STT batch summary")
    print(f"  client wall time (POST -> done): {(time.perf_counter() - t0) * 1000:.0f} ms")
    if extra_seen:
        print(f"  >>> NON-STANDARD event fields: {sorted(extra_seen)}")
    else:
        print("  no non-standard timing fields beyond the documented event schema.")


async def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--only", choices=["llm", "tts", "stt", "all"], default="all")
    p.add_argument("--cluster", choices=["global", "china"], default=get("STEPFUN_CLUSTER"))
    p.add_argument("--language", default=get("STEPFUN_LANGUAGE", "en"))
    p.add_argument("--llm-model", default=get("STEPFUN_LLM_MODEL", "step-3.7-flash"))
    p.add_argument("--reasoning-effort", default=get("STEPFUN_REASONING_EFFORT", "low"))
    p.add_argument("--tts-model", default=get("STEPFUN_TTS_MODEL", "stepaudio-3-tts"))
    p.add_argument("--voice", default=get("STEPFUN_TTS_VOICE", "vibrant-youth"))
    p.add_argument("--stt-model", default=get("STEPFUN_ASR_MODEL", "stepaudio-2.5-asr-stream"))
    p.add_argument("--batch-model", default=get("STEPFUN_BATCH_ASR_MODEL", "stepaudio-3-asr-max"))
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument(
        "--wav", help="16 kHz mono 16-bit PCM WAV to transcribe (else one is synthesized)"
    )
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()
    args.api_key = require("STEPFUN_API_KEY", hint="ASR, TTS and LLM all use the same key.")

    print(f"cluster={args.cluster}  key=...{args.api_key[-4:]}  timeout={args.timeout}s")
    async with aiohttp.ClientSession() as session:
        if args.only in ("llm", "all"):
            await probe_llm(session, args)
        if args.only in ("tts", "all"):
            await probe_tts(session, args)
        if args.only in ("stt", "all"):
            audio = await get_stt_audio(session, args)
            if audio:
                await probe_stt_stream(session, args, audio)
                await probe_stt_batch(session, args, audio)
            else:
                print("  skipping STT (no audio available)")
    print("\ndone.")


if __name__ == "__main__":
    asyncio.run(main())
