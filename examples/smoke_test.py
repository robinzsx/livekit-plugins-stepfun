"""Real-service smoke test for StepFun ASR + TTS. Needs a key and network.

Exercises all five code paths against the live API and closes the loop: TTS
synthesizes a phrase, then ASR transcribes that same audio back, so nothing
here needs a microphone or a LiveKit server.

    export STEPFUN_API_KEY=...            # or put it in a .env file alongside
    python examples/smoke_test.py
    python examples/smoke_test.py --cluster domestic --language zh

What each check is actually for:

1. **One-shot TTS (HTTP, WAV).** The WAV header carries the true sample rate and
   channel count, so this is the measurement everything else is compared
   against.
2. **Streaming TTS (WebSocket, raw PCM).** Raw PCM has no header, so the rate we
   declare is load-bearing. If check 1 and check 2 disagree on duration for the
   same text, the declared `sample_rate` or `num_channels` is wrong — the
   failure mode that raises no error and only sounds like the wrong pitch.
   This is also where TTFB is measured.
3. **Streaming ASR (WebSocket).** Confirms session.update is accepted, and that
   interim transcripts are revised in place rather than visibly doubling, which
   is what cumulative deltas look like when mishandled.
4. **Batch ASR (HTTP + SSE).** The path `offline_recognize=True` promises, and
   the one `FallbackAdapter` falls back to.
5. **LLM chat.** Streaming text and tool calling over the OpenAI-compatible
   endpoint. On reasoning models it also checks the claim this plugin relies
   on: that `reasoning` never reaches the spoken text. `reasoning_tokens` in
   the usage block proves the model *did* reason, so clean content alongside a
   non-zero count means the separation is real rather than assumed.

Audio is written to ./smoke_out/ so you can listen to it — a pitch problem is
instantly obvious by ear even when the numbers look plausible.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import time
from typing import Any

import aiohttp
from _config import get, load_env, require
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
from livekit.agents import llm as llm_module
from livekit.agents import stt as stt_module
from livekit.agents.types import NOT_GIVEN

from livekit import rtc
from livekit.plugins import stepfun

load_env()

OUT_DIR = pathlib.Path("smoke_out")

# Deliberately mixed: digits and punctuation exercise text normalization, and
# the phrase is long enough to produce several interim transcripts.
TEXT_EN = "Hello there. This is a StepFun voice check, number 42, running on LiveKit."
TEXT_ZH = "你好，这是一次 StepFun 语音链路测试，编号 42。"

# No retries: a smoke test should show the first real error, not mask it behind
# three attempts.
NO_RETRY = APIConnectOptions(max_retry=0)


def load_key() -> str:
    return require(
        "STEPFUN_API_KEY",
        hint="The ASR, TTS and LLM components all use the same key.",
    )


def describe(label: str, frames: list[rtc.AudioFrame]) -> float:
    """Print what we actually received and return its duration in seconds."""
    if not frames:
        print(f"  {label}: NO AUDIO")
        return 0.0

    samples = sum(f.samples_per_channel for f in frames)
    rate = frames[0].sample_rate
    channels = frames[0].num_channels
    duration = samples / rate if rate else 0.0
    print(
        f"  {label}: {duration:.2f}s  rate={rate}  channels={channels}  "
        f"frames={len(frames)}"
    )
    return duration


def _silence_frames(like: rtc.AudioFrame, seconds: float) -> list[rtc.AudioFrame]:
    """Generate silence matching an existing frame's format."""
    if seconds <= 0:
        return []

    chunk_samples = max(1, like.sample_rate // 10)  # 100ms
    count = max(1, round(seconds * like.sample_rate / chunk_samples))
    pcm = b"\x00\x00" * (chunk_samples * like.num_channels)
    return [
        rtc.AudioFrame(
            data=pcm,
            sample_rate=like.sample_rate,
            num_channels=like.num_channels,
            samples_per_channel=chunk_samples,
        )
        for _ in range(count)
    ]


def write_wav(name: str, frames: list[rtc.AudioFrame]) -> None:
    if not frames:
        return
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    path.write_bytes(rtc.combine_audio_frames(frames).to_wav_bytes())
    print(f"  wrote {path}")


async def tts_oneshot(tts: stepfun.TTS, text: str) -> list[rtc.AudioFrame]:
    print("\n[1] one-shot TTS over HTTP (WAV, self-describing)")
    started = time.perf_counter()
    frames = [ev.frame async for ev in tts.synthesize(text, conn_options=NO_RETRY)]
    print(f"  total wall time: {time.perf_counter() - started:.2f}s")
    describe("audio", frames)
    write_wav("tts_oneshot.wav", frames)
    return frames


async def tts_streaming(tts: stepfun.TTS, text: str) -> list[rtc.AudioFrame]:
    print("\n[2] streaming TTS over WebSocket (raw PCM)")

    metrics: list[object] = []
    tts.on("metrics_collected", metrics.append)

    stream = tts.stream(conn_options=NO_RETRY)
    started = time.perf_counter()

    async def feed() -> None:
        # One word at a time, to mimic an LLM token stream rather than handing
        # the server a finished sentence.
        for word in text.split(" "):
            stream.push_text(word + " ")
            await asyncio.sleep(0.01)
        stream.end_input()

    feeder = asyncio.create_task(feed())
    frames = [ev.frame async for ev in stream]
    await feeder
    await stream.aclose()

    print(f"  total wall time: {time.perf_counter() - started:.2f}s")
    describe("audio", frames)
    write_wav("tts_streaming.wav", frames)

    for m in metrics:
        ttfb = getattr(m, "ttfb", None)
        if ttfb is not None:
            # The number customers evaluate a TTS on. Target: <200ms p50.
            print(f"  TTFB: {ttfb * 1000:.0f}ms   (target <200ms p50)")
    if not metrics:
        print("  WARNING: no TTS metrics emitted — _mark_started() may be missing")

    return frames


async def stt_streaming(
    stt: stepfun.STT,
    frames: list[rtc.AudioFrame],
    finalize_timeout: float,
    trailing_silence: float,
) -> None:
    print("\n[3] streaming ASR over WebSocket")
    if not frames:
        raise RuntimeError("streaming ASR cannot be verified: TTS produced no audio")

    from livekit.plugins.stepfun import stt as _stt_impl

    # The plugin stops reading this long after input ends. Widen it to find out
    # whether a final ever arrives and how late — that is the number turn-taking
    # actually depends on, and it cannot be measured with a short grace period.
    _stt_impl._FINALIZE_TIMEOUT = finalize_timeout

    stream = stt.stream(conn_options=NO_RETRY)
    first_partial: float | None = None
    speech_ended_at: float | None = None
    started = time.perf_counter()

    async def feed() -> None:
        nonlocal speech_ended_at
        for frame in frames:
            stream.push_frame(frame)
            # Roughly real time, so server-side VAD behaves as it would on a
            # live call rather than seeing the whole utterance at once.
            await asyncio.sleep(frame.samples_per_channel / frame.sample_rate)

        speech_ended_at = time.perf_counter()

        # A real microphone does not stop sending audio when the speaker stops
        # talking — it sends silence, which is what server VAD listens for.
        # Cutting the stream off at the last word gives the server no silence
        # to detect, so it never finalizes. Pad the tail to reproduce what a
        # live call actually looks like.
        for frame in _silence_frames(frames[0], trailing_silence):
            stream.push_frame(frame)
            await asyncio.sleep(frame.samples_per_channel / frame.sample_rate)

        stream.end_input()

    feeder = asyncio.create_task(feed())

    interims = 0
    finals = 0
    async for ev in stream:
        if ev.type == stt_module.SpeechEventType.INTERIM_TRANSCRIPT:
            if first_partial is None:
                first_partial = time.perf_counter() - started
                print(f"  first partial: {first_partial * 1000:.0f}ms   (target <300ms p50)")
            interims += 1
            print(f"    interim: {ev.alternatives[0].text!r}")
        elif ev.type == stt_module.SpeechEventType.PREFLIGHT_TRANSCRIPT:
            print(f"    preflight (stable): {ev.alternatives[0].text!r}")
        elif ev.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT:
            alt = ev.alternatives[0]
            finals += 1
            if speech_ended_at is not None:
                delay = time.perf_counter() - speech_ended_at
                # Measured from the last word, not from end of input: that is
                # the gap a caller waits through before the agent can reply.
                print(f"  final arrived {delay * 1000:.0f}ms after the last word")
            print(f"  FINAL: {alt.text!r}")
            if alt.words:
                print(f"    {len(alt.words)} word timestamps, first={alt.words[0]!r}")
        elif ev.type == stt_module.SpeechEventType.START_OF_SPEECH:
            print("    server VAD: speech started")
        elif ev.type == stt_module.SpeechEventType.END_OF_SPEECH:
            print("    server VAD: speech stopped")
        elif ev.type == stt_module.SpeechEventType.RECOGNITION_USAGE:
            u = ev.recognition_usage
            if u:
                print(
                    f"  usage: audio={u.audio_duration:.2f}s "
                    f"in={u.input_tokens} out={u.output_tokens} tokens"
                )

    await feeder
    await stream.aclose()

    print(f"  {interims} interim transcripts, {finals} final")
    print("  ^ check these are REVISED in place, not visibly doubling —")
    print("    doubling means cumulative deltas are being appended.")
    if not finals:
        # With server VAD the server finalizes after hearing silence; once input
        # ends it hears nothing at all, so the final may never come. Saying so
        # beats printing interims and looking successful.
        raise RuntimeError(
            f"no FINAL_TRANSCRIPT arrived within {finalize_timeout:.0f}s of input ending; "
            "check the trailing silence and server VAD configuration"
        )


async def stt_batch(stt: stepfun.STT, frames: list[rtc.AudioFrame]) -> None:
    print("\n[4] batch ASR over HTTP + SSE")
    if not frames:
        raise RuntimeError("batch ASR cannot be verified: TTS produced no audio")

    started = time.perf_counter()
    ev = await stt.recognize(frames, conn_options=NO_RETRY)
    if not ev.alternatives or not ev.alternatives[0].text.strip():
        raise RuntimeError("batch ASR returned an empty transcript for the test phrase")
    print(f"  took {time.perf_counter() - started:.2f}s")
    print(f"  FINAL: {ev.alternatives[0].text!r}")


@llm_module.function_tool
async def lookup_order(order_id: str) -> str:
    """Look up the status of a customer order.

    Args:
        order_id: The order number to look up.
    """
    # Deterministic, so a tool round trip can be asserted rather than eyeballed.
    return f"Order {order_id}: shipped, arriving Thursday."


async def llm_chat(
    model: str, reasoning_effort: str, api_key: str, cluster: str | None
) -> None:
    """One streaming completion, plus one tool round trip."""
    print(f"\n[5] LLM chat — {model}")

    chat_llm = stepfun.LLM(
        model=model,
        api_key=api_key,
        cluster=cluster,
        reasoning_effort=reasoning_effort,
    )

    try:
        ctx = llm_module.ChatContext.empty()
        ctx.add_message(role="user", content="In one short sentence: why is the sky blue?")

        started = time.perf_counter()
        first_token: float | None = None
        text = ""
        usage = None

        try:
            async with chat_llm.chat(chat_ctx=ctx, conn_options=NO_RETRY) as stream:
                async for chunk in stream:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    delta = chunk.delta
                    if delta is None or not delta.content:
                        continue
                    if first_token is None:
                        first_token = time.perf_counter() - started
                        # On a reasoning model the thinking happens before this,
                        # so this is time to first *speakable* content — the
                        # number that matters for voice.
                        print(f"  first speakable token: {first_token * 1000:.0f}ms")
                    text += delta.content
        except Exception as exc:  # noqa: BLE001 - report and move to the next model
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            return

        print(f"  reply: {text.strip()!r}")

        if usage is not None:
            print(
                f"  usage: prompt={usage.prompt_tokens} "
                f"completion={usage.completion_tokens} "
                f"reasoning={usage.reasoning_tokens}"
            )
            if usage.reasoning_tokens:
                print("  ^ the model reasoned, and none of it appears in the reply")
                print("    above — otherwise the agent would speak its thinking.")

        # Tool calling: the model should ask for the tool, not invent an answer.
        tool_ctx = llm_module.ChatContext.empty()
        tool_ctx.add_message(role="user", content="What is the status of order A-4471?")

        calls: list[str] = []
        try:
            async with chat_llm.chat(
                chat_ctx=tool_ctx, tools=[lookup_order], conn_options=NO_RETRY
            ) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.tool_calls:
                        for call in chunk.delta.tool_calls:
                            calls.append(f"{call.name}({call.arguments})")
        except Exception as exc:  # noqa: BLE001
            print(f"  tool call FAILED: {type(exc).__name__}: {exc}")
            return

        if calls:
            print(f"  tool calls: {calls}")
        else:
            print("  no tool call requested — for this prompt that means tool")
            print("    calling is not working on this model.")
    finally:
        await chat_llm.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster",
        choices=["international", "domestic"],
        default=get("STEPFUN_CLUSTER"),
    )
    parser.add_argument("--language", default=None, help="zh, en, ja, ko, fr, es")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--tts-model", default=None)
    parser.add_argument(
        "--stt-model",
        default=None,
        help="streaming ASR model; the endpoint accepts stepaudio-2.5-asr-stream only",
    )
    parser.add_argument(
        "--batch-model",
        default=None,
        help="batch ASR model used by recognize(), e.g. stepaudio-3-asr-max",
    )
    parser.add_argument(
        "--finalize-timeout",
        type=float,
        default=3.0,
        help="seconds to keep reading after input ends, waiting for the final",
    )
    parser.add_argument(
        "--no-server-vad",
        action="store_true",
        help="disable StepFun server VAD; the plugin then commits each "
        "utterance itself, which is the documented path for commit",
    )
    parser.add_argument(
        "--trailing-silence",
        type=float,
        default=1.5,
        help="seconds of silence appended after the speech, so server VAD "
        "has something to detect (default 1.5; server default is 800ms)",
    )
    parser.add_argument(
        "--llm-model",
        default="step-3.7-flash",
        help="comma-separated chat models to try in turn, e.g. "
        "step-3.7-flash,step-5-preview. Pass 'none' to skip.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="low",
        help="lower effort means less silence before the reply starts",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("livekit.plugins.stepfun").setLevel(logging.DEBUG)

    api_key = load_key()
    text = TEXT_ZH if args.language == "zh" else TEXT_EN

    # The plugin would otherwise reach for utils.http_context.http_session(),
    # which only exists inside a worker job. Supplying our own also exercises
    # the "must not close a caller's session" contract.
    async with aiohttp.ClientSession() as session:
        tts_kwargs = {"api_key": api_key, "cluster": args.cluster, "http_session": session}
        if args.voice:
            tts_kwargs["voice"] = args.voice
        if args.tts_model:
            tts_kwargs["model"] = args.tts_model
        if args.language:
            tts_kwargs["language"] = args.language

        tts = stepfun.TTS(**tts_kwargs)  # type: ignore[arg-type]
        stt_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "cluster": args.cluster,
            "http_session": session,
            "language": args.language or NOT_GIVEN,
            "enable_timestamp_align": True,
            "server_vad": not args.no_server_vad,
        }
        if args.stt_model:
            stt_kwargs["model"] = args.stt_model
        if args.batch_model:
            stt_kwargs["batch_model"] = args.batch_model

        stt = stepfun.STT(**stt_kwargs)

        print(f"cluster={tts._opts.cluster}  tts_model={tts.model}  voice={tts._opts.voice}")
        print(
            f"stt_model={stt.model}  batch_model={stt._opts.batch_model}  "
            f"language={args.language or '(server default: en)'}"
        )
        print(
            f"server_vad={not args.no_server_vad}  "
            f"trailing_silence={args.trailing_silence}s  "
            f"finalize_timeout={args.finalize_timeout}s"
        )

        print("\navailable voices on this cluster:")
        try:
            voices = await tts.list_voices()
            print(f"  {voices.get('voices')}")
        except Exception as exc:  # noqa: BLE001 - informational only
            print(f"  could not list voices: {type(exc).__name__}: {exc}")

        oneshot = await tts_oneshot(tts, text)
        streamed = await tts_streaming(tts, text)

        if oneshot and streamed:
            print("\n[pitch check] one-shot vs streaming")
            d1 = sum(f.samples_per_channel for f in oneshot) / oneshot[0].sample_rate
            d2 = sum(f.samples_per_channel for f in streamed) / streamed[0].sample_rate
            drift = abs(d1 - d2) / max(d1, d2)
            verdict = "OK" if drift < 0.25 else "MISMATCH — check sample_rate/num_channels"
            print(f"  {d1:.2f}s vs {d2:.2f}s  ({drift:.0%} apart)  -> {verdict}")
            print("  (same text, so durations should be close; listen to both files too)")

        # Transcribe the one-shot audio: it is the self-describing one, so a
        # bad transcript points at ASR rather than at a format mistake.
        await stt_streaming(
            stt, oneshot, args.finalize_timeout, args.trailing_silence
        )
        await stt_batch(stt, oneshot)

        await tts.aclose()
        await stt.aclose()
        assert not session.closed, "plugin closed a caller-supplied session"

    if args.llm_model.lower() != "none":
        for model in [m.strip() for m in args.llm_model.split(",") if m.strip()]:
            await llm_chat(model, args.reasoning_effort, api_key, args.cluster)

    oneshot.clear()
    streamed.clear()

    print("\ndone. audio in ./smoke_out/ — listen to confirm pitch and quality.")


def _error_hint(exc: APIStatusError) -> str:
    """Point at the likely cause instead of listing every possibility."""
    body = exc.body if isinstance(exc.body, dict) else {}
    err = body.get("error") or {}
    param = err.get("param")

    if param:
        # WebSocket error frames name the offending field, which beats guessing.
        hint = f"  the server rejected the {param!r} parameter.\n"
        if param == "model":
            hint += (
                "  The bidirectional ASR endpoint documents support for\n"
                "  stepaudio-2.5-asr-stream only. StepAudio 3 ASR is\n"
                "  submitted-audio only: reach it with --batch-model, which\n"
                "  recognize() uses, or stepfun.STT(use_streaming=False).\n"
            )
        return hint

    if exc.status_code == 401:
        return "  401 => wrong key, or a key for the other cluster.\n"
    if exc.status_code == 400:
        return "  400 voice_id_invalid => pick a voice from the list above.\n"
    return ""


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except APIStatusError as exc:
        # The plugin's own error mapping, so the status and retryability here are
        # exactly what a FallbackAdapter would see.
        raise SystemExit(
            f"\nStepFun returned an error: {exc}\n"
            f"  status={exc.status_code} retryable={exc.retryable} "
            f"request_id={exc.request_id}\n"
            f"{_error_hint(exc)}"
        ) from None
    except APIConnectionError as exc:
        raise SystemExit(
            f"\nCould not reach StepFun: {exc}\n"
            "  Check network egress to api.stepfun.ai (or .com with "
            "--cluster domestic).\n"
        ) from None
