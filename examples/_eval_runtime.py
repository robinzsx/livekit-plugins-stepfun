"""Optional local evaluation service attached to a running AgentSession."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import importlib.metadata
import json
import logging
import math
import platform
import time
from typing import Any

from _eval_audio import SAMPLE_RATE, Playback, ReplayAudioInput, decode_audio
from _eval_metrics import TurnRecord
from aiohttp import web
from livekit.agents import Agent, llm, stt

logger = logging.getLogger("stepfun-eval")
INSTRUCTIONS = (
    "You are a concise, friendly voice assistant. Keep answers "
    "conversational and short enough to listen to."
)


def _content(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    delta = getattr(chunk, "delta", None)
    return getattr(delta, "content", "") or ""


def _versions() -> dict[str, str | None]:
    versions = {}
    for name in (
        "livekit-agents",
        "livekit",
        "livekit-plugins-stepfun",
        "livekit-plugins-openai",
        "livekit-plugins-silero",
        "livekit-local-inference",
        "aiohttp",
        "av",
        "numpy",
        "pyrnnoise",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class InstrumentedAgent(Agent):
    """Observe public agent nodes without changing provider implementations."""

    def __init__(self, *, evaluation=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._evaluation = evaluation

    async def stt_node(self, audio, model_settings):
        async with contextlib.aclosing(
            Agent.default.stt_node(self, audio, model_settings)
        ) as stream:
            async for event in stream:
                if self._evaluation:
                    self._evaluation.observe_stt(event)
                yield event

    async def on_user_turn_completed(self, turn_ctx, new_message):
        if self._evaluation and self._evaluation.active:
            self._evaluation.active.note("turn_decision")
        await super().on_user_turn_completed(turn_ctx, new_message)

    async def llm_node(self, chat_ctx, tools, model_settings):
        record = self._evaluation.active if self._evaluation else None
        request = record.begin_request("llm") if record else None
        try:
            async with contextlib.aclosing(
                Agent.default.llm_node(self, chat_ctx, tools, model_settings)
            ) as stream:
                async for chunk in stream:
                    if (
                        request is not None
                        and _content(chunk).strip()
                        and "first_content" not in request
                    ):
                        request["first_content"] = time.perf_counter()
                        record.note("llm_first_content")
                    yield chunk
            if request is not None:
                request["cancelled"] = False
        finally:
            if request is not None:
                request["end"] = time.perf_counter()

    async def tts_node(self, text, model_settings):
        record = self._evaluation.active if self._evaluation else None
        request = record.begin_request("tts") if record else None

        async def observed_text():
            async for chunk in text:
                if request is not None and chunk.strip() and "first_text" not in request:
                    request["first_text"] = time.perf_counter()
                    record.note("tts_first_text")
                yield chunk

        try:
            async with contextlib.aclosing(
                Agent.default.tts_node(self, observed_text(), model_settings)
            ) as stream:
                async for frame in stream:
                    if request is not None and "first_audio" not in request:
                        request["first_audio"] = time.perf_counter()
                        record.note("tts_first_audio")
                    yield frame
            if request is not None:
                request["cancelled"] = False
        finally:
            if request is not None:
                request["end"] = time.perf_counter()


class EvaluationController:
    def __init__(
        self,
        session,
        *,
        config: dict[str, Any],
        providers: dict[str, Any],
        max_consecutive_failures: int = 3,
    ) -> None:
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least 1")
        self.session = session
        self.providers = providers
        self._max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self.config = {
            **config,
            "instructions": INSTRUCTIONS,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": _versions(),
        }
        self.audio = ReplayAudioInput()
        self.active: TurnRecord | None = None
        self._speech_records: dict[str, TurnRecord] = {}
        self._handles: list[Any] = []
        self._changed = asyncio.Event()
        self._lock = asyncio.Lock()
        self._runner: web.AppRunner | None = None
        self._request_task: asyncio.Task | None = None
        self._failed: str | None = None
        self._closing = False
        self._evaluation_count = 0
        session.on("speech_created", self._speech_created)
        session.on("agent_state_changed", self._state_changed)
        session.on("conversation_item_added", self._conversation_item)
        session.on("error", self._error)
        # Component listeners also work when the component is benchmarked outside the session.
        for provider in providers.values():
            provider.on("metrics_collected", self._metrics)
        session.on("metrics_collected", self._session_metric)

    def _speech_created(self, event) -> None:
        if self.active is None:
            return
        handle = event.speech_handle
        self._handles.append(handle)
        self._speech_records[handle.id] = self.active
        self.active.note("speech_created", speech_id=handle.id)
        handle.add_done_callback(lambda _: self._changed.set())
        self._changed.set()

    def _state_changed(self, event) -> None:
        if self.active:
            self.active.note("agent_state", state=event.new_state)
        self._changed.set()

    def _conversation_item(self, event) -> None:
        if self.active is None:
            return
        item = event.item
        if getattr(item, "role", None) == "assistant":
            self.active.responses.append(item.text_content or "")
        self.active.note(
            "conversation_item",
            role=getattr(item, "role", None),
            text=getattr(item, "text_content", None),
            metrics=getattr(item, "metrics", None),
        )
        self._changed.set()

    def _error(self, event) -> None:
        error = {
            "message": str(event.error),
            "recoverable": bool(getattr(event.error, "recoverable", False)),
        }
        if self.active:
            self.active.errors.append(error)
            self.active.note("provider_error", **error)
        elif not error["recoverable"] and not self._closing:
            self._failed = (
                f"Provider unavailable before replay: {error['message']}. Restart the agent."
            )
        self._changed.set()

    def _session_metric(self, event) -> None:
        if getattr(event.metrics, "type", "") in (
            "eou_metrics",
            "eot_inference_metrics",
            "vad_metrics",
        ):
            self._metrics(event.metrics)

    def _metrics(self, metric) -> None:
        record = self._speech_records.get(getattr(metric, "speech_id", None)) or self.active
        if record is not None:
            payload = metric.model_dump(mode="json")
            record.sdk_metrics.append(payload)
            record.note("sdk_metrics", metrics=payload)

    def observe_stt(self, event) -> None:
        if self.active is None or not event.alternatives:
            return
        text = event.alternatives[0].text
        if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            self.active.note("asr_partial", text=text)
        elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            self.active.transcripts.append(text)
            self.active.note("asr_final", text=text)
        self._changed.set()

    async def start(self, port: int) -> None:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_get("/health", self._health)
        app.router.add_post("/evaluate", self._evaluate)
        self._runner = web.AppRunner(app, shutdown_timeout=1)
        try:
            await self._runner.setup()
            await web.TCPSite(self._runner, "127.0.0.1", port).start()
        except BaseException:
            await self._runner.cleanup()
            self._runner = None
            raise
        logger.info("Dataset evaluation ready at http://127.0.0.1:%s (microphone replaced)", port)

    async def _health(self, request) -> web.Response:
        return web.json_response(
            {
                "ready": not self._closing and not self._failed,
                "busy": self._lock.locked(),
                "error": self._failed,
                "consecutive_failures": self._consecutive_failures,
                "config": self.config,
            }
        )

    async def _evaluate(self, request) -> web.Response:
        if self._closing or self._failed:
            raise web.HTTPConflict(text=self._failed or "Agent is shutting down")
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType(text="Expected application/json")
        source = self.session.input.audio
        while source is not None and source is not self.audio:
            source = source.source
        if source is None or self.session.output.audio is None:
            raise web.HTTPConflict(text="Console audio mode changed; restart the evaluation agent")
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            mode = payload.get("mode", "agent")
            if mode not in ("agent", "stt", "llm", "tts"):
                raise ValueError("mode must be agent, stt, llm, or tts")
            timeout = float(payload.get("response_timeout", 120))
            if not math.isfinite(timeout) or not 1 <= timeout <= 600:
                raise ValueError("response_timeout must be between 1 and 600 seconds")
            pcm = b""
            if mode in ("agent", "stt"):
                encoded = payload.get("audio_b64", "")
                pcm = await asyncio.to_thread(
                    decode_audio, base64.b64decode(encoded, validate=True)
                )
            text = payload.get("text") or payload.get("reference")
            if mode in ("llm", "tts") and (not isinstance(text, str) or not text.strip()):
                raise ValueError("LLM/TTS benchmarks require text or a reference transcript")
            if len(str(text or "")) > 20000:
                raise ValueError("Text exceeds 20000 characters")
            start, end = payload.get("speech_start_s"), payload.get("speech_end_s")
            if (start is None) != (end is None):
                raise ValueError("Provide both speech_start_s and speech_end_s, or neither")
            if start is not None:
                start, end = float(start), float(end)
                if not 0 <= start < end <= len(pcm) / (SAMPLE_RATE * 2):
                    raise ValueError("Speech annotations must lie within the audio duration")
            if not isinstance(payload.get("reset_history", True), bool):
                raise ValueError("reset_history must be a boolean")
            keywords = payload.get("expected_keywords", [])
            if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
                raise ValueError("expected_keywords must be a list of strings")
            reference = payload.get("reference")
            if reference is not None and not isinstance(reference, str):
                raise ValueError("reference must be a string")
        except (ValueError, TypeError, binascii.Error, json.JSONDecodeError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        metadata = {key: value for key, value in payload.items() if key != "audio_b64"}
        metadata.update(
            mode=mode,
            audio_duration_s=len(pcm) / (SAMPLE_RATE * 2),
            boundary_source="annotated" if start is not None else "file",
        )
        async with self._lock:
            if self._failed or self._closing:
                raise web.HTTPConflict(text=self._failed or "Agent is shutting down")
            self._request_task = asyncio.current_task()
            self._evaluation_count += 1
            metadata["agent_evaluation_index"] = self._evaluation_count
            record = self.active = TurnRecord(str(payload.get("sample_id", "sample")))
            self._handles = []
            status = "ok"
            try:
                if mode == "agent" and payload.get("reset_history", True):
                    await self.session.current_agent.update_chat_ctx(llm.ChatContext.empty())
                record.note("evaluation_start")
                await asyncio.wait_for(
                    self._run(mode, pcm, text, record, start, end),
                    timeout=timeout + len(pcm) / (SAMPLE_RATE * 2),
                )
                # Let metrics monitors run before freezing this turn's record.
                await asyncio.sleep(0)
                record.note("response_done")
            except (TimeoutError, asyncio.TimeoutError):
                status = "timeout"
                record.errors.append(
                    {
                        "message": "Timed out waiting for the response to finish",
                        "recoverable": False,
                    }
                )
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exc:
                status = "error"
                record.errors.append({"message": str(exc), "recoverable": False})
            finally:
                if status != "ok":
                    record.note("evaluation_failed", status=status)
                    self.audio.clear()
                    # A failed turn must not take the agent down with it: the
                    # failure is already on the record, so drop queued speech
                    # and keep serving. Only a run of consecutive failures
                    # means the agent itself is broken, and only then does the
                    # endpoint refuse further work.
                    if not self._closing:
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(self.session.interrupt(force=True), 5)
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= self._max_consecutive_failures:
                            self._failed = (
                                f"{self._consecutive_failures} consecutive evaluations "
                                "failed; restart the agent before continuing"
                            )
                elif not self._closing:
                    self._consecutive_failures = 0
                self.active = None
                self._request_task = None
                self._speech_records.clear()
            return web.json_response(record.result(status=status, metadata=metadata))

    async def _run(self, mode, pcm, text, record, start, end) -> None:
        if mode == "agent":
            playback = Playback(pcm, record, asyncio.Event(), start, end)
            self.audio.playback = playback
            await playback.finished.wait()
            await self._wait_response(record)
        elif mode == "stt":
            await self._benchmark_stt(pcm, record, start, end)
        elif mode == "llm":
            context = llm.ChatContext.empty()
            context.add_message(role="system", content=INSTRUCTIONS)
            context.add_message(role="user", content=text)
            probe = record.begin_request("llm")
            response = ""
            try:
                async with self.providers["llm"].chat(chat_ctx=context) as stream:
                    async for chunk in stream:
                        content = _content(chunk)
                        if content.strip() and "first_content" not in probe:
                            probe["first_content"] = time.perf_counter()
                            record.note("llm_first_content")
                        response += content
                probe["cancelled"] = False
                record.responses.append(response)
                if not response.strip():
                    raise RuntimeError("LLM returned no speakable content")
            finally:
                probe["end"] = time.perf_counter()
        else:
            probe = record.begin_request("tts")
            try:
                async with self.providers["tts"].stream() as stream:
                    probe["first_text"] = time.perf_counter()
                    record.note("tts_first_text")
                    stream.push_text(text)
                    stream.end_input()
                    async for _ in stream:
                        if "first_audio" not in probe:
                            probe["first_audio"] = time.perf_counter()
                            record.note("tts_first_audio")
                probe["cancelled"] = False
            finally:
                probe["end"] = time.perf_counter()

    async def _wait_response(self, record) -> None:
        while True:
            self._changed.clear()
            if any(not error.get("recoverable") for error in record.errors):
                raise RuntimeError("Provider failed; see recorded error events")
            completed = [
                handle for handle in self._handles if handle.done() and not handle.interrupted
            ]
            if (
                completed
                and all(handle.done() for handle in self._handles)
                and "output_first_audio" in record.marks
                and "asr_final" in record.marks
                and self.session.agent_state in ("idle", "listening")
            ):
                for handle in completed:
                    if handle.exception():
                        raise handle.exception()
                await self.session.output.audio.wait_for_playout()
                return
            await self._changed.wait()

    async def _benchmark_stt(self, pcm, record, start, end) -> None:
        provider = self.providers["stt"]
        if not provider.capabilities.streaming:
            from livekit import rtc

            record.note("input_start")
            # All samples are available immediately in this isolated batch benchmark.
            record.note("input_end")
            event = await provider.recognize(rtc.AudioFrame(pcm, SAMPLE_RATE, 1, len(pcm) // 2))
            self.observe_stt(event)
            return
        feed = ReplayAudioInput()
        playback = Playback(pcm, record, asyncio.Event(), start, end)
        feed.playback = playback
        final = asyncio.Event()
        async with provider.stream() as stream:

            async def send():
                async for frame in feed:
                    stream.push_frame(frame)

            async def receive():
                async for event in stream:
                    self.observe_stt(event)
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        final.set()

            tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
            try:

                async def completed():
                    await playback.finished.wait()
                    await final.wait()

                completion = asyncio.create_task(completed())
                tasks.append(completion)
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in tasks[:2]:
                    if task.done():
                        task.result()
                        raise RuntimeError("STT stream ended before recognition completed")
                await completion
            finally:
                feed.close()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        self._closing = True
        self.audio.close()
        if self._request_task is not None and self._request_task is not asyncio.current_task():
            self._request_task.cancel()
            await asyncio.gather(self._request_task, return_exceptions=True)
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
