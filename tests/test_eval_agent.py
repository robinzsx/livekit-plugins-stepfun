"""Dataset replay, playout gating, and latency attribution without provider calls."""

from __future__ import annotations

import asyncio
import base64
import io as bytes_io
import json
import sys
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    AgentSession,
    APIConnectionError,
    llm,
    tts,
)
from livekit.agents.voice import io

from livekit.plugins import stepfun

from .fake_ws import FakePool, FakeWebSocket
from .test_stepfun_session import ScriptedTurnDetector, ScriptedVAD
from .test_stepfun_stt import _FakeSSESession

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from _eval_audio import (  # noqa: E402
    ObservedAudioOutput,
    Playback,
    ReplayAudioInput,
    decode_audio,
)
from _eval_metrics import TurnRecord, error_counts, render_report, summarize  # noqa: E402
from _eval_runtime import EvaluationController, InstrumentedAgent  # noqa: E402
from eval_agent import compare_runs, load_dataset, run_dataset, write_artifacts  # noqa: E402


def wav_bytes(*, rate=16000, channels=1, seconds=0.12):
    buffer = bytes_io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\xf4\x01" * round(rate * seconds) * channels)
    return buffer.getvalue()


@pytest.mark.parametrize("rate,channels", [(16000, 1), (24000, 1), (48000, 2)])
def test_decode_normalizes_audio_without_changing_duration(rate, channels):
    pcm = decode_audio(wav_bytes(rate=rate, channels=channels))
    assert len(pcm) == 1920 * 2


def test_decoder_rejects_non_audio():
    with pytest.raises(ValueError, match="could not be decoded"):
        decode_audio(b"not audio")


async def test_replay_is_paced_and_keeps_sending_silence():
    source = ReplayAudioInput()
    record = TurnRecord("file")
    playback = Playback(b"\x01\x00" * 800, record, asyncio.Event(), 0.0, 0.05)
    source.playback = playback
    started = time.perf_counter()
    frames = [await anext(source) for _ in range(4)]
    assert time.perf_counter() - started >= 0.045
    assert playback.finished.is_set()
    assert sum(frame.samples_per_channel for frame in frames[:3]) == 800
    assert not any(frames[-1].data)
    assert record.marks["input_end"] >= record.marks["input_start"] + 0.045
    assert record.marks["speech_end"] >= record.marks["speech_start"] + 0.045
    source.close()
    with pytest.raises(StopAsyncIteration):
        await anext(source)


async def test_file_end_uses_audio_clock_when_consumer_is_delayed():
    source = ReplayAudioInput()
    record = TurnRecord("file")
    source.playback = Playback(b"\x01\x00" * 320, record, asyncio.Event())
    await anext(source)
    await asyncio.sleep(0.05)
    await anext(source)
    assert record.delta_ms("input_end", "input_start") == pytest.approx(20, abs=2)
    source.close()


def test_dataset_folder_and_manifest(tmp_path):
    folder = tmp_path / "mono"
    folder.mkdir()
    for name in ("002.WAV", "001.wav"):
        (folder / name).write_bytes(wav_bytes())
    (folder / "ignore.txt").write_text("ignore")
    entries = load_dataset(folder)
    assert [entry["id"] for entry in entries] == ["001.wav", "002.WAV"]
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(json.dumps({"audio": "mono/001.wav", "reference": "hello"}) + "\n")
    assert load_dataset(manifest)[0]["audio"] == str(folder / "001.wav")
    assert load_dataset(folder, limit=1) == entries[:1]
    with pytest.raises(ValueError, match="No WAV"):
        load_dataset(tmp_path)


def test_metrics_preserve_missing_values_and_exclude_cancelled_requests():
    record = TurnRecord("sample", origin=10)
    record.note("input_start", at=10)
    record.note("input_end", at=12)
    record.note("asr_partial", at=10.3)
    record.note("asr_final", at=12.4)
    record.requests = [
        {"component": "llm", "start": 10, "first_content": 10.1, "cancelled": True},
        {"component": "llm", "start": 12.4, "first_content": 12.9, "end": 13, "cancelled": False},
    ]
    record.sdk_metrics = [{"type": "stt_metrics", "duration": 0, "streamed": True}]
    row = record.result(status="ok", metadata={})["row"]
    assert row["input_to_first_partial_ms"] == pytest.approx(300)
    assert row["input_end_to_final_ms"] == pytest.approx(400)
    assert row["llm_first_content_ms"] == pytest.approx(500)
    assert row["speech_end_to_final_ms"] is None
    assert row["stt_batch_request_ms"] is None


def test_quality_and_summary_denominators():
    assert error_counts("Hello, WORLD!", "hello world")["wer"] == 0
    assert error_counts("你好世界", "你好世")["cer"] == pytest.approx(0.25)
    assert error_counts("", "extra")["cer"] is None
    results = [
        {"row": {"status": "ok", "latency_ms": 10}},
        {"row": {"status": "ok", "latency_ms": 30}},
        {"row": {"status": "timeout", "latency_ms": 9999}},
    ]
    summary = summarize(results)
    assert summary["failure_rate"] == pytest.approx(1 / 3)
    assert summary["latencies"]["latency_ms"] == {"n": 2, "p50": 20, "p95": 30}


def test_artifacts_and_comparison_escape_dataset_text(tmp_path):
    result = TurnRecord("<script>bad</script>").result(status="ok", metadata={})
    run = {"label": "<svg onload=alert(1)>", "config": {}, "results": [result]}
    write_artifacts(tmp_path, run)
    (tmp_path / "run.json").write_text(json.dumps({"label": run["label"], "config": {}}))
    (tmp_path / "results.jsonl").write_text(json.dumps(result) + "\n")
    comparison = tmp_path / "comparison.html"
    compare_runs([tmp_path], comparison)
    report = comparison.read_text()
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "&lt;svg" in report
    assert (tmp_path / "turns.csv").is_file()
    assert "not the acoustic speaker" in render_report([run])


@pytest.mark.parametrize("recoverable", [True, False])
async def test_readiness_reports_provider_failure_before_first_replay(recoverable):
    controller = EvaluationController(MagicMock(), config={}, providers={})
    controller._error(SimpleNamespace(error=SimpleNamespace(recoverable=recoverable)))
    health = json.loads((await controller._health(None)).text)
    assert health["ready"] is recoverable
    assert bool(health["error"]) is not recoverable


class ScriptedLLM(llm.LLM):
    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kwargs):
        return ScriptedLLMStream(
            llm=self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class ScriptedLLMStream(llm.LLMStream):
    async def _run(self):
        self._event_ch.send_nowait(
            llm.ChatChunk(id="test", delta=llm.ChoiceDelta(role="assistant"))
        )
        await asyncio.sleep(0.01)
        self._event_ch.send_nowait(
            llm.ChatChunk(id="test", delta=llm.ChoiceDelta(role="assistant", content="Hello back."))
        )
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="test",
                usage=llm.CompletionUsage(
                    prompt_tokens=5,
                    completion_tokens=3,
                    total_tokens=8,
                ),
            )
        )


class ScriptedTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True), sample_rate=16000, num_channels=1
        )

    def stream(self, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return ScriptedTTSStream(tts=self, conn_options=conn_options)

    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return tts.ChunkedStreamAdapter(tts=self, input_text=text, conn_options=conn_options)


class ScriptedTTSStream(tts.SynthesizeStream):
    async def _run(self, output_emitter):
        output_emitter.initialize(
            request_id="test", sample_rate=16000, num_channels=1, mime_type="audio/pcm", stream=True
        )
        output_emitter.start_segment(segment_id="test")
        async for chunk in self._input_ch:
            if isinstance(chunk, str):
                self._mark_started()
        output_emitter.push(b"\x01\x00" * 1600)
        output_emitter.end_segment()


class FailingTTS(ScriptedTTS):
    def stream(self, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return FailingTTSStream(tts=self, conn_options=conn_options)


class FailingTTSStream(tts.SynthesizeStream):
    """A socket that dies mid-generation, after audio has already started.

    This is the case the framework deliberately does not retry — the user
    already heard the partial reply — and therefore the case that has to be
    survivable instead. The sleep matters: the emitter records pushed audio
    asynchronously, so raising immediately would race it and look like a
    pre-audio failure (which *is* retried).
    """

    async def _run(self, output_emitter):
        output_emitter.initialize(
            request_id="test", sample_rate=16000, num_channels=1, mime_type="audio/pcm", stream=True
        )
        output_emitter.start_segment(segment_id="test")
        async for chunk in self._input_ch:
            if isinstance(chunk, str):
                self._mark_started()
        output_emitter.push(b"\x01\x00" * 1600)
        await asyncio.sleep(0.05)
        raise APIConnectionError(
            "stepfun tts: websocket closed unexpectedly (close_code=1006, "
            "reason='connection reset')"
        )


class DelayedOutput(io.AudioOutput):
    def __init__(self):
        super().__init__(label="test speaker", capabilities=io.AudioOutputCapabilities(pause=False))
        self.completed = 0
        self.pending: asyncio.Task | None = None
        self.played_at = 0.0

    async def capture_frame(self, frame):
        await super().capture_frame(frame)

    def flush(self):
        super().flush()
        self.pending = asyncio.create_task(self._finish())

    async def _finish(self):
        await asyncio.sleep(0.1)
        self.completed += 1
        self.played_at = time.perf_counter()
        self.on_playback_finished(playback_position=0.1, interrupted=False)

    def clear_buffer(self):
        super().flush()
        if self.pending is not None:
            self.pending.cancel()
        self.on_playback_finished(playback_position=0, interrupted=True)


class ScriptedSocket(FakeWebSocket):
    def __init__(self):
        super().__init__()
        self.queue_server_event({"type": "session.updated"})
        self.speaking = False
        self.item = 0

    async def send_str(self, data):
        await super().send_str(data)
        event = json.loads(data)
        if event["type"] != "input_audio_buffer.append":
            return
        speaking = any(base64.b64decode(event["audio"]))
        if speaking and not self.speaking:
            self.speaking = True
            self.item += 1
            self.queue_server_event({"type": "input_audio_buffer.speech_started"})
            self.queue_server_event(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": str(self.item),
                    "text": "hello",
                }
            )
        elif not speaking and self.speaking:
            self.speaking = False
            self.queue_server_event({"type": "input_audio_buffer.speech_stopped"})
            self.queue_server_event(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": str(self.item),
                    "transcript": "hello",
                }
            )


class ScriptedPool(FakePool):
    def __init__(self):
        super().__init__(None)

    @asynccontextmanager
    async def connection(self, *, timeout):
        yield ScriptedSocket()


class Request:
    content_type = "application/json"

    def __init__(self, **payload):
        self.payload = payload

    async def json(self):
        return self.payload


@pytest.fixture
async def running_eval():
    resources = []

    async def create(streaming=True, tts_provider=None):
        stt = stepfun.STT(
            use_streaming=streaming,
            http_session=_FakeSSESession(['data: {"type":"transcript.text.done","text":"hello"}']),
        )
        if streaming:
            stt._pool = ScriptedPool()
        chat = ScriptedLLM()
        speech = tts_provider or ScriptedTTS()
        detector = ScriptedTurnDetector()
        detector.allow_completion.set()
        session = AgentSession(
            stt=stt,
            llm=chat,
            tts=speech,
            vad=ScriptedVAD(),
            aec_warmup_duration=None,
            turn_handling={
                "turn_detection": "stt" if streaming else detector,
                "endpointing": {"min_delay": 0},
                "interruption": {"mode": "vad"},
            },
        )
        controller = EvaluationController(
            session,
            config={"streaming": streaming},
            providers={"stt": stt, "llm": chat, "tts": speech},
        )
        output = DelayedOutput()
        session.input.audio = controller.audio
        session.output.audio = ObservedAudioOutput(output, get_record=lambda: controller.active)
        agent = InstrumentedAgent(instructions="Test", evaluation=controller)
        await session.start(agent=agent)
        resources.append((session, controller, stt, output))
        return controller, output

    yield create
    for session, controller, stt, output in resources:
        await controller.aclose()
        await session.aclose()
        await stt.aclose()
        if output.pending is not None:
            await asyncio.gather(output.pending, return_exceptions=True)


@pytest.mark.parametrize("streaming", [True, False])
async def test_two_clips_wait_for_actual_playout_and_keep_session_running(running_eval, streaming):
    controller, output = await running_eval(streaming)
    for index in (1, 2):
        request = Request(
            audio_b64=base64.b64encode(wav_bytes()).decode(),
            sample_id=str(index),
            response_timeout=3,
            speech_start_s=0,
            speech_end_s=0.12,
            reference="hello",
        )
        response = await asyncio.wait_for(controller._evaluate(request), 5)
        result = json.loads(response.text)
        assert result["row"]["status"] == "ok", result
        assert output.completed == index
        assert controller.session.agent_state == "listening"
        assert result["row"]["transcript"] == "hello"
        assert "Hello back" in result["row"]["response"]
        assert result["row"]["llm_first_content_ms"] >= 5
        assert result["row"]["tts_first_audio_ms"] is not None
        assert result["row"]["response_playout_ms"] >= 90
        assert result["row"]["speech_end_to_first_output_ms"] is not None
        assert result["row"]["cer"] == 0
        assert result["row"]["llm_prompt_tokens"] == 5
        assert result["row"]["agent_evaluation_index"] == index
        assert controller.active is None
        assert controller._failed is None


async def test_waiter_ignores_cancelled_speculative_speech(running_eval):
    controller, _ = await running_eval()
    record = TurnRecord("test")
    controller._handles = [SimpleNamespace(done=lambda: True, interrupted=True)]
    record.note("output_first_audio")
    task = asyncio.create_task(controller._wait_response(record))
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_bad_audio_and_annotations_do_not_poison_agent(running_eval):
    from aiohttp import web

    controller, _ = await running_eval()
    with pytest.raises(web.HTTPBadRequest):
        await controller._evaluate(Request(audio_b64="invalid base64"))
    with pytest.raises(web.HTTPBadRequest) as invalid:
        await controller._evaluate(
            Request(
                audio_b64=base64.b64encode(wav_bytes()).decode(), speech_start_s=0, speech_end_s=100
            )
        )
    assert "annotations" in invalid.value.text
    assert controller._failed is None


@pytest.mark.parametrize("mode", ["llm", "tts", "stt"])
async def test_component_benchmarks(running_eval, mode):
    controller, output = await running_eval()
    response = await asyncio.wait_for(
        controller._evaluate(
            Request(
                mode=mode,
                text="Hello",
                audio_b64=base64.b64encode(wav_bytes()).decode(),
                response_timeout=3,
            )
        ),
        5,
    )
    result = json.loads(response.text)
    assert result["row"]["status"] == "ok", result
    assert result["row"]["input_end_to_first_output_ms"] is None
    assert output.completed == 0


async def test_timeout_is_recorded_and_the_agent_keeps_serving(running_eval, monkeypatch):
    """One failed trial must not disable the endpoint.

    The timeout is on the record with its error, and the very next trial runs
    normally against the same agent — that is the whole point of recording
    failures instead of stopping on them.
    """
    controller, _ = await running_eval()

    async def blocked(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(controller, "_run", blocked)
    response = await controller._evaluate(Request(mode="llm", text="Hello", response_timeout=1))
    result = json.loads(response.text)
    assert result["row"]["status"] == "timeout"
    assert result["errors"]
    assert controller.active is None
    assert controller._failed is None
    monkeypatch.undo()

    response = await asyncio.wait_for(
        controller._evaluate(Request(mode="llm", text="Hello", response_timeout=3)), 5
    )
    result = json.loads(response.text)
    assert result["row"]["status"] == "ok", result
    assert controller._consecutive_failures == 0


async def test_consecutive_failures_eventually_stop_the_endpoint(running_eval, monkeypatch):
    """A streak means the agent itself is broken; then refusing work is right."""
    from aiohttp import web

    controller, _ = await running_eval()

    async def blocked(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(controller, "_run", blocked)
    for _ in range(controller._max_consecutive_failures):
        response = await controller._evaluate(Request(mode="llm", text="Hello", response_timeout=1))
        assert json.loads(response.text)["row"]["status"] == "timeout"
    assert controller._failed is not None
    with pytest.raises(web.HTTPConflict):
        await controller._evaluate(Request(mode="llm", text="Next"))


async def test_tts_failure_is_recorded_and_the_agent_keeps_running(running_eval):
    """A dropped TTS websocket mid-reply ends the turn, not the agent.

    The failure is recorded with the close code that explains it, and the
    endpoint keeps serving — the session outlives the turn.
    """
    controller, _ = await running_eval(tts_provider=FailingTTS())

    async def one_trial(sample_id):
        request = Request(
            audio_b64=base64.b64encode(wav_bytes()).decode(),
            sample_id=sample_id,
            response_timeout=3,
            speech_start_s=0,
            speech_end_s=0.12,
            reference="hello",
        )
        response = await asyncio.wait_for(controller._evaluate(request), 5)
        return json.loads(response.text)

    result = await one_trial("1")
    assert result["row"]["status"] == "error"
    assert any("close_code=1006" in error["message"] for error in result["errors"])
    assert controller.active is None
    assert controller._failed is None

    # Still serving: the next trial is attempted, fails the same way, and is
    # recorded too — the agent did not shut down between turns.
    result = await one_trial("2")
    assert result["row"]["status"] == "error"
    assert controller._failed is None


async def test_http_runner_replays_sorted_folder_and_saves_reports(tmp_path, running_eval):
    controller, output = await running_eval()
    await controller.start(0)
    port = controller._runner.addresses[0][1]
    folder = tmp_path / "mono"
    folder.mkdir()
    for name in ("002.wav", "001.wav"):
        (folder / name).write_bytes(wav_bytes())
    args = SimpleNamespace(
        dataset=folder,
        limit=None,
        mode="agent",
        text=None,
        endpoint=f"http://127.0.0.1:{port}",
        output=tmp_path / "results",
        repeat=1,
        shuffle=False,
        seed=42,
        timeout=3,
        gap=0,
        label="test",
        keep_history=False,
        wait_ready=1,
        max_consecutive_failures=3,
    )
    assert await run_dataset(args) == 0
    assert output.completed == 2
    rows = [
        json.loads(line)["row"] for line in (args.output / "results.jsonl").read_text().splitlines()
    ]
    assert [row["sample_id"] for row in rows] == ["001.wav#r1", "002.wav#r1"]
    assert all(row["response_playout_ms"] >= 90 for row in rows)
    assert (args.output / "events.jsonl").is_file()
    assert (args.output / "report.html").is_file()
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["successful"] == 2
    assert summary["latencies"]["input_end_to_first_output_ms"]["n"] == 2


async def test_http_runner_records_failed_trials_and_finishes_the_run(tmp_path, running_eval):
    """Failed trials are data, not a reason to abandon the dataset.

    Both trials fail against the same agent and both are written out with
    their errors; the run reports failure through its exit code instead of
    stopping at the first bad turn.
    """
    controller, output = await running_eval(tts_provider=FailingTTS())
    await controller.start(0)
    port = controller._runner.addresses[0][1]
    folder = tmp_path / "mono"
    folder.mkdir()
    for name in ("001.wav", "002.wav"):
        (folder / name).write_bytes(wav_bytes())
    args = SimpleNamespace(
        dataset=folder,
        limit=None,
        mode="agent",
        text=None,
        endpoint=f"http://127.0.0.1:{port}",
        output=tmp_path / "results",
        repeat=1,
        shuffle=False,
        seed=42,
        timeout=3,
        gap=0,
        label="failing",
        keep_history=False,
        wait_ready=1,
        max_consecutive_failures=3,
    )
    assert await run_dataset(args) == 1
    records = [
        json.loads(line) for line in (args.output / "results.jsonl").read_text().splitlines()
    ]
    assert [record["row"]["status"] for record in records] == ["error", "error"]
    assert all(record["errors"] for record in records)
    # The close code survives into the artifacts, which is what makes the
    # failure diagnosable after the run.
    assert any("close_code=1006" in e["message"] for record in records for e in record["errors"])
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["samples"] == 2
    assert summary["failed"] == 2
