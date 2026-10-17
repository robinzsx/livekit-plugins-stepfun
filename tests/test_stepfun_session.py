"""Exercise actual AgentSession audio wiring with scripted provider responses.

These verify endpointing and SDK plumbing, not recognition accuracy or the
quality of the local turn model. No service credentials or model downloads.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents import Agent, AgentSession, inference, llm, utils, vad
from livekit.agents.voice import io

from livekit import rtc
from livekit.plugins import stepfun

from .fake_ws import FakePool, FakeWebSocket
from .test_stepfun_stt import _FakeSSESession, _frame


class AudioFeed(io.AudioInput):
    def __init__(self):
        super().__init__(label="test audio")
        self.frames = utils.aio.Chan[rtc.AudioFrame]()

    async def __anext__(self):
        return await self.frames.__anext__()


class TurnProbe(Agent):
    def __init__(self):
        super().__init__(instructions="Test turn boundaries", llm=None, tts=None)
        self.turns = asyncio.Queue()

    async def on_user_turn_completed(self, turn_ctx, new_message):
        self.turns.put_nowait(new_message.text_content)
        raise llm.StopResponse()


class ScriptedVAD(vad.VAD):
    """Treat nonzero fixture PCM as speech and zero PCM as silence."""

    def __init__(self):
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.1))

    def stream(self):
        return ScriptedVADStream(self)


class ScriptedVADStream(vad.VADStream):
    async def _main_task(self):
        frames = []
        try:
            async for frame in self._input_ch:
                if isinstance(frame, self._FlushSentinel):
                    frames = []
                    continue
                speaking = any(frame.data)
                if speaking:
                    if not frames:
                        self._event_ch.send_nowait(self._event(vad.VADEventType.START_OF_SPEECH))
                    frames.append(frame)
                elif frames:
                    inference_event = self._event(vad.VADEventType.INFERENCE_DONE)
                    inference_event.raw_accumulated_silence = 0.55
                    self._event_ch.send_nowait(inference_event)
                    self._event_ch.send_nowait(self._event(vad.VADEventType.END_OF_SPEECH, frames))
                    frames = []
        finally:
            self._event_ch.close()

    def _event(self, kind, frames=None):
        return vad.VADEvent(
            type=kind,
            samples_index=0,
            timestamp=0,
            speech_duration=0.1,
            silence_duration=0,
            frames=frames or [],
        )


class ScriptedTurnDetector:
    model = "scripted"
    provider = "test"

    def __init__(self):
        self.transcripts = []
        self.called = asyncio.Event()
        self.allow_completion = asyncio.Event()

    async def supports_language(self, language):
        return True

    async def unlikely_threshold(self, language):
        return 0.5

    async def predict_end_of_turn(self, chat_ctx, *, timeout=None):
        self.transcripts.append(chat_ctx.items[-1].text_content)
        self.called.set()
        await self.allow_completion.wait()
        return 1.0


@pytest.mark.parametrize("local_vad", [False, True])
async def test_streaming_server_endpointing_commits_two_turns(local_vad):
    ws = FakeWebSocket()
    ws.queue_server_event({"type": "session.updated"})
    speech = stepfun.STT(api_key="test-key", http_session=_FakeSSESession([]))
    speech._pool = FakePool(ws)
    feed, probe = AudioFeed(), TurnProbe()
    session = AgentSession(
        stt=speech,
        vad=ScriptedVAD() if local_vad else None,
        turn_handling={
            "turn_detection": "stt",
            "interruption": {"mode": "vad"},
            "endpointing": {"min_delay": 0.0},
        },
    )
    session.input.audio = feed
    try:
        await session.start(agent=probe)
        feed.frames.send_nowait(_frame())
        for text in ("hello", "another question"):
            ws.queue_server_event({"type": "input_audio_buffer.speech_started"})
            ws.queue_server_event(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "text": text,
                    "language": "en",
                }
            )
            ws.queue_server_event({"type": "input_audio_buffer.speech_stopped"})
            ws.queue_server_event(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": text,
                    "language": "en",
                }
            )
            assert await asyncio.wait_for(probe.turns.get(), 3) == text
        assert not ws.frames_of_type("input_audio_buffer.commit")
    finally:
        await session.aclose()
        await speech.aclose()
        feed.frames.close()


async def test_batch_waits_for_vad_then_runs_turn_detector():
    transport = _FakeSSESession(['data: {"type":"transcript.text.done","text":"hello"}'])
    speech = stepfun.STT(api_key="test-key", use_streaming=False, http_session=transport)
    detector = ScriptedTurnDetector()
    feed, probe = AudioFeed(), TurnProbe()
    session = AgentSession(
        stt=speech,
        vad=ScriptedVAD(),
        turn_handling={
            "turn_detection": detector,
            "interruption": {"mode": "vad"},
            "endpointing": {"min_delay": 0.0},
        },
    )
    session.input.audio = feed
    try:
        await session.start(agent=probe)
        frame = rtc.AudioFrame(
            data=b"\xf4\x01" * 1600, sample_rate=16000, num_channels=1, samples_per_channel=1600
        )
        feed.frames.send_nowait(frame)
        # A speech start alone must not trigger a batch upload.
        await asyncio.sleep(0.05)
        assert transport.url == ""
        assert probe.turns.empty()
        feed.frames.send_nowait(_frame())
        await asyncio.wait_for(detector.called.wait(), 3)
        assert probe.turns.empty()
        assert detector.transcripts[-1] == "hello"
        detector.allow_completion.set()
        assert await asyncio.wait_for(probe.turns.get(), 3) == "hello"
        assert transport.json["audio"]["input"]["transcription"]["model"] == "stepaudio-3-asr-max"
    finally:
        await session.aclose()
        await speech.aclose()
        feed.frames.close()


class PlaybackSink(io.AudioOutput):
    def __init__(self):
        super().__init__(label="test speaker", capabilities=io.AudioOutputCapabilities(pause=False))
        self.started = asyncio.Event()
        self.cleared = asyncio.Event()

    async def capture_frame(self, frame):
        await super().capture_frame(frame)
        self.started.set()
        await asyncio.sleep(0.02)

    def flush(self):
        super().flush()
        self.on_playback_finished(playback_position=0.1, interrupted=False)

    def clear_buffer(self):
        super().flush()
        self.on_playback_finished(playback_position=0.1, interrupted=True)
        self.cleared.set()


async def test_streaming_without_local_vad_can_interrupt_on_transcript():
    ws = FakeWebSocket()
    ws.queue_server_event({"type": "session.updated"})
    speech = stepfun.STT(api_key="test-key", http_session=_FakeSSESession([]))
    speech._pool = FakePool(ws)
    feed, probe, output = AudioFeed(), TurnProbe(), PlaybackSink()
    session = AgentSession(
        stt=speech,
        vad=None,
        aec_warmup_duration=None,
        turn_handling={"turn_detection": "stt", "interruption": {"mode": "vad"}},
    )
    session.input.audio = feed
    session.output.audio = output

    async def reply_audio():
        for _ in range(500):
            yield _frame()

    try:
        await session.start(agent=probe)
        feed.frames.send_nowait(_frame())
        reply = session.say("A long reply", audio=reply_audio())
        await asyncio.wait_for(output.started.wait(), 3)
        ws.queue_server_event({"type": "input_audio_buffer.speech_started"})
        ws.queue_server_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "text": "please stop",
                "language": "en",
            }
        )
        await asyncio.wait_for(output.cleared.wait(), 3)
        assert reply.interrupted
    finally:
        await session.aclose()
        await speech.aclose()
        feed.frames.close()


async def test_batch_with_builtin_local_audio_turn_detector(monkeypatch):
    from livekit.agents.inference.eot import transports

    predictions = []

    class LocalModel:
        def predict(self, audio):
            predictions.append(len(audio))
            return 1.0

    # Exercise the real built-in detector/transport; replace only native model
    # inference so this deterministic test does not need weights or a download.
    monkeypatch.setattr(transports, "_EOT", LocalModel)
    transport = _FakeSSESession(['data: {"type":"transcript.text.done","text":"hello"}'])
    speech = stepfun.STT(api_key="test-key", use_streaming=False, http_session=transport)
    feed, probe = AudioFeed(), TurnProbe()
    session = AgentSession(
        stt=speech,
        vad=ScriptedVAD(),
        turn_handling={
            "turn_detection": inference.TurnDetector(version="v1-mini"),
            "interruption": {"mode": "vad"},
            "endpointing": {"min_delay": 0.0},
        },
    )
    session.input.audio = feed
    try:
        await session.start(agent=probe)
        feed.frames.send_nowait(
            rtc.AudioFrame(
                data=b"\xf4\x01" * 1600, sample_rate=16000, num_channels=1, samples_per_channel=1600
            )
        )
        await asyncio.sleep(0.05)
        assert transport.url == ""
        feed.frames.send_nowait(_frame())
        assert await asyncio.wait_for(probe.turns.get(), 3) == "hello"
        assert predictions and predictions[0] > 0
        assert transport.json["audio"]["input"]["transcription"]["model"] == "stepaudio-3-asr-max"
    finally:
        await session.aclose()
        await speech.aclose()
        feed.frames.close()


async def test_local_vad_clears_console_audio_before_batch_transcription():
    from livekit.agents.cli.tcp_console import TcpAudioOutput

    class SpeakingVADStream(vad.VADStream):
        async def _main_task(self):
            duration = 0.0
            try:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        continue
                    kind = vad.VADEventType.INFERENCE_DONE
                    if duration == 0:
                        kind = vad.VADEventType.START_OF_SPEECH
                    duration += data.duration
                    self._event_ch.send_nowait(
                        vad.VADEvent(
                            type=kind,
                            samples_index=0,
                            timestamp=0,
                            speech_duration=duration,
                            silence_duration=0,
                            raw_accumulated_speech=duration,
                            speaking=True,
                            probability=1.0,
                            frames=[data],
                        )
                    )
            finally:
                self._event_ch.close()

    class SpeakingVAD(ScriptedVAD):
        def stream(self):
            return SpeakingVADStream(self)

    class Transport:
        def __init__(self):
            self.started = asyncio.Event()
            self.cleared = asyncio.Event()

        def send_message_threadsafe(self, message):
            if message.WhichOneof("message") == "audio_output":
                self.started.set()
            elif message.WhichOneof("message") == "audio_playback_clear":
                self.cleared.set()

    transport = Transport()
    output = TcpAudioOutput(transport)
    http = _FakeSSESession(['data: {"type":"transcript.text.done","text":"stop"}'])
    speech = stepfun.STT(api_key="test-key", use_streaming=False, http_session=http)
    feed = AudioFeed()
    session = AgentSession(
        stt=speech,
        vad=SpeakingVAD(),
        aec_warmup_duration=None,
        turn_handling={
            "turn_detection": "vad",
            "interruption": {
                "mode": "vad",
                "min_duration": 0.2,
                "min_words": 0,
                "resume_false_interruption": False,
            },
        },
    )
    session.input.audio = feed
    session.output.audio = output

    async def reply_audio():
        for _ in range(50):
            yield _frame()
            await asyncio.sleep(0)

    try:
        await session.start(agent=TurnProbe())
        reply = session.say("A long reply", audio=reply_audio())
        await asyncio.wait_for(transport.started.wait(), 3)
        # Speech stays ongoing: no VAD END_OF_SPEECH and no batch request.
        # The real console must receive clear while the user is still talking.
        for _ in range(3):
            feed.frames.send_nowait(_frame())
        await asyncio.wait_for(transport.cleared.wait(), 3)
        assert reply.interrupted
        assert http.url == ""
    finally:
        await session.aclose()
        await speech.aclose()
        feed.frames.close()
