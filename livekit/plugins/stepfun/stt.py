from __future__ import annotations

import asyncio
import json
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import is_given

from . import _utils
from .log import logger
from .models import (
    DEFAULT_STT_SAMPLE_RATE,
    DEFAULT_VAD_SILENCE_DURATION_MS,
    DEFAULT_VAD_THRESHOLD,
    INCREMENTAL_DELTA_STT_MODELS,
    NUM_CHANNELS,
    PCM_BITS,
    PCM_CODEC,
    STT_FRAME_MS,
    STT_MAX_SESSION_DURATION,
    STT_RECOGNIZE_PATH,
    STT_STREAM_PATH,
    TIMESTAMP_ALIGN_STT_MODELS,
    Cluster,
    Language,
    STTBatchModels,
    STTModels,
)

# Event names on the bidirectional WebSocket. Note these are *not* the names the
# HTTP+SSE batch route uses (that one emits transcript.text.delta / .done); the
# two surfaces have separate vocabularies and mixing them up means listening for
# events that never arrive.
_EV_TRANSCRIPT_DELTA = "conversation.item.input_audio_transcription.delta"
_EV_TRANSCRIPT_COMPLETED = "conversation.item.input_audio_transcription.completed"
_EV_SPEECH_STARTED = "input_audio_buffer.speech_started"
_EV_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
_EV_SESSION_CREATED = "session.created"
_EV_SESSION_UPDATED = "session.updated"
_EV_COMMITTED = "input_audio_buffer.committed"
_EV_ITEM_CREATED = "conversation.item.created"
_EV_ERROR = "error"

#: How long to wait for the server to acknowledge our session.update before
#: giving up on the connection.
_SESSION_ACK_TIMEOUT = 10.0

#: How long to keep reading after the input ends, waiting for the transcript of
#: the audio already sent.
#:
#: With server VAD the server finalizes an utterance after it hears
#: `silence_duration_ms` of silence — but once input ends it hears nothing at
#: all rather than silence, so the final may never arrive. The stream cannot
#: wait indefinitely for it, and cutting it off immediately would drop the last
#: utterance of every session.
_FINALIZE_TIMEOUT = 3.0

if TYPE_CHECKING:
    # aiohttp 3.14 made ClientWebSocketResponse generic over whether receive()
    # decodes, and ws_connect() returns the [bool] flavour while the bare name
    # defaults to [Literal[True]]. Aliased so the subscript stays out of
    # runtime expressions, which older aiohttp cannot evaluate.
    _WebSocket = aiohttp.ClientWebSocketResponse[bool]
else:
    _WebSocket = aiohttp.ClientWebSocketResponse


@dataclass
class _STTOptions:
    model: STTModels | str
    batch_model: STTBatchModels | str
    language: NotGivenOr[Language | str]
    sample_rate: int
    #: Context or terminology hints for the streaming endpoint's `prompt` field.
    prompt: NotGivenOr[str]
    #: Hotword list for the batch endpoint. Documented on the domestic cluster
    #: only, so it is omitted internationally rather than risking a 400.
    hotwords: list[str]
    enable_itn: bool
    enable_timestamp_align: bool
    server_vad: bool
    vad_silence_duration_ms: int
    vad_threshold: float
    cluster: Cluster
    ws_base_url: str
    http_base_url: str

    @property
    def deltas_are_cumulative(self) -> bool:
        """Whether a transcription delta replaces the previous text or extends it.

        `stepaudio-2.5-asr-stream` and later send the whole transcript so far on
        every delta, corrections included, and the provider's docs say to
        replace the displayed text wholesale. `step-asr-1.1-stream` is the
        legacy exception that sends incremental fragments.

        Cumulative is the default for anything not on that denylist, including
        model strings this plugin has never heard of. Appending cumulative text
        compounds every partial into garbage, so an unknown model must not fall
        into that branch by accident.
        """
        return self.model not in INCREMENTAL_DELTA_STT_MODELS

    @property
    def timestamps_available(self) -> bool:
        return self.enable_timestamp_align and self.model in TIMESTAMP_ALIGN_STT_MODELS


class STT(stt.STT):
    """StepFun streaming and batch speech recognition.

    Usage::

        from livekit.plugins import stepfun

        session = AgentSession(
            stt=stepfun.STT(language="zh"),
            ...
        )
    """

    def __init__(
        self,
        *,
        model: STTModels | str = "stepaudio-2.5-asr-stream",
        batch_model: STTBatchModels | str = "stepaudio-3-asr-max",
        language: NotGivenOr[Language | str] = NOT_GIVEN,
        api_key: str | None = None,
        cluster: Cluster | None = None,
        base_url: str | None = None,
        http_base_url: str | None = None,
        sample_rate: int = DEFAULT_STT_SAMPLE_RATE,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        hotwords: list[str] | None = None,
        enable_itn: bool = True,
        enable_timestamp_align: bool = False,
        server_vad: bool = True,
        vad_silence_duration_ms: int = DEFAULT_VAD_SILENCE_DURATION_MS,
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
        use_streaming: bool = True,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            model: Model for the bidirectional streaming endpoint. Only used
                when `use_streaming` is True.
            batch_model: Model for the HTTP + SSE endpoint, used by `recognize()`
                and whenever `use_streaming` is False. Note the two endpoints
                accept disjoint model sets: `stepaudio-3-asr-max` is documented
                for the batch route only, and the streaming route accepts only
                `stepaudio-2.5-asr-stream`.
            language: Recognition language. Leaving it unset selects **English**
                server-side — it is not inferred from the cluster, so Chinese
                callers must pass language="zh" explicitly.
            prompt: Context or terminology hints for streaming recognition.
            hotwords: Hotword list for batch recognition. Documented on the
                domestic cluster only and omitted internationally.
            enable_timestamp_align: Request per-word timestamps. Enabling this
                also flips `aligned_transcript` to "word", which is what turns
                on LiveKit's adaptive interruption. Streaming only.
            server_vad: Let the server detect utterance boundaries and emit
                speech_started / speech_stopped. When disabled, nothing is
                segmented automatically and this plugin commits each utterance
                itself on flush. Streaming only.
            use_streaming: Whether to advertise native streaming recognition.

                Set False to run the HTTP + SSE route instead, which is the only
                way to use `batch_model` in a live conversation. The framework
                then wraps this STT in a `stt.StreamAdapter`, which buffers each
                utterance until the local VAD reports end of speech and sends it
                as a single request — so an `AgentSession` must be given a `vad`
                or it raises.

                The cost is latency and interim transcripts: nothing is
                recognized until the user stops talking, and the adapter
                declares `interim_results=False`, so there is no partial
                transcript to drive preemptive generation. Prefer the default
                unless the batch model's accuracy is worth that.
        """
        resolved_cluster = _utils.resolve_cluster(cluster)
        aligned: Literal["word", False] = (
            "word"
            if use_streaming and enable_timestamp_align and model in TIMESTAMP_ALIGN_STT_MODELS
            else False
        )

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_streaming,
                # The batch route returns incremental SSE deltas, but
                # StreamAdapter discards them and surfaces only the final
                # transcript, so claiming interim results here would be a lie.
                interim_results=use_streaming,
                # The batch SSE route works either way, so callers can reuse one
                # instance for offline work and FallbackAdapter has somewhere to
                # go when streaming fails.
                offline_recognize=True,
                # Not supported on the streaming endpoint.
                diarization=False,
                # Tri-state: Literal["word", "chunk", False].
                #
                # Known framework bug worth noting for callers: stt.FallbackAdapter
                # omits aligned_transcript when building its own capabilities, so
                # wrapping this STT in one silently drops word alignment
                # (livekit/agents issue #5436). stt.StreamAdapter omits it too.
                aligned_transcript=aligned,
                # The streaming endpoint takes free-text `prompt` hints rather
                # than a keyterm list, so the framework's keyterm hook is not
                # wired up. Pass prompt=... instead.
                keyterms=False,
                # Streaming ASR does not consume conversation context.
                chat_context=False,
            ),
        )

        if enable_timestamp_align and not aligned:
            if not use_streaming:
                logger.warning(
                    "enable_timestamp_align applies to the streaming endpoint "
                    "only; ignoring it because use_streaming=False"
                )
            else:
                logger.warning(
                    "enable_timestamp_align is only supported by %s; ignoring it",
                    ", ".join(sorted(TIMESTAMP_ALIGN_STT_MODELS)),
                    extra={"model": model},
                )

        self._use_streaming = use_streaming
        self._api_key = _utils.resolve_api_key(api_key)
        self._opts = _STTOptions(
            model=model,
            batch_model=batch_model,
            language=language,
            sample_rate=sample_rate,
            prompt=prompt,
            hotwords=hotwords or [],
            enable_itn=enable_itn,
            enable_timestamp_align=enable_timestamp_align,
            server_vad=server_vad,
            vad_silence_duration_ms=vad_silence_duration_ms,
            vad_threshold=vad_threshold,
            cluster=resolved_cluster,
            ws_base_url=_utils.resolve_ws_base_url(base_url, resolved_cluster),
            http_base_url=_utils.resolve_http_base_url(http_base_url, resolved_cluster),
        )

        self._session = http_session
        self._owns_session = http_session is None
        self._streams = weakref.WeakSet[RecognizeStream]()
        self._pool = utils.ConnectionPool[_WebSocket](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            # The endpoint documents no idle timeout value, only a
            # max_idle_timeout error code proving one exists, so prewarmed
            # connections are retired conservatively.
            max_session_duration=STT_MAX_SESSION_DURATION,
        )

    @property
    def model(self) -> str:
        # Report whichever endpoint is actually serving recognition. The two
        # routes take disjoint model sets, and stt.StreamAdapter delegates this
        # property to the wrapped STT, so returning the streaming model while
        # the batch route does the work would mislabel every metric.
        if not self._use_streaming:
            return str(self._opts.batch_model)
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "StepFun"

    # -- connections -------------------------------------------------------

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            # Reuse the framework's session: it pools connections and is scoped
            # to the job lifecycle.
            self._session = utils.http_context.http_session()
            self._owns_session = False
        return self._session

    async def _connect_ws(self, timeout: float) -> _WebSocket:
        url = f"{self._opts.ws_base_url}{STT_STREAM_PATH}"
        try:
            return await asyncio.wait_for(
                self._ensure_session().ws_connect(
                    url, headers=_utils.auth_headers(self._api_key)
                ),
                timeout,
            )
        except Exception as exc:
            raise _utils.translate_exception(exc, context="stepfun stt connect") from exc

    async def _close_ws(self, ws: _WebSocket) -> None:
        await ws.close()

    def prewarm(self) -> None:
        """Complete DNS, TLS and the WebSocket handshake ahead of time, so cold
        start cost does not land on the first turn.
        """
        self._ensure_session()
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        await self._pool.aclose()

        # Only close a session we created ourselves. Closing one that was passed
        # in is a common way for plugins to break their callers.
        if self._owns_session and self._session is not None:
            await self._session.close()
        self._session = None

    # -- configuration -----------------------------------------------------

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[Language | str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update options in place.

        This mutates the shared self._opts while one instance is reused across
        many conversation turns. RecognizeStream therefore snapshots _opts in its
        constructor rather than reading it lazily in _run(); reading it lazily
        races with the next update_options() call and shows up as intermittently
        using the previous turn's configuration, which is very hard to
        reproduce.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(prompt):
            self._opts.prompt = prompt

        # Pooled sockets were configured by a session.update built from the old
        # options, so stop handing them out.
        self._pool.invalidate()

    # -- streaming ---------------------------------------------------------

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        if not self._use_streaming:
            # Declaring streaming=False and then serving a stream anyway would
            # make the capability flag a lie, and would quietly send audio to
            # the streaming endpoint with a model it does not accept.
            raise RuntimeError(
                "this STT was constructed with use_streaming=False, so it has no "
                "streaming transport. Either construct it with use_streaming=True, "
                "or pass it to an AgentSession with a vad= so the framework wraps "
                "it in a stt.StreamAdapter and uses the batch endpoint."
            )

        opts = replace(self._opts)
        if is_given(language):
            opts.language = language

        stream = RecognizeStream(stt=self, opts=opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    # -- batch -------------------------------------------------------------

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """One-shot recognition over HTTP + SSE.

        Serves batch workloads, StreamAdapter setups and FallbackAdapter's
        offline escape hatch. The audio goes up in a single request and
        transcripts come back incrementally; only the final one matters here.
        """
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language

        frame = rtc.combine_audio_frames(buffer)

        transcription: dict[str, Any] = {
            "model": opts.batch_model,
            "enable_itn": opts.enable_itn,
        }
        if is_given(opts.language):
            transcription["language"] = opts.language
        if opts.hotwords and opts.cluster == "domestic":
            # Documented for the domestic cluster only; sending it
            # internationally risks a 400 on an unknown parameter.
            transcription["hotwords"] = opts.hotwords

        payload = {
            "audio": {
                "data": _utils.pcm_to_b64(frame.data.tobytes()),
                "input": {
                    "transcription": transcription,
                    # rate/bits/channel are mandatory for raw PCM. Taking them
                    # from the frame rather than from opts.sample_rate matters:
                    # the batch path receives whatever the caller buffered, with
                    # no resampling in between.
                    "format": {
                        "type": "pcm",
                        "codec": PCM_CODEC,
                        "rate": frame.sample_rate,
                        "bits": PCM_BITS,
                        "channel": frame.num_channels,
                    },
                },
            }
        }

        url = f"{opts.http_base_url}{STT_RECOGNIZE_PATH}"
        headers = {
            **_utils.auth_headers(self._api_key),
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        text = ""
        usage: dict[str, Any] = {}
        session_id = ""

        try:
            async with self._ensure_session().post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=None, sock_connect=conn_options.timeout
                ),
            ) as resp:
                await _utils.raise_for_status(resp)

                async for event in _utils.iter_sse_events(resp):
                    etype = event.get("type")
                    if etype == "transcript.text.delta":
                        # Incremental on this route, unlike the WebSocket's
                        # cumulative deltas.
                        text += event.get("delta") or ""
                    elif etype == "transcript.text.done":
                        text = event.get("text") or text
                        usage = event.get("usage") or {}
                        session_id = (event.get("meta") or {}).get("session_id", "")
                    elif etype == "error":
                        raise _utils.status_error(
                            status_code=-1,
                            message=f"stepfun stt recognize: {event.get('message')}",
                            request_id=(event.get("meta") or {}).get("session_id"),
                            body=event,
                        )
        except Exception as exc:
            raise _utils.translate_exception(exc, context="stepfun stt recognize") from exc

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=session_id or utils.shortuuid(),
            alternatives=[
                stt.SpeechData(
                    # This route echoes no detected language, so report what was
                    # asked for; English is the server-side default.
                    language=LanguageCode(
                        opts.language if is_given(opts.language) else "en"
                    ),
                    text=text,
                )
            ],
            recognition_usage=stt.RecognitionUsage(
                audio_duration=utils.audio.calculate_audio_duration(buffer),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
        )


def _close_detail(ws: _WebSocket, msg: Any) -> str:
    """Describe why a socket closed, as far as the transport can tell.

    A bare "closed unexpectedly" sends people to check their network, which is
    the wrong place when the real cause is a configuration the server refused —
    an unsupported model, for instance, which this endpoint signals by dropping
    the connection rather than returning an error frame.
    """
    code = ws.close_code
    if code is None and msg.type is aiohttp.WSMsgType.CLOSE:
        code = msg.data
    parts = []
    if code is not None:
        parts.append(f"code={code}")
    reason = msg.extra or None
    if reason:
        parts.append(f"reason={reason!r}")
    return f" ({', '.join(parts)})" if parts else ""


class RecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        # Hand sample_rate to the base class: it wires up an rtc.AudioResampler
        # inside push_frame() to downsample the 48kHz WebRTC audio to whatever
        # rate we need. Do not build a second resampler here.
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._stt_impl = stt
        self._opts = opts  # already replace()d in STT.stream(); this is a snapshot
        self._request_id = ""
        self._interim_text = ""
        self._input_done = False

        # RECOGNITION_USAGE is the *only* trigger for STTMetrics in streaming
        # mode — the base class's _metrics_monitor_task listens for that event
        # alone to build STTMetrics and fire metrics_collected. Not emitting it
        # means no STT metrics for the entire session (the Azure plugin hit
        # exactly this, issue #6151).
        #
        # Accumulated by hand rather than with a helper: utils.PeriodicCollector
        # existed in 1.x and was removed in 2.x. Reporting on each final (and
        # once more at close) also lines usage up with conversation turns, which
        # a wall-clock timer would not.
        self._audio_duration = 0.0
        self._pending_tokens = (0, 0)

    # -- main loop ---------------------------------------------------------

    async def _run(self) -> None:
        self._request_id = utils.shortuuid()
        self._interim_text = ""
        self._input_done = False

        async with self._stt_impl._pool.connection(
            timeout=self._conn_options.timeout
        ) as ws:
            self._report_connection_acquired(
                self._stt_impl._pool.last_acquire_time,
                self._stt_impl._pool.last_connection_reused,
            )

            await self._configure_session(ws)

            tasks = [
                asyncio.create_task(self._send_task(ws), name="stepfun_stt_send"),
                asyncio.create_task(self._recv_task(ws), name="stepfun_stt_recv"),
            ]
            # Side task, not part of the gather: it sleeps between pings, so
            # waiting on it would hold teardown open for a whole ping interval
            # after the stream is otherwise finished. A ping failure means a
            # broken socket, which _recv_task reports anyway.
            keepalive = asyncio.create_task(
                self._keepalive_task(ws), name="stepfun_stt_keepalive"
            )
            try:
                # Not gather(): _recv_task only returns when the socket closes,
                # and with server VAD the server keeps the session open for more
                # utterances rather than closing after a final. Waiting for both
                # therefore hangs forever once input ends. Wait for whichever
                # finishes first, then give the receive side a bounded grace
                # period to deliver the transcript for the audio already sent.
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                send_task, recv_task = tasks
                if send_task.done() and not recv_task.done():
                    await asyncio.wait([recv_task], timeout=_FINALIZE_TIMEOUT)
            finally:
                await utils.aio.cancel_and_wait(*tasks, keepalive)

                # The socket carries per-utterance state (buffered audio, item
                # ids) with no documented reset, so it is not safe to hand to
                # the next stream. Drop it and re-arm the pool so the next
                # stream still starts from a completed handshake.
                self._stt_impl._pool.remove(ws)
                self._stt_impl._pool.prewarm()

            # Surface a failure from either task; the framework then retries
            # according to conn_options.max_retry. This plugin performs no
            # retries of its own — see the note in _utils.
            for task in tasks:
                if not task.cancelled():
                    task.result()

    async def _configure_session(self, ws: _WebSocket) -> None:
        """Send session.update and wait for the server to acknowledge it.

        This endpoint accepts no query parameters: the model, audio format and
        turn detection are configured entirely by this frame. Skipping it leaves
        the session unconfigured, which is why pumping audio straight after
        connecting cannot work.
        """
        transcription: dict[str, Any] = {
            "model": self._opts.model,
            "enable_itn": self._opts.enable_itn,
        }
        if is_given(self._opts.language):
            transcription["language"] = self._opts.language
        if is_given(self._opts.prompt):
            transcription["prompt"] = self._opts.prompt
        if self._opts.timestamps_available:
            transcription["enable_timestamp_align"] = True

        audio_input: dict[str, Any] = {
            "format": {
                "type": "pcm",
                "codec": PCM_CODEC,
                "rate": self._opts.sample_rate,
                "bits": PCM_BITS,
                "channel": NUM_CHANNELS,
            },
            "transcription": transcription,
        }
        if self._opts.server_vad:
            audio_input["turn_detection"] = {
                "type": "server_vad",
                "silence_duration_ms": self._opts.vad_silence_duration_ms,
                "threshold": self._opts.vad_threshold,
            }

        await ws.send_str(
            json.dumps(
                {
                    "event_id": _utils.event_id(),
                    "type": "session.update",
                    "session": {"audio": {"input": audio_input}},
                }
            )
        )

        # Wait for the ack before sending audio, so a rejected configuration
        # surfaces as a configuration error rather than as silence.
        try:
            await asyncio.wait_for(self._await_session_ack(ws), _SESSION_ACK_TIMEOUT)
        except TimeoutError as exc:
            raise APIConnectionError(
                "stepfun stt: timed out waiting for session.update acknowledgement"
            ) from exc

    async def _await_session_ack(self, ws: _WebSocket) -> None:
        while True:
            msg = await ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                raise APIConnectionError(
                    "stepfun stt: the server closed the connection while "
                    "configuring the session"
                    f"{_close_detail(ws, msg)}. This usually means the session "
                    "configuration was refused — most often an unsupported "
                    f"model (model={self._opts.model!r}) or audio format — "
                    "rather than a network problem."
                )
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            event = json.loads(msg.data)
            etype = event.get("type")
            if etype in (_EV_SESSION_CREATED, _EV_SESSION_UPDATED):
                logger.debug(
                    "stepfun stt: session ready",
                    extra={
                        "type": etype,
                        "session_id": (event.get("meta") or {}).get("session_id"),
                    },
                )
                return
            if etype == _EV_ERROR:
                raise _utils.translate_asr_ws_error(
                    event, context="stepfun stt session.update"
                )
            # session.created can arrive unprompted before our ack; anything
            # else at this stage is noise.
            logger.debug("stepfun stt: ignoring event during setup", extra={"type": etype})

    # -- upstream ----------------------------------------------------------

    async def _send_task(self, ws: _WebSocket) -> None:
        samples_per_frame = self._opts.sample_rate * STT_FRAME_MS // 1000
        bstream = utils.audio.AudioByteStream(
            self._opts.sample_rate,
            NUM_CHANNELS,
            samples_per_channel=samples_per_frame,
        )

        async for data in self._input_ch:
            frames: list[rtc.AudioFrame] = []
            flushing = False

            if isinstance(data, rtc.AudioFrame):
                frames.extend(bstream.write(data.data.tobytes()))
            elif isinstance(data, self._FlushSentinel):
                # Drain the sub-frame tail held by AudioByteStream, or that
                # audio is lost.
                frames.extend(bstream.flush())
                flushing = True

            for frame in frames:
                self._audio_duration += frame.samples_per_channel / frame.sample_rate
                await ws.send_str(
                    json.dumps(
                        {
                            "event_id": _utils.event_id(),
                            "type": "input_audio_buffer.append",
                            # Raw PCM16, matching the format declared in
                            # session.update. (The provider's field note says
                            # "WAV" here while its format block specifies PCM —
                            # the declared format is the consistent reading.)
                            "audio": _utils.pcm_to_b64(frame.data.tobytes()),
                        }
                    )
                )

            if flushing:
                await self._maybe_commit(ws)

        self._input_done = True
        await self._maybe_commit(ws)

    async def _maybe_commit(self, ws: _WebSocket) -> None:
        """Tell the server the current utterance is finished.

        Only sent when server VAD is off, per the documented guidance: with
        turn_detection=server_vad the server owns segmentation, and committing
        there would compete with it.

        Both modes finalize correctly, verified against the live service:

          * server_vad=True — the server finalizes once it *hears*
            `silence_duration_ms` of silence. A final arrived 1372ms after the
            last word (the server's own 800ms silence window plus processing).
            This is the `AgentSession` mode: the framework never calls flush()
            on a natively streaming STT, so nothing would ever commit with
            server VAD off, but a live microphone supplies the silence the
            server is waiting for.
          * server_vad=False — flush() commits and the final arrived 438ms
            later. This is the mode for driving `stream()` yourself.

        One consequence worth knowing: server VAD needs *audio* containing
        silence, not an absence of audio. A caller that stops sending the
        instant the speaker stops talking gives it nothing to detect and never
        sees a final. That is a property of silence detection, not a server
        defect — a real microphone keeps streaming.
        """
        if self._opts.server_vad:
            return

        await ws.send_str(
            json.dumps(
                {"event_id": _utils.event_id(), "type": "input_audio_buffer.commit"}
            )
        )

    # -- downstream --------------------------------------------------------

    async def _recv_task(self, ws: _WebSocket) -> None:
        while True:
            msg = await ws.receive()

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                if self._input_done:
                    # Final settle, so the last few seconds of usage are not
                    # dropped.
                    self._report_usage()
                    return
                raise APIConnectionError(
                    "stepfun stt: the server closed the connection mid-stream"
                    f"{_close_detail(ws, msg)}. If this happens immediately "
                    "after the first audio, the server is most likely "
                    f"rejecting the model (model={self._opts.model!r}) — the "
                    "bidirectional endpoint documents support for "
                    "stepaudio-2.5-asr-stream only."
                )

            if msg.type != aiohttp.WSMsgType.TEXT:
                logger.debug("stepfun stt: unexpected ws frame type %s", msg.type)
                continue

            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("stepfun stt: failed to decode ws frame")
                continue

            self._handle_event(event)

    def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")

        if etype == _EV_ERROR:
            raise _utils.translate_asr_ws_error(event, context="stepfun stt")

        if etype == _EV_TRANSCRIPT_DELTA:
            self._handle_delta(event)
            return

        if etype == _EV_TRANSCRIPT_COMPLETED:
            self._handle_completed(event)
            return

        # Genuine server-side VAD events, so translating them does not violate
        # the "never fabricate VAD" rule: nothing here is inferred from
        # client-side energy detection, which would fight silero.VAD and the
        # turn detector in a mixed setup. They only exist when server VAD is on.
        if etype == _EV_SPEECH_STARTED:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.START_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
            return

        if etype == _EV_SPEECH_STOPPED:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    request_id=self._request_id,
                )
            )
            return

        if etype in (_EV_SESSION_CREATED, _EV_SESSION_UPDATED, _EV_COMMITTED, _EV_ITEM_CREATED):
            logger.debug("stepfun stt: %s", etype)
            return

        logger.debug("stepfun stt: unhandled event type %s", etype)

    def _handle_delta(self, event: dict[str, Any]) -> None:
        text = event.get("text") or ""

        if self._opts.deltas_are_cumulative:
            # Replaces, never appends — the payload is the full transcript so
            # far, including revisions to text already emitted.
            self._interim_text = text
        else:
            self._interim_text += text

        if not self._interim_text:
            return

        # `stash` holds the trailing characters the server may still rewrite, so
        # everything before it is settled. That is exactly what
        # PREFLIGHT_TRANSCRIPT is for: stable enough for preemptive generation
        # while the rest of the utterance is still in flux.
        stash = event.get("stash") or ""
        if stash and self._interim_text.endswith(stash):
            stable = self._interim_text[: -len(stash)]
            if stable:
                self._emit(
                    stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, stable, event, words=None
                )

        self._emit(
            stt.SpeechEventType.INTERIM_TRANSCRIPT,
            self._interim_text,
            event,
            words=self._timed_words(event),
        )

    def _handle_completed(self, event: dict[str, Any]) -> None:
        # The completed frame carries the final text under `transcript`, not
        # `text`.
        text = event.get("transcript") or self._interim_text
        self._interim_text = ""

        usage = event.get("usage") or {}
        self._pending_tokens = (
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

        self._emit(
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            text,
            event,
            words=self._timed_words(event),
        )
        # Settle usage right after each final so metrics align with turns.
        self._report_usage()

    def _timed_words(self, event: dict[str, Any]) -> list[TimedString] | None:
        """Translate the per-word `words` array, when timestamp alignment is on.

        Entries look like {"word", "start", "end", "final"} with start/end as
        floats in seconds. A `final` of false means the word sits in the stash
        region and its timing is still an estimate, so those are skipped rather
        than published as if they were settled.

        start_time_offset anchors the words against the session timeline, which
        the base class maintains across reconnects.
        """
        raw = event.get("words")
        if not raw or not self._opts.timestamps_available:
            return None

        words = [
            TimedString(
                entry.get("word", ""),
                start_time=entry.get("start", NOT_GIVEN),
                end_time=entry.get("end", NOT_GIVEN),
                start_time_offset=self.start_time_offset,
            )
            for entry in raw
            if entry.get("final", True)
        ]
        return words or None

    def _emit(
        self,
        event_type: stt.SpeechEventType,
        text: str,
        raw: dict[str, Any],
        *,
        words: list[TimedString] | None,
    ) -> None:
        if not text:
            return

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=event_type,
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(
                        # The endpoint echoes no detected language, so report
                        # the configured one; English is the server default.
                        language=LanguageCode(
                            self._opts.language
                            if is_given(self._opts.language)
                            else "en"
                        ),
                        text=text,
                        words=words,
                    )
                ],
            )
        )

    def _report_usage(self) -> None:
        """Emit accumulated usage, then reset.

        Audio duration is accumulated client-side; the token counts come from
        the server's own `usage` block on the completed event, so they match
        what is billed. Emitting nothing when there is nothing to report keeps
        the close-time settle from producing an empty metrics record.
        """
        input_tokens, output_tokens = self._pending_tokens
        if not self._audio_duration and not input_tokens and not output_tokens:
            return

        audio_duration = self._audio_duration
        self._audio_duration = 0.0
        self._pending_tokens = (0, 0)

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=audio_duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )
        )

    # -- keepalive ---------------------------------------------------------

    async def _keepalive_task(self, ws: _WebSocket) -> None:
        """Keep the socket alive through long silences.

        The endpoint exposes `max_idle_timeout` and `pong_timeout` error codes,
        which prove both an idle timeout and a heartbeat exist, but documents
        neither the interval nor a keepalive frame format. WebSocket-level pings
        are the portable choice; the interval is deliberately short.

        Runs until cancelled by _run().
        """
        while True:
            await asyncio.sleep(10.0)
            await ws.ping()


# Deprecated alias. The base class renamed SpeechStream to RecognizeStream and
# kept its own alias; this mirrors that so existing imports keep working.
SpeechStream = RecognizeStream
