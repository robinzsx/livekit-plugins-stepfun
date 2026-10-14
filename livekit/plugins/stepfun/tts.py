from __future__ import annotations

import asyncio
import json
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import is_given

from . import _utils
from .log import logger
from .models import (
    DEFAULT_TTS_SAMPLE_RATE,
    DEFAULT_TTS_VOICE,
    INSTRUCTION_MAX_CHARS,
    INSTRUCTION_TTS_MODELS,
    NUM_CHANNELS,
    SYSTEM_VOICES_MODEL,
    SYSTEM_VOICES_PATH,
    TTS_MAX_INPUT_CHARS,
    TTS_MAX_SESSION_DURATION,
    TTS_SAMPLE_RATES,
    TTS_STREAM_PATH,
    TTS_SYNTHESIZE_PATH,
    Cluster,
    Language,
    TTSModels,
    TTSVoices,
)

# Client -> server
_EV_CREATE = "tts.create"
_EV_TEXT_DELTA = "tts.text.delta"
_EV_TEXT_FLUSH = "tts.text.flush"
_EV_TEXT_DONE = "tts.text.done"

# Server -> client
_EV_CONNECTION_DONE = "tts.connection.done"
_EV_RESPONSE_CREATED = "tts.response.created"
_EV_SENTENCE_START = "tts.response.sentence.start"
_EV_SENTENCE_END = "tts.response.sentence.end"
_EV_AUDIO_DELTA = "tts.response.audio.delta"
_EV_AUDIO_DONE = "tts.response.audio.done"
_EV_SUBTITLE = "tts.response.subtitle"
_EV_TEXT_FLUSHED = "tts.text.flushed"
_EV_ERROR = "tts.response.error"

#: Raw PCM16 on the streaming path. AudioEmitter passes it straight through as
#: frames, which is the lowest-latency option; it also means the declared
#: sample_rate is load-bearing, since raw PCM carries no header to correct it.
_STREAM_MIME_TYPE = "audio/pcm"

#: WAV on the one-shot path. This is deliberate: AudioEmitter reads the real
#: sample rate and channel count out of the WAV header, so the one-shot path
#: cannot be mis-tuned into playing at the wrong pitch even if the declared
#: rate were wrong.
_CHUNKED_RESPONSE_FORMAT = "wav"
_CHUNKED_MIME_TYPE = "audio/wav"

_HANDSHAKE_TIMEOUT = 10.0

if TYPE_CHECKING:
    # aiohttp 3.14 made ClientWebSocketResponse generic over whether receive()
    # decodes, and ws_connect() returns the [bool] flavour while the bare name
    # defaults to [Literal[True]]. Aliased so the subscript stays out of
    # runtime expressions, which older aiohttp cannot evaluate.
    _WebSocket = aiohttp.ClientWebSocketResponse[bool]
else:
    _WebSocket = aiohttp.ClientWebSocketResponse

#: Characters that open or close an inline directive on the models that parse
#: them. Both the ASCII and the full-width forms appear in provider examples.
_INLINE_DIRECTIVE_CHARS = str.maketrans({"(": None, ")": None, "（": None, "）": None})


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str
    instruction: NotGivenOr[str]
    language: NotGivenOr[Language | str]
    speed: float
    volume: float
    sample_rate: int
    mode: Literal["default", "sentence"]
    text_normalization: Literal["standard", "enhanced"]
    markdown_filter: bool
    timestamp: bool
    cluster: Cluster
    ws_base_url: str
    http_base_url: str
    # Whether inline directives embedded in downstream text are allowed to
    # reach the server. Defaults to False — see TTS.__init__ for why.
    enable_inline_instruction: bool

    @property
    def supports_instruction(self) -> bool:
        """Whether this model accepts `instruction` and parses inline directives.

        The other models reject `instruction` outright, so it must not be sent
        to them, and they do not interpret parentheses — which also means no
        escaping is needed there.
        """
        return self.model in INSTRUCTION_TTS_MODELS


# eq=False keeps object identity semantics, which this type needs for two
# reasons. utils.ConnectionPool stores connections as dict keys and set members,
# and a dataclass with the default eq=True sets __hash__ to None — making this
# unhashable and the pool unusable. Identity is also the correct comparison
# here: two connections are the same connection only if they are the same
# object, never because they happen to share a session id.
@dataclass(eq=False)
class _Connection:
    """A streaming TTS socket together with the session the server issued for it.

    The pair travels together because the server, not the client, owns the
    session id: it arrives in `tts.connection.done` right after the handshake
    and every subsequent frame has to quote it. Capturing it at connect time is
    also what makes prewarming worthwhile — a pooled connection has already
    paid for DNS, TLS, the WebSocket upgrade *and* this round trip.
    """

    ws: _WebSocket
    session_id: str


class TTS(tts.TTS):
    """StepFun speech synthesis.

    Usage::

        from livekit.plugins import stepfun

        session = AgentSession(
            tts=stepfun.TTS(voice="vibrant-youth", instruction="warm, professional"),
            ...
        )
    """

    # ------------------------------------------------------------------
    # On the Markup mechanism
    # ------------------------------------------------------------------
    #
    # tts.TTS.Markup is the framework's formal channel for telling the LLM which
    # nonverbal markers it may emit (laughter, sighs, breaths), which is what
    # actually surfaces this provider's expressiveness instead of leaving each
    # caller to prompt-engineer it themselves.
    #
    # It cannot be enabled from here, though. Markup._provider_key() is only a
    # key into the framework's own tables in livekit/agents/tts/_provider_format
    # (supported_nonverbals / llm_instructions / convert_markup). Returning
    # "stepfun" with no upstream entry yields empty tables — an override that
    # declares nothing. Landing this therefore requires a change to
    # livekit-agents itself, as part of the upstream plugin PR, not a change
    # here.

    def __init__(
        self,
        *,
        model: TTSModels | str = "stepaudio-3-tts",
        voice: TTSVoices | str = DEFAULT_TTS_VOICE,
        instruction: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[Language | str] = NOT_GIVEN,
        speed: float = 1.0,
        volume: float = 1.0,
        sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
        mode: Literal["default", "sentence"] = "default",
        text_normalization: Literal["standard", "enhanced"] = "standard",
        markdown_filter: bool = False,
        timestamp: bool = False,
        api_key: str | None = None,
        cluster: Cluster | None = None,
        base_url: str | None = None,
        http_base_url: str | None = None,
        enable_inline_instruction: bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            voice: Voice ID. An ID the cluster does not know is rejected with
                `400 voice_id_invalid`, so use `list_voices()` to discover them
                rather than guessing. The international catalogue is a strict
                subset of the domestic one.
            instruction: Global natural-language direction for emotion and
                persona. Only honoured by stepaudio-2.5-tts (200 chars) and
                stepaudio-3-tts (500 chars); other models reject it. We control
                this value, so it is safe to pass through verbatim.
            language: Leaving it unset selects **English** server-side.
            mode: "default" buffers and auto-segments the incoming text, which
                is what you want when feeding it an LLM's token stream;
                "sentence" splits on terminal punctuation instead. Either way
                the server does its own sentence buffering, so this plugin does
                not tokenize ahead of it.
            timestamp: Request word-level subtitles. Enabling this also flips
                `aligned_transcript` on.
            enable_inline_instruction: Whether parenthesised inline directives
                found in *downstream text* are allowed to take effect.

                Defaults to False, deliberately. In a LiveKit pipeline the text
                pushed to TTS is generated freely by the LLM and is not under
                our control. On the models that parse them, text inside
                parentheses is read as direction and is not spoken — so an
                unlucky parenthetical silently drops words, and a user who can
                influence the LLM's wording can reach the voice settings
                indirectly. See `_escape_text` for what the safe default does.
        """
        if sample_rate not in TTS_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {sorted(TTS_SAMPLE_RATES)}, got {sample_rate}"
            )
        if not 0.5 <= speed <= 2.0:
            raise ValueError(f"speed must be between 0.5 and 2.0, got {speed}")
        if not 0.1 <= volume <= 2.0:
            raise ValueError(f"volume must be between 0.1 and 2.0, got {volume}")

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=timestamp,
            ),
            # Must match what the server actually produces. A wrong value
            # raises no error; it just plays back at the wrong pitch.
            sample_rate=sample_rate,
            # Not documented for the streaming endpoint, which exposes no
            # channel parameter. Mono is near-certain for a single TTS voice,
            # and the WAV header on the one-shot path confirms it at runtime.
            num_channels=NUM_CHANNELS,
        )

        resolved_cluster = _utils.resolve_cluster(cluster)
        _utils.warn_if_voice_unavailable(voice, resolved_cluster)

        if is_given(instruction) and model not in INSTRUCTION_TTS_MODELS:
            logger.warning(
                "instruction is only supported by %s and is rejected by other "
                "models; dropping it",
                ", ".join(sorted(INSTRUCTION_TTS_MODELS)),
                extra={"model": model},
            )
            instruction = NOT_GIVEN
        elif is_given(instruction):
            cap = INSTRUCTION_MAX_CHARS.get(str(model))
            if cap is not None and len(instruction) > cap:
                raise ValueError(
                    f"instruction for {model} is capped at {cap} characters, "
                    f"got {len(instruction)}"
                )

        self._api_key = _utils.resolve_api_key(api_key)
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            instruction=instruction,
            language=language,
            speed=speed,
            volume=volume,
            sample_rate=sample_rate,
            mode=mode,
            text_normalization=text_normalization,
            markdown_filter=markdown_filter,
            timestamp=timestamp,
            cluster=resolved_cluster,
            ws_base_url=_utils.resolve_ws_base_url(base_url, resolved_cluster),
            http_base_url=_utils.resolve_http_base_url(http_base_url, resolved_cluster),
            enable_inline_instruction=enable_inline_instruction,
        )

        self._session = http_session
        self._owns_session = http_session is None
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[_Connection](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            # The server drops an idle streaming connection after roughly 60s,
            # so pooled ones are retired well before that.
            max_session_duration=TTS_MAX_SESSION_DURATION,
        )

    @property
    def model(self) -> str:
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "StepFun"

    # -- connections -------------------------------------------------------

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
            self._owns_session = False
        return self._session

    async def _connect_ws(self, timeout: float) -> _Connection:
        """Connect and complete the server's half of the handshake.

        Unlike the ASR endpoint, this one takes the model as a query parameter.
        The server then sends `tts.connection.done` carrying the session id that
        every later frame must quote, so the handshake is finished here rather
        than at first use — that way a prewarmed connection is genuinely ready.
        """
        url = f"{self._opts.ws_base_url}{TTS_STREAM_PATH}?model={self._opts.model}"

        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(
                    url, headers=_utils.auth_headers(self._api_key)
                ),
                timeout,
            )
        except Exception as exc:
            raise _utils.translate_exception(exc, context="stepfun tts connect") from exc

        try:
            session_id = await asyncio.wait_for(
                _await_connection_done(ws), _HANDSHAKE_TIMEOUT
            )
        except BaseException:
            await ws.close()
            raise

        return _Connection(ws=ws, session_id=session_id)

    async def _close_ws(self, conn: _Connection) -> None:
        await conn.ws.close()

    def prewarm(self) -> None:
        """Open a connection and complete its handshake ahead of time.

        TTFB is the first number customers look at when evaluating a TTS, so
        moving the handshake out of the first utterance pays off visibly.
        """
        self._ensure_session()
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        await self._pool.aclose()

        # Only close a session we created ourselves.
        if self._owns_session and self._session is not None:
            await self._session.close()
        self._session = None

    # -- configuration -----------------------------------------------------

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        instruction: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        volume: NotGivenOr[float] = NOT_GIVEN,
        language: NotGivenOr[Language | str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            _utils.warn_if_voice_unavailable(voice, self._opts.cluster)
            self._opts.voice = voice
        if is_given(instruction):
            self._opts.instruction = instruction
        if is_given(speed):
            self._opts.speed = speed
        if is_given(volume):
            self._opts.volume = volume
        if is_given(language):
            self._opts.language = language

        # A pooled socket is bound to the model it was opened with (query
        # parameter), so reusing one after a model change would silently
        # synthesize with the old one.
        self._pool.invalidate()

    # -- voices ------------------------------------------------------------

    async def list_voices(self) -> dict[str, Any]:
        """Fetch the official voice catalogue for this cluster.

        Each cluster accepts exactly one `model` value on this endpoint and 400s
        on anything else, which is the one place the two genuinely disagree.
        Cloned voices are account-specific and do not appear here.
        """
        url = f"{self._opts.http_base_url}{SYSTEM_VOICES_PATH}"
        params = {"model": SYSTEM_VOICES_MODEL[self._opts.cluster]}

        try:
            async with self._ensure_session().get(
                url, params=params, headers=_utils.auth_headers(self._api_key)
            ) as resp:
                await _utils.raise_for_status(resp)
                data: dict[str, Any] = await resp.json()
                return data
        except Exception as exc:
            raise _utils.translate_exception(exc, context="stepfun tts list_voices") from exc

    # -- entry points ------------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """One-shot synthesis over HTTP, for whole-utterance playback such as
        session.say(). More robust and cheaper on connections than streaming,
        and the provider notes that stepaudio-2.5-tts in particular sounds
        better here than over the WebSocket.
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        """Streaming synthesis over WebSocket, to synthesize as the LLM streams
        its output. Lower latency.
        """
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream


async def _await_connection_done(ws: _WebSocket) -> str:
    """Read frames until the server issues the session id."""
    while True:
        msg = await ws.receive()

        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
        ):
            raise APIConnectionError("stepfun tts: websocket closed during handshake")

        if msg.type != aiohttp.WSMsgType.TEXT:
            continue

        event = json.loads(msg.data)
        etype = event.get("type")

        if etype == _EV_CONNECTION_DONE:
            session_id = (event.get("data") or {}).get("session_id")
            if not session_id:
                raise APIConnectionError(
                    "stepfun tts: connection.done carried no session_id"
                )
            return str(session_id)

        if etype == _EV_ERROR:
            raise _utils.translate_tts_ws_error(event, context="stepfun tts handshake")

        logger.debug("stepfun tts: ignoring event during handshake", extra={"type": etype})


# ---------------------------------------------------------------------------
# One-shot synthesis
# ---------------------------------------------------------------------------


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        # Snapshot options in __init__, never lazily in _run() — otherwise this
        # races with the next update_options() call.
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload: dict[str, Any] = {
            "model": self._opts.model,
            "voice": self._opts.voice,
            "input": _escape_text(self.input_text, self._opts),
            "response_format": _CHUNKED_RESPONSE_FORMAT,
            "sample_rate": self._opts.sample_rate,
            "speed": self._opts.speed,
            "volume": self._opts.volume,
            "text_normalization": self._opts.text_normalization,
        }
        if self._opts.markdown_filter:
            payload["markdown_filter"] = True
        if is_given(self._opts.language):
            payload["language"] = self._opts.language
        if is_given(self._opts.instruction) and self._opts.supports_instruction:
            payload["instruction"] = self._opts.instruction

        url = f"{self._opts.http_base_url}{TTS_SYNTHESIZE_PATH}"

        try:
            async with self._tts._ensure_session().post(
                url,
                json=payload,
                headers=_utils.auth_headers(self._tts._api_key),
                timeout=aiohttp.ClientTimeout(
                    total=30.0, sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                await _utils.raise_for_status(resp)

                output_emitter.initialize(
                    request_id=resp.headers.get("x-request-id") or utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    # WAV, so the real rate and channel count come from the
                    # header rather than from our declaration.
                    mime_type=_CHUNKED_MIME_TYPE,
                )
                async for chunk, _ in resp.content.iter_chunks():
                    output_emitter.push(chunk)

                output_emitter.flush()
        except Exception as exc:
            raise _utils.translate_exception(exc, context="stepfun tts synthesize") from exc


# ---------------------------------------------------------------------------
# Streaming synthesis
# ---------------------------------------------------------------------------


class SynthesizeStream(tts.SynthesizeStream):
    """One segment per instance, over one WebSocket session.

    Both sides of the contract want it that way. The framework deprecated
    handling several segments in one SynthesizeStream — push() warns and drops
    input past the first segment, directing plugins to pooled connections
    instead. And the provider's session ends at `tts.text.done`, after which it
    releases the socket. So a segment maps exactly onto a session, and segment
    boundaries inside a stream are expressed with `tts.text.flush`, which drains
    buffered audio without ending anything.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._generation_done = False

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        self._generation_done = False

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=_STREAM_MIME_TYPE,
            stream=True,
        )

        async with self._tts._pool.connection(
            timeout=self._conn_options.timeout
        ) as conn:
            self._acquire_time = self._tts._pool.last_acquire_time
            self._connection_reused = self._tts._pool.last_connection_reused

            try:
                await self._create_session(conn)

                # Using the server's session id as the segment id also records
                # it on the trace as a provider request id, so a trace can be
                # lined up against the provider's own logs when debugging.
                output_emitter.start_segment(segment_id=conn.session_id)

                tasks = [
                    asyncio.create_task(
                        self._send_task(conn), name="stepfun_tts_send"
                    ),
                    asyncio.create_task(
                        self._recv_task(conn, output_emitter), name="stepfun_tts_recv"
                    ),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.cancel_and_wait(*tasks)
            finally:
                # The server releases the socket once generation ends, and an
                # interrupted one is left in an unknown state, so it must never
                # go back into the pool. Re-arm so the next turn still starts
                # from a completed handshake.
                self._tts._pool.remove(conn)
                self._tts._pool.prewarm()

    async def _create_session(self, conn: _Connection) -> None:
        """Send tts.create and wait for the server to accept it.

        Generation may only start after `tts.response.created`; sending text
        before that is out of protocol.
        """
        data: dict[str, Any] = {
            "session_id": conn.session_id,
            "voice_id": self._opts.voice,
            "response_format": "pcm",
            "sample_rate": self._opts.sample_rate,
            # Note the suffixed names: the streaming endpoint takes
            # speed_ratio/volume_ratio where the HTTP one takes speed/volume.
            "speed_ratio": self._opts.speed,
            "volume_ratio": self._opts.volume,
            "mode": self._opts.mode,
        }
        if is_given(self._opts.language):
            data["language"] = self._opts.language
        if is_given(self._opts.instruction) and self._opts.supports_instruction:
            data["instruction"] = self._opts.instruction
        if self._opts.markdown_filter:
            data["markdown_filter"] = True
        if self._opts.timestamp:
            data["timestamp"] = True

        await conn.ws.send_str(json.dumps({"type": _EV_CREATE, "data": data}))

        try:
            await asyncio.wait_for(self._await_created(conn), _HANDSHAKE_TIMEOUT)
        except TimeoutError as exc:
            raise APIConnectionError(
                "stepfun tts: timed out waiting for tts.response.created"
            ) from exc

    async def _await_created(self, conn: _Connection) -> None:
        while True:
            msg = await conn.ws.receive()

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                raise APIConnectionError(
                    "stepfun tts: websocket closed before tts.response.created"
                )

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            event = json.loads(msg.data)
            etype = event.get("type")

            if etype == _EV_RESPONSE_CREATED:
                return
            if etype == _EV_ERROR:
                raise _utils.translate_tts_ws_error(event, context="stepfun tts create")

            logger.debug("stepfun tts: ignoring event before created", extra={"type": etype})

    # -- upstream: text ----------------------------------------------------

    async def _send_task(self, conn: _Connection) -> None:
        """Consume self._input_ch, typed as str | _FlushSentinel.

        The sentinel must genuinely reach the wire. Plugins have shipped with a
        bare `continue` here, so the segment-end signal only went out once the
        input channel drained — making stream.flush() a wire-level no-op,
        leaving the provider's per-stream timeout running and ending in a string
        of 408s (livekit/agents issue #6425). It happens to work under
        AgentSession because AgentActivity splits segments above the plugin, but
        callers using tts.stream() + flush() directly get burned.

        Here the sentinel maps to `tts.text.flush` and *not* `tts.text.done`:
        done ends the whole session and makes the server release the socket,
        which would strand any text still arriving behind it.
        """
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                await self._send(conn, _EV_TEXT_FLUSH)
                continue

            text = _escape_text(data, self._opts)
            if not text:
                continue

            for chunk in _split_for_wire(text):
                # The TTFB anchor: _started_time stays 0 until this is called,
                # and _emit_metrics() bails out while it is 0 — so without this
                # the plugin reports no TTS metrics at all.
                self._mark_started()
                await self._send(conn, _EV_TEXT_DELTA, text=chunk)

        # Input is exhausted, so end generation. The server answers with
        # tts.response.audio.done and then closes the socket.
        await self._send(conn, _EV_TEXT_DONE)

    async def _send(self, conn: _Connection, event_type: str, **fields: Any) -> None:
        await conn.ws.send_str(
            json.dumps(
                {
                    "type": event_type,
                    "data": {"session_id": conn.session_id, **fields},
                }
            )
        )

    # -- downstream: audio -------------------------------------------------

    async def _recv_task(
        self, conn: _Connection, output_emitter: tts.AudioEmitter
    ) -> None:
        while True:
            msg = await conn.ws.receive()

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                if self._generation_done:
                    return
                raise APIConnectionError("stepfun tts: websocket closed unexpectedly")

            if msg.type != aiohttp.WSMsgType.TEXT:
                # Audio arrives as base64 inside JSON text frames; the protocol
                # defines no binary frames on this endpoint.
                logger.debug("stepfun tts: unexpected ws frame type %s", msg.type)
                continue

            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("stepfun tts: failed to decode ws frame")
                continue

            if self._handle_event(event, output_emitter):
                return

    def _handle_event(self, event: dict[str, Any], output_emitter: tts.AudioEmitter) -> bool:
        """Returns True once the segment is complete."""
        etype = event.get("type")
        data = event.get("data") or {}

        if etype == _EV_ERROR:
            raise _utils.translate_tts_ws_error(event, context="stepfun tts")

        if etype == _EV_AUDIO_DELTA:
            audio = data.get("audio")
            if audio:
                output_emitter.push(_utils.b64_to_pcm(audio))
            return False

        if etype == _EV_AUDIO_DONE:
            # The payload also carries the whole session's audio, which we
            # ignore: every byte of it already arrived as deltas, and pushing it
            # again would duplicate the utterance.
            self._generation_done = True
            output_emitter.end_segment()
            return True

        if etype == _EV_SUBTITLE:
            self._push_subtitles(data, output_emitter)
            return False

        if etype in (_EV_SENTENCE_START, _EV_SENTENCE_END):
            # Per-sentence boundaries inside our one segment. The framework has
            # no equivalent, and they must not be mapped onto segment
            # boundaries: end_segment() followed by a push raises
            # "start_segment() must be called before pushing audio data"
            # (livekit/agents issue #6929).
            logger.debug(
                "stepfun tts: %s", etype, extra={"request_id": data.get("request_id")}
            )
            return False

        if etype in (_EV_TEXT_FLUSHED, _EV_RESPONSE_CREATED, _EV_CONNECTION_DONE):
            logger.debug("stepfun tts: %s", etype)
            return False

        logger.debug("stepfun tts: unhandled event type %s", etype)
        return False

    def _push_subtitles(
        self, data: dict[str, Any], output_emitter: tts.AudioEmitter
    ) -> None:
        """Translate a subtitle event into timed transcript strings.

        Times are milliseconds and already cumulative across earlier sentences,
        which lines up with one segment per stream. Alignment is asynchronous,
        so an event may arrive after audio for later text; the framework tolerates
        that, and all of them precede audio.done.
        """
        items = data.get("items") or []
        timed = [
            TimedString(
                item.get("text", ""),
                start_time=item.get("start_time", 0) / 1000.0,
                end_time=item.get("end_time", 0) / 1000.0,
            )
            for item in items
            if item.get("text")
        ]
        if timed:
            output_emitter.push_timed_transcript(timed)


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------


def _escape_text(text: str, opts: _TTSOptions) -> str:
    """Neutralize anything downstream text could trip in the directive parser.

    On stepaudio-2.5-tts and stepaudio-3-tts, text inside parentheses is read as
    performance direction and is not spoken. That is a useful authoring feature
    when a human writes the script, and a liability in a LiveKit pipeline, where
    the text is whatever the LLM produced: a stray "(as you mentioned)" would be
    silently swallowed, and a user who can steer the LLM's wording could reach
    the voice settings through it.

    The provider documents no escape sequence, so the only fail-safe action
    available is to remove the delimiters and keep their contents as ordinary
    words — the server then speaks the text instead of interpreting it. That is
    lossy for text that *was* meant as direction, which is why
    `enable_inline_instruction=True` turns it off, and why it is skipped
    entirely on the models that do not parse parentheses at all.
    """
    if opts.enable_inline_instruction or not opts.supports_instruction:
        return text
    return text.translate(_INLINE_DIRECTIVE_CHARS)


def _split_for_wire(text: str) -> list[str]:
    """Split text to respect the per-frame character cap.

    LLM tokens never come close to it, but `session.say()` with a long string
    routed through stream() can, and an over-long frame is rejected outright.
    """
    if len(text) <= TTS_MAX_INPUT_CHARS:
        return [text]
    return [
        text[i : i + TTS_MAX_INPUT_CHARS]
        for i in range(0, len(text), TTS_MAX_INPUT_CHARS)
    ]
