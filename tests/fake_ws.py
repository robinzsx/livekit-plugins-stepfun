"""Scripted WebSocket responses and connection-pool lifecycle tracking."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Generic, TypeVar

import aiohttp


class FakeWSMessage:
    def __init__(self, type_: aiohttp.WSMsgType, data: Any) -> None:
        self.type = type_
        self.data = data
        self.extra = None


class FakeWebSocket:
    """A good-enough stand-in for aiohttp.ClientWebSocketResponse."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.close_code: int | None = None
        self._incoming: asyncio.Queue[FakeWSMessage] = asyncio.Queue()

    # -- the interface the plugin calls ------------------------------------

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def ping(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def receive(self) -> FakeWSMessage:
        return await self._incoming.get()

    # -- the interface tests drive -----------------------------------------

    def queue_server_event(self, event: dict[str, Any]) -> None:
        self._incoming.put_nowait(FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event)))

    def queue_binary(self, data: bytes) -> None:
        self._incoming.put_nowait(FakeWSMessage(aiohttp.WSMsgType.BINARY, data))

    def queue_close(self) -> None:
        self._incoming.put_nowait(FakeWSMessage(aiohttp.WSMsgType.CLOSED, None))

    def queue_error(self) -> None:
        self._incoming.put_nowait(
            FakeWSMessage(aiohttp.WSMsgType.ERROR, ConnectionResetError("connection reset"))
        )

    # -- assertion helpers -------------------------------------------------

    def frames_of_type(self, type_: str) -> list[dict[str, Any]]:
        """Every frame we sent with the given ``type``."""
        return [f for f in self.sent if f.get("type") == type_]

    def data_of_type(self, type_: str) -> list[dict[str, Any]]:
        """The ``data`` payloads of frames with the given ``type``.

        The streaming TTS protocol wraps every field in ``data``, so asserting
        on the payload directly keeps tests readable.
        """
        return [f.get("data") or {} for f in self.sent if f.get("type") == type_]

    def types_sent(self) -> list[str]:
        return [f.get("type", "") for f in self.sent]


T = TypeVar("T")


class FakePool(Generic[T]):
    """Stands in for ``utils.ConnectionPool``.

    Hands out one preconfigured connection and records whether the plugin
    returned it to the pool or dropped it. Both streaming surfaces are supposed
    to drop theirs, since neither socket is safe to reuse afterwards.
    """

    def __init__(self, conn: T) -> None:
        self._conn = conn
        self.last_acquire_time = 0.25
        self.last_connection_reused = False
        self.removed: list[T] = []
        self.prewarm_calls = 0
        self.invalidate_calls = 0

    @asynccontextmanager
    async def connection(self, *, timeout: float) -> AsyncIterator[T]:
        yield self._conn

    def remove(self, conn: T) -> None:
        self.removed.append(conn)

    def prewarm(self) -> None:
        self.prewarm_calls += 1

    def invalidate(self) -> None:
        self.invalidate_calls += 1

    async def aclose(self) -> None:
        pass
