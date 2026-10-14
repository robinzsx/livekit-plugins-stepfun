from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Any, get_args

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.utils import shortuuid

from .log import logger
from .models import (
    CLUSTER_HTTP_BASE_URLS,
    CLUSTER_WS_BASE_URLS,
    DEFAULT_CLUSTER,
    INTERNATIONAL_TTS_VOICES,
    OPENAI_COMPAT_SUFFIX,
    Cluster,
    TTSVoices,
)

_KNOWN_TTS_VOICES = frozenset(get_args(TTSVoices))

# ---------------------------------------------------------------------------
# Credentials and endpoints
# ---------------------------------------------------------------------------


def resolve_api_key(api_key: str | None) -> str:
    """Explicit argument wins, then the environment variable.

    Note that the two clusters issue separate credentials: a domestic key will
    not authenticate against the international endpoint, or vice versa.
    """
    key = api_key or os.environ.get("STEPFUN_API_KEY")
    if not key:
        raise ValueError(
            "StepFun API key is required. "
            "Pass api_key=... or set the STEPFUN_API_KEY environment variable."
        )
    return key


def resolve_cluster(cluster: Cluster | None) -> Cluster:
    if cluster is not None:
        return cluster

    env = os.environ.get("STEPFUN_CLUSTER")
    if env in ("international", "domestic"):
        return env  # type: ignore[return-value]
    if env:
        raise ValueError(
            f"STEPFUN_CLUSTER must be 'international' or 'domestic', got {env!r}"
        )
    return DEFAULT_CLUSTER


def resolve_ws_base_url(base_url: str | None, cluster: Cluster) -> str:
    return (base_url or CLUSTER_WS_BASE_URLS[cluster]).rstrip("/")


def resolve_http_base_url(base_url: str | None, cluster: Cluster) -> str:
    return (base_url or CLUSTER_HTTP_BASE_URLS[cluster]).rstrip("/")


def resolve_llm_base_url(base_url: str | None, cluster: Cluster) -> str:
    """Base URL for the OpenAI-compatible chat client.

    That client expects the API version in the base URL rather than in each
    request path, so the suffix is appended here. An explicit base_url that
    already carries it is left alone, since a doubled `/v1/v1` 404s in a way
    that is annoying to diagnose.
    """
    resolved = resolve_http_base_url(base_url, cluster)
    if resolved.endswith(OPENAI_COMPAT_SUFFIX):
        return resolved
    return f"{resolved}{OPENAI_COMPAT_SUFFIX}"


def auth_headers(api_key: str) -> dict[str, str]:
    """The protocol requires the API key in a WebSocket header.

    That carries an ecosystem limitation worth knowing: browsers cannot set
    headers on a native WebSocket, so frontend code cannot connect directly
    (StepFun's own Step-Realtime-Console demo has to proxy through a server for
    exactly this reason). It is fine for a Python agent, which is a server-side
    process, but supporting browser-based agents-js setups later will require
    the server to accept a query parameter or an ephemeral token instead.
    """
    return {"Authorization": f"Bearer {api_key}"}


def warn_if_voice_unavailable(voice: TTSVoices | str, cluster: Cluster) -> None:
    """Warn when a voice is known to be domestic-only but used internationally.

    No ID translation happens here, and none is needed: the international
    catalogue is a strict subset of the domestic one rather than a different
    naming scheme. This only turns the server's opaque `400 voice_id_invalid`
    into something actionable ahead of the request.

    Deliberately a warning rather than an error: cloned voices are
    account-specific and will never appear in a static list, so rejecting
    anything unrecognized here would break legitimate callers.
    """
    if cluster != "international":
        return
    if voice in INTERNATIONAL_TTS_VOICES:
        return
    if voice not in _KNOWN_TTS_VOICES:
        # Unrecognized entirely — most likely a cloned voice, which is
        # account-specific and legitimately absent from any static list.
        return

    logger.warning(
        "voice is not published for the international cluster; the request will "
        "likely fail with 400 voice_id_invalid. Use TTS.list_voices() to see "
        "what is available.",
        extra={"voice": voice, "cluster": cluster},
    )


def event_id() -> str:
    """Every client frame on the ASR WebSocket carries its own event_id."""
    return shortuuid()


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------


def pcm_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def b64_to_pcm(data: str) -> bytes:
    return base64.b64decode(data)


# ---------------------------------------------------------------------------
# Server-sent events
# ---------------------------------------------------------------------------


async def iter_sse_events(resp: aiohttp.ClientResponse) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded JSON payloads from an SSE response.

    Only the `data:` field is of interest here; the batch ASR route carries the
    event type inside the JSON rather than in an SSE `event:` line. A literal
    `[DONE]` terminates the stream, following the convention the TTS SSE route
    uses.
    """
    async for raw in resp.content:
        line = raw.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue

        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            if payload == "[DONE]":
                return
            continue

        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("stepfun: failed to decode sse payload")


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------
#
# This table is the only input to FallbackAdapter's failover decisions. Getting
# `retryable` wrong is worse than leaving it unset: marking an auth failure as
# retryable makes a customer's fallback chain hammer a provider that can never
# succeed instead of switching to the backup.
#
# Second rule: this plugin never retries internally. The framework already
# retries according to APIConnectOptions.max_retry, and stacking another layer
# on top produces multiplicative amplification (3 x 3 = 9 requests) where only
# the framework's three attempts are visible in the logs. LiveKit hit this and
# fixed it deliberately (PR #5335) — the OpenAI plugin now forces the SDK's
# max_retries to 0. Our aiohttp calls configure no retries at all.

#: HTTP status codes worth retrying.
#:
#: 400 (bad parameters), 401 (bad key), 402 (no balance), 404 (wrong path) and
#: 451 (content moderation) all require the caller to change something, so a
#: retry can only fail the same way.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: Error identifiers that make a 429 *non*-retryable.
#:
#: 429 is overloaded on this API: ordinary rate limiting (retry works), or a
#: project/member Credit ceiling (retry cannot work until the limit is raised or
#: the quota resets on the 1st of next month). Treating the latter as retryable
#: would keep a FallbackAdapter chain pinned to a provider that is guaranteed to
#: fail for the rest of the month instead of moving on.
_QUOTA_ERROR_IDENTIFIERS = (
    "insufficient_credit",
    "project_credit_limit_exceeded",
    "member_project_credit_limit_exceeded",
)

#: WebSocket error codes on the streaming ASR endpoint that are worth retrying.
#: Both timeout codes mean the connection went stale, which reconnecting fixes.
_RETRYABLE_WS_CODES = frozenset({"internal_error", "max_idle_timeout", "pong_timeout"})


def _is_quota_exhaustion(status_code: int, body: Any) -> bool:
    if status_code != 429 or not body:
        return False
    text = body if isinstance(body, str) else str(body)
    return any(marker in text for marker in _QUOTA_ERROR_IDENTIFIERS)


def status_error(
    *,
    status_code: int,
    message: str,
    request_id: str | None = None,
    body: Any = None,
) -> APIStatusError:
    retryable = status_code in _RETRYABLE_STATUS and not _is_quota_exhaustion(
        status_code, body
    )
    return APIStatusError(
        message,
        status_code=status_code,
        request_id=request_id,
        body=body,
        retryable=retryable,
    )


async def raise_for_status(resp: aiohttp.ClientResponse) -> None:
    """Single error exit point for the HTTP paths."""
    if resp.status < 400:
        return

    request_id = resp.headers.get("x-request-id")
    try:
        body = await resp.text()
    except Exception:  # noqa: BLE001 - a failed body read must not mask the real error
        body = None

    raise status_error(
        status_code=resp.status,
        message=f"StepFun API error: {resp.status}",
        request_id=request_id,
        body=body,
    )


def translate_exception(exc: BaseException, *, context: str) -> BaseException:
    """Translate aiohttp/asyncio exceptions into LiveKit's standard exceptions.

    Already-translated errors pass straight through, so wrapping a call site
    that itself called `raise_for_status` does not downgrade a precise
    APIStatusError into a generic connection error.
    """
    if isinstance(exc, (APIStatusError, APIConnectionError, APITimeoutError)):
        return exc
    if isinstance(exc, TimeoutError):
        return APITimeoutError(f"{context}: timed out")
    if isinstance(exc, aiohttp.ClientResponseError):
        return status_error(
            status_code=exc.status,
            message=f"{context}: {exc.message}",
            request_id=None,
        )
    if isinstance(exc, aiohttp.ClientError):
        return APIConnectionError(f"{context}: {exc}")
    return exc


def translate_asr_ws_error(frame: dict[str, Any], *, context: str) -> BaseException:
    """Translate an `error` frame from the streaming ASR WebSocket.

        {"event_id": ..., "type": "error", "meta": {...},
         "error": {"type": ..., "code": ..., "message": ..., "param": ...}}

    `error.type` is one of invalid_request_error / internal_error / risk;
    `error.code` is one of invalid_value / missing_param / internal_error /
    max_idle_timeout / pong_timeout / risk_blocked.
    """
    err = frame.get("error") or {}
    code = err.get("code")
    message = err.get("message") or "unknown error"
    meta = frame.get("meta") or {}

    logger.warning(
        "stepfun asr: server error frame",
        extra={
            "context": context,
            "code": code,
            "error_type": err.get("type"),
            "param": err.get("param"),
            "session_id": meta.get("session_id"),
        },
    )

    return APIStatusError(
        f"{context}: {message}",
        status_code=-1,
        request_id=meta.get("session_id"),
        body=frame,
        retryable=code in _RETRYABLE_WS_CODES,
    )


def translate_tts_ws_error(frame: dict[str, Any], *, context: str) -> BaseException:
    """Translate a `tts.response.error` frame from the streaming TTS WebSocket.

        {"event_id": ..., "type": "tts.response.error",
         "data": {"session_id": ..., "code": "503", "message": ..., "details": {...}}}

    Unlike the ASR surface, `code` here is an HTTP status rendered as a string,
    so retryability comes from the HTTP table.
    """
    data = frame.get("data") or {}
    raw_code = data.get("code")
    message = data.get("message") or "unknown error"

    try:
        status_code = int(str(raw_code))
    except (TypeError, ValueError):
        status_code = -1

    logger.warning(
        "stepfun tts: server error frame",
        extra={
            "context": context,
            "code": raw_code,
            "session_id": data.get("session_id"),
        },
    )

    if status_code < 0:
        # An unparseable code carries no retry semantics. Treat it as terminal so
        # a FallbackAdapter moves on rather than retrying an unknown failure.
        return APIStatusError(
            f"{context}: {message}",
            status_code=-1,
            request_id=data.get("session_id"),
            body=frame,
            retryable=False,
        )

    return status_error(
        status_code=status_code,
        message=f"{context}: {message}",
        request_id=data.get("session_id"),
        body=frame,
    )
