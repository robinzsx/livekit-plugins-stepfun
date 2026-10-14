"""StepFun chat models, over the OpenAI-compatible endpoint.

Unlike the ASR and TTS sides, there is no provider-specific protocol work here:
StepFun's `/v1/chat/completions` is OpenAI-shaped, so this is a thin subclass of
`livekit-plugins-openai`'s LLM that pins the endpoint, resolves credentials and
the cluster the same way the rest of this plugin does, and declares which model
strings exist.

That is deliberate. Reimplementing a chat client to change a hostname and a
default model would be a large amount of code with nothing provider-specific in
it, and it would drift from the OpenAI plugin's own handling of streaming,
tool-call aggregation and usage accounting.

**On reasoning models.** `step-5-preview` and `step-3.7-flash` return their
thinking in a separate `reasoning` field. The framework's chat parser reads only
`delta.content`, `delta.tool_calls` and `delta.extra_content`, so `reasoning` is
dropped before it can reach the TTS — the agent never narrates its own thinking,
which is what you want in a voice pipeline. The cost is that the reasoning trace
is not available for logging either; recovering it would mean overriding private
framework internals, so it is deliberately not attempted here.

What reasoning models *do* cost you is latency: the thinking happens before the
first speakable token, so the user hears silence for its whole duration. Prefer
`reasoning_effort="low"` for conversation, and measure time to first
*speakable* content rather than time to first token.
"""

from __future__ import annotations

from typing import Any

from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from . import _utils
from .log import logger
from .models import (
    DEFAULT_LLM_MODEL,
    REASONING_EFFORT_LLM_MODELS,
    Cluster,
    LLMModels,
    ReasoningEffort,
)

try:
    # Imported as a module rather than `from livekit.plugins import openai`:
    # livekit.plugins is a namespace package, so the attribute form makes type
    # checkers resolve it against the namespace instead of the optional
    # distribution, and it cannot be excluded cleanly when absent.
    import livekit.plugins.openai as _openai
except ImportError as exc:  # pragma: no cover - exercised by the install, not tests
    raise ImportError(
        "stepfun.LLM requires the OpenAI plugin, which provides the "
        "OpenAI-compatible chat client it builds on.\n"
        "    pip install 'livekit-plugins-stepfun[llm]'\n"
        "or\n"
        "    pip install livekit-plugins-openai\n"
        "The STT and TTS components do not need it."
    ) from exc


class LLM(_openai.LLM):  # type: ignore[misc, unused-ignore]
    """StepFun chat completion models.

    The ignore above carries two codes on purpose. ``misc`` is needed when the
    OpenAI plugin is *not* installed: the base class then resolves to ``Any``
    and subclassing it is an error under strict mode. ``unused-ignore`` is
    needed when it *is* installed, because the first code then has nothing to
    suppress and ``warn_unused_ignores`` would flag it. Both configurations are
    supported, so both are silenced here rather than by weakening the global
    mypy settings.

    Usage::

        from livekit.plugins import stepfun

        session = AgentSession(
            stt=stepfun.STT(),
            llm=stepfun.LLM(model="step-3.7-flash", reasoning_effort="low"),
            tts=stepfun.TTS(),
        )
    """

    def __init__(
        self,
        *,
        model: LLMModels | str = DEFAULT_LLM_MODEL,
        api_key: str | None = None,
        cluster: Cluster | None = None,
        base_url: str | None = None,
        reasoning_effort: NotGivenOr[ReasoningEffort] = NOT_GIVEN,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: Chat model ID. Defaults to `step-3.7-flash` rather than
                `step-5-preview`: the latter is a 1M-context reasoning model
                built for agentic and software-engineering work, and its
                thinking time lands entirely before the agent can start
                speaking.
            api_key: Falls back to `STEPFUN_API_KEY`. Credentials are not
                interchangeable between clusters — a domestic key returns 401
                against the international endpoint — so this resolves exactly
                the same way as it does for STT and TTS.
            cluster: `international` (default, `api.stepfun.ai`) or `domestic`
                (`api.stepfun.com`). Also read from `STEPFUN_CLUSTER`.
            base_url: Overrides the cluster entirely. The `/v1` suffix the
                OpenAI client expects is appended if absent.
            reasoning_effort: `low`, `medium` or `high`. Only `step-3.7-flash`
                is documented as supporting the three levels, so a warning is
                logged for other models rather than silently assuming it works.
            kwargs: Forwarded to the underlying OpenAI-compatible LLM —
                `temperature`, `tool_choice`, `max_completion_tokens` and so on.
        """
        resolved_cluster = _utils.resolve_cluster(cluster)
        resolved_key = _utils.resolve_api_key(api_key)
        resolved_url = _utils.resolve_llm_base_url(base_url, resolved_cluster)

        if is_given(reasoning_effort):
            if model not in REASONING_EFFORT_LLM_MODELS:
                logger.warning(
                    "reasoning_effort is only documented for %s; the server may "
                    "ignore it on this model",
                    ", ".join(sorted(REASONING_EFFORT_LLM_MODELS)),
                    extra={"model": model},
                )
            kwargs["reasoning_effort"] = reasoning_effort

        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=resolved_url,
            **kwargs,
        )

        self._stepfun_cluster = resolved_cluster

    @property
    def provider(self) -> str:
        # The parent reports the base URL's host, which is accurate but reads as
        # an endpoint rather than a vendor in metrics and traces.
        return "StepFun"

    @property
    def cluster(self) -> Cluster:
        return self._stepfun_cluster
