"""LLM provider selection, with optional per-turn mixing.

STT and TTS stay on StepFun; only the chat model is swappable here. The entry
point is `build_llm()`, which reads the environment once and returns a single
`llm.LLM` for `AgentSession`:

    LLM_PROVIDER=stepfun              # default, and the only behaviour prior to this
    LLM_PROVIDER=deepseek
    LLM_PROVIDER=mix                  # alternate per turn, see LLM_MIX_STRATEGY

The mixing router is a real `llm.LLM` subclass rather than something the session
has to know about: `AgentSession.llm` is read-only and the SDK resolves it once
per turn, so a wrapper that picks a delegate inside `chat()` is the only place
the swap can happen without touching the SDK or the agent.
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from _config import get
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, MetricsCollectedEvent, llm
from livekit.agents.types import NOT_GIVEN, NotGivenOr

# Model IDs are not validated against a literal here: both vendors add and
# retire IDs faster than this example is updated, and a wrong one surfaces as a
# clear 404 from the provider's own API.

DEFAULT_STEPFUN_MODEL = "step-3.7-flash"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

#: `reasoning_effort` is only documented for StepFun's step-3.7-flash.
STEPFUN_REASONING_EFFORT_MODELS = frozenset({"step-3.7-flash"})

MixStrategy = str  # "round_robin" | "weighted"


@dataclass(frozen=True)
class LLMChoice:
    """One buildable chat model, plus what the example needs to describe it.

    `key` is the selector token used in env vars; `provider` is the vendor label
    that goes into metrics, matching what each plugin reports for itself so a
    mixed run is not split across `stepfun` and `StepFun`.
    """

    key: str  # "stepfun" | "deepseek"
    provider: str  # "StepFun" | "DeepSeek"
    model: str
    factory: Callable[[], llm.LLM]

    def describe(self) -> dict[str, Any]:
        return {"llm_provider": self.provider, "llm_model": self.model}


def _build_stepfun(model: str, reasoning_effort: str | None) -> llm.LLM:
    # Imported lazily so a deepseek-only run still starts if the stepfun chat
    # extra is absent; `stepfun` itself is importable for STT/TTS regardless.
    from livekit.plugins import stepfun

    kwargs: dict[str, Any] = {"model": model}
    # Sending it to a model that does not document it would be silently ignored
    # server-side, so omit rather than lie in the config block.
    if reasoning_effort and reasoning_effort != "none" and model in STEPFUN_REASONING_EFFORT_MODELS:
        kwargs["reasoning_effort"] = reasoning_effort
    return stepfun.LLM(**kwargs)


def _build_deepseek(model: str, api_key: str | None) -> llm.LLM:
    # `with_deepseek` is a plain constructor on the OpenAI plugin: same protocol,
    # different base_url. No extra dependency beyond what the examples already
    # install.
    from livekit.plugins import openai

    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit(
            "LLM_PROVIDER=deepseek requires DEEPSEEK_API_KEY.\n"
            "  Export it or add it to .env, then restart."
        )
    return _DeepSeekLLM(openai.LLM.with_deepseek(model=model, api_key=key))


class _DeepSeekLLM(llm.LLM):
    """Reports DeepSeek as the vendor.

    `openai.LLM.with_deepseek` reports the base URL's host, so metrics and
    traces would label the run `api.deepseek.com`. That is the endpoint, not the
    vendor, and it makes eval results awkward to group by. Same reason
    `stepfun.LLM` overrides `provider`.
    """

    def __init__(self, inner: llm.LLM) -> None:
        super().__init__()
        self._inner = inner

    def chat(self, **kwargs: Any) -> llm.LLMStream:
        return self._inner.chat(**kwargs)

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return "DeepSeek"

    def prewarm(self, *, loop: Any = None) -> None:
        self._inner.prewarm(loop=loop)

    async def aclose(self) -> None:
        await self._inner.aclose()


def _choice(provider: str, *, reasoning_effort: str | None) -> LLMChoice:
    if provider == "stepfun":
        model = get("STEPFUN_LLM_MODEL", DEFAULT_STEPFUN_MODEL) or DEFAULT_STEPFUN_MODEL
        return LLMChoice(
            key="stepfun",
            provider="StepFun",
            model=model,
            factory=lambda: _build_stepfun(model, reasoning_effort),
        )
    if provider == "deepseek":
        model = get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL) or DEFAULT_DEEPSEEK_MODEL
        return LLMChoice(
            key="deepseek",
            provider="DeepSeek",
            model=model,
            factory=lambda: _build_deepseek(model, None),
        )
    raise ValueError(f"unknown LLM provider {provider!r}; expected stepfun, deepseek or mix")


def _parse_weights(spec: str | None, providers: list[str]) -> list[float]:
    """`stepfun:3,deepseek:1` -> per-turn probabilities in `providers` order."""
    if not spec:
        return [1.0] * len(providers)

    weights = {p: 0.0 for p in providers}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, raw = part.partition(":")
        name = name.strip()
        if name not in weights:
            raise ValueError(
                f"LLM_MIX_WEIGHTS names {name!r}, which is not in LLM_MIX_PROVIDERS; "
                f"expected a subset of {', '.join(providers)}"
            )
        try:
            value = float(raw or "1")
        except ValueError as exc:
            raise ValueError(f"LLM_MIX_WEIGHTS: {part!r} does not parse") from exc
        if value < 0:
            raise ValueError(f"LLM_MIX_WEIGHTS: {part!r} must not be negative")
        weights[name] = value

    if not any(weights.values()):
        raise ValueError("LLM_MIX_WEIGHTS must give at least one provider a positive weight")
    return [weights[p] for p in providers]


class MixingLLM(llm.LLM):
    """Delegates each turn to one of several chat models.

    A turn is one `chat()` call, which is exactly the unit the SDK asks for a
    reply — so the choice is per-turn without needing any session surgery. Tool
    calls inside a turn stay on the model that started it: the SDK re-enters
    `chat()` for a follow-up tool turn, so a multi-step exchange may switch
    mid-conversation. That is visible in metrics (`model_name` per turn) and is
    the intended behaviour of "mix per turn" rather than a bug.
    """

    def __init__(self, choices: list[LLMChoice], *, strategy: MixStrategy, weights: list[float]):
        super().__init__()
        if not choices:
            raise ValueError("MixingLLM needs at least one choice")
        self._choices = choices
        self._strategy = strategy
        self._weights = weights
        self._cursor = 0
        # Build eagerly: a bad key or missing optional extra should fail at
        # startup, not on the caller's first sentence.
        self._delegates: list[llm.LLM] = [c.factory() for c in choices]
        self._last_index = 0
        # An LLMStream emits metrics on the LLM it was constructed with, which
        # is the delegate, not this router. Re-emit so listeners attached to
        # the router — the eval controller, and anything else that treats it as
        # `session.llm` — still receive them.
        for delegate in self._delegates:
            delegate.on("metrics_collected", self._forward_metrics)

    # -- selection ---------------------------------------------------------

    def _next_index(self) -> int:
        if self._strategy == "round_robin":
            index = self._cursor % len(self._choices)
            self._cursor += 1
            return index

        # random.choices gives the weighted draw without a cumulative table.
        # The RNG is process-global, which is fine: the draw only needs to be
        # unbiased across turns, not independently seeded per session.
        return random.choices(range(len(self._choices)), weights=self._weights, k=1)[0]

    def _pick(self) -> tuple[LLMChoice, llm.LLM]:
        index = self._next_index()
        self._last_index = index
        return self._choices[index], self._delegates[index]

    def _forward_metrics(self, event: MetricsCollectedEvent) -> None:
        # The delegate already labels its metrics with its own model/provider;
        # this only makes them reachable from a listener on the router, which is
        # what `session.llm` is.
        self.emit("metrics_collected", event)

    # -- llm.LLM API -------------------------------------------------------

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: Any = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        _, delegate = self._pick()
        return delegate.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )

    @property
    def model(self) -> str:
        # Reported after selection, so metrics and traces label the model that
        # actually answered this turn. Before the first turn it names the first
        # configured model.
        return self._choices[self._last_index].model

    @property
    def provider(self) -> str:
        return self._choices[self._last_index].provider

    def roster(self) -> list[dict[str, Any]]:
        """Every configured model, for `describe_llm` and startup logging."""
        return [c.describe() for c in self._choices]

    def prewarm(self, *, loop: Any = None) -> None:
        # Every delegate may be used within the first few turns, so pay the
        # connection setup for all of them rather than for one.
        if loop is None:
            # `LLM.prewarm` falls back to `asyncio.get_event_loop()`, which
            # raises rather than creating a loop on 3.14+. Resolve once here
            # and hand it down, so a prewarm outside a running loop degrades to
            # "no prewarm" instead of taking the process down at startup.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        for delegate in self._delegates:
            delegate.prewarm(loop=loop)

    async def aclose(self) -> None:
        for delegate in self._delegates:
            await delegate.aclose()

    @property
    def choices(self) -> list[LLMChoice]:
        return list(self._choices)

    def __iter__(self) -> Iterator[LLMChoice]:
        return iter(self._choices)


def build_llm(*, reasoning_effort: str | None = None) -> llm.LLM:
    """Resolve LLM_PROVIDER into a single chat model, or a mixing router."""
    provider = (get("LLM_PROVIDER", "stepfun") or "stepfun").strip().lower()

    if provider != "mix":
        return _choice(provider, reasoning_effort=reasoning_effort).factory()

    names = [
        n.strip().lower()
        for n in (get("LLM_MIX_PROVIDERS", "stepfun,deepseek") or "").split(",")
        if n.strip()
    ]
    if not names:
        raise ValueError("LLM_PROVIDER=mix requires LLM_MIX_PROVIDERS")

    strategy = (get("LLM_MIX_STRATEGY", "round_robin") or "round_robin").strip().lower()
    if strategy not in ("round_robin", "weighted"):
        raise ValueError("LLM_MIX_STRATEGY must be 'round_robin' or 'weighted'")

    choices = [_choice(name, reasoning_effort=reasoning_effort) for name in names]
    # Duplicates would make one provider's round-robin turns silently skew, and
    # it is always a typo rather than intent.
    if len({c.key for c in choices}) != len(choices):
        raise ValueError(f"LLM_MIX_PROVIDERS lists a provider twice: {get('LLM_MIX_PROVIDERS')!r}")
    weights = _parse_weights(get("LLM_MIX_WEIGHTS"), names)
    return MixingLLM(choices, strategy=strategy, weights=weights)


def describe_llm(instance: llm.LLM) -> dict[str, Any]:
    """Provider/model metadata for the eval config.

    For a router this records the whole roster, since a single model name would
    misdescribe the run.
    """
    if isinstance(instance, MixingLLM):
        return {
            "llm_provider": "mix",
            "llm_strategy": instance._strategy,
            "llm_weights": instance._weights,
            "llm_models": instance.roster(),
        }
    return {"llm_provider": instance.provider, "llm_model": instance.model}
