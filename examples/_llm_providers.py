"""Chat-model selection.

STT and TTS stay on StepFun; only the chat model is swappable, and the choice
is made once per process:

    LLM_PROVIDER=stepfun    # default
    LLM_PROVIDER=deepseek

`build_llm()` returns a plain `llm.LLM`, so `AgentSession` and the evaluation
harness need to know nothing about the switch.

Why no per-turn switching: alternating models inside one conversation makes a
run harder to attribute — a transcript is no longer reproducible from a
model name, and latency figures blend two providers. Change the env, restart a
session, and compare runs instead.
"""

from __future__ import annotations

import os
from typing import Any

from _config import get
from livekit.agents import llm

# Model IDs are not validated against a literal here: both vendors add and
# retire IDs faster than this example is updated, and a wrong one surfaces as a
# clear 404 from the provider's own API.

DEFAULT_STEPFUN_MODEL = "step-3.7-flash"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4.1-flash"

#: Models whose API documents `reasoning_effort`.
STEPFUN_REASONING_EFFORT_MODELS = frozenset({"step-5-preview", "step-3.7-flash"})

PROVIDERS = ("stepfun", "deepseek")


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


def build_llm(*, reasoning_effort: str | None = None) -> llm.LLM:
    """Resolve LLM_PROVIDER into a chat model.

    Built eagerly at call time, so a bad key or a missing optional extra fails
    at startup rather than on the caller's first sentence.
    """
    provider = (get("LLM_PROVIDER", "stepfun") or "stepfun").strip().lower()

    if provider == "stepfun":
        model = get("STEPFUN_LLM_MODEL", DEFAULT_STEPFUN_MODEL) or DEFAULT_STEPFUN_MODEL
        return _build_stepfun(model, reasoning_effort)

    if provider == "deepseek":
        model = get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL) or DEFAULT_DEEPSEEK_MODEL
        return _build_deepseek(model, None)

    raise ValueError(f"unknown LLM provider {provider!r}; expected one of {', '.join(PROVIDERS)}")


def describe_llm(instance: llm.LLM) -> dict[str, Any]:
    """Provider/model metadata for the eval config and the startup log."""
    return {"llm_provider": instance.provider, "llm_model": instance.model}
