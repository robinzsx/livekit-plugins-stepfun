"""Pin the chat-model selection.

STT and TTS are deliberately out of scope: they stay on StepFun.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

pytest.importorskip("livekit.plugins.openai")

import _llm_providers as providers  # noqa: E402
from livekit.agents import llm  # noqa: E402


class FakeLLM(llm.LLM):
    """A delegate with no network, for the paths that only need an LLM."""

    def __init__(self, model: str) -> None:
        super().__init__()
        self._model = model

    def chat(self, **_kwargs) -> llm.LLMStream:
        return MagicMock(spec=llm.LLMStream)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._model.split("-")[0]

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "STEPFUN_LLM_MODEL",
        "STEPFUN_REASONING_EFFORT",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_API_KEY",
        "LLM_MIX_PROVIDERS",
        "LLM_MIX_STRATEGY",
        "LLM_MIX_WEIGHTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STEPFUN_API_KEY", "sk-test-stepfun")


# -- stepfun ---------------------------------------------------------------


def test_default_is_stepfun_step_3_7_flash(monkeypatch):
    built = FakeLLM("stepfun")
    captured: dict = {}

    def fake_build(model, effort):
        captured.update({"model": model, "effort": effort})
        return built

    monkeypatch.setattr(providers, "_build_stepfun", fake_build)

    assert providers.build_llm(reasoning_effort="low") is built
    assert captured == {"model": "step-3.7-flash", "effort": "low"}


def test_stepfun_model_is_configurable(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        providers,
        "_build_stepfun",
        lambda model, effort: captured.update({"m": model}) or FakeLLM("stepfun"),
    )
    monkeypatch.setenv("STEPFUN_LLM_MODEL", "step-5-preview")

    providers.build_llm()
    assert captured["m"] == "step-5-preview"


def test_reasoning_effort_is_omitted_for_models_that_do_not_document_it(monkeypatch):
    from livekit.plugins import stepfun

    calls: list[dict] = []
    monkeypatch.setattr(stepfun, "LLM", lambda **kw: calls.append(kw) or FakeLLM("stepfun"))

    providers._build_stepfun("step-3.5-flash", "high")
    assert calls[0]["model"] == "step-3.5-flash"
    assert "reasoning_effort" not in calls[0]


def test_reasoning_effort_none_is_not_forwarded(monkeypatch):
    """`none` means "let the provider decide", not "send the string none"."""
    from livekit.plugins import stepfun

    calls: list[dict] = []
    monkeypatch.setattr(stepfun, "LLM", lambda **kw: calls.append(kw) or FakeLLM("stepfun"))

    providers._build_stepfun("step-3.7-flash", "none")
    assert "reasoning_effort" not in calls[0]


@pytest.mark.parametrize("model", ["step-5-preview", "step-3.7-flash"])
def test_reasoning_effort_is_forwarded_for_every_model_that_documents_it(monkeypatch, model):
    """Keep this in step with REASONING_EFFORT_LLM_MODELS in the plugin."""
    from livekit.plugins import stepfun

    calls: list[dict] = []
    monkeypatch.setattr(stepfun, "LLM", lambda **kw: calls.append(kw) or FakeLLM("stepfun"))

    providers._build_stepfun(model, "high")
    assert calls[0]["reasoning_effort"] == "high"


def test_reasoning_effort_is_forwarded_for_a_supported_model(monkeypatch):
    from livekit.plugins import stepfun

    calls: list[dict] = []
    monkeypatch.setattr(stepfun, "LLM", lambda **kw: calls.append(kw) or FakeLLM("stepfun"))

    providers._build_stepfun("step-3.7-flash", "high")
    assert calls[0]["reasoning_effort"] == "high"


# -- deepseek --------------------------------------------------------------


def test_deepseek_is_built_through_the_openai_compatible_client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")

    from livekit.plugins import openai

    inner = FakeLLM("deepseek-reasoner")
    with_deepseek = MagicMock(return_value=inner)
    monkeypatch.setattr(openai.LLM, "with_deepseek", with_deepseek)

    built = providers.build_llm(reasoning_effort="low")

    with_deepseek.assert_called_once_with(model="deepseek-reasoner", api_key="sk-deepseek")
    assert built.model == "deepseek-reasoner"
    # The OpenAI client would otherwise report `api.deepseek.com`, the endpoint
    # rather than the vendor.
    assert built.provider == "DeepSeek"


def test_deepseek_has_a_documented_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    from livekit.plugins import openai

    with_deepseek = MagicMock(return_value=FakeLLM("deepseek"))
    monkeypatch.setattr(openai.LLM, "with_deepseek", with_deepseek)

    providers.build_llm()
    assert with_deepseek.call_args.kwargs["model"] == "deepseek-v4.1-flash"


def test_deepseek_without_a_key_fails_loudly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY"):
        providers.build_llm()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="unknown LLM provider"):
        providers.build_llm()


def test_mix_is_no_longer_accepted(monkeypatch):
    """Per-turn switching was removed; the old value must not silently pass."""
    monkeypatch.setenv("LLM_PROVIDER", "mix")
    with pytest.raises(ValueError, match="unknown LLM provider"):
        providers.build_llm()


# -- reporting -------------------------------------------------------------


def test_describe_llm_reports_one_provider():
    assert providers.describe_llm(FakeLLM("deepseek-v4.1-flash")) == {
        "llm_provider": "deepseek",
        "llm_model": "deepseek-v4.1-flash",
    }
