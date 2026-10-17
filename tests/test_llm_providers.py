"""Pin the LLM provider switch and the per-turn mixing router.

STT and TTS are deliberately out of scope: they stay on StepFun, and
`tests/test_voice_agent.py` already covers how they are wired.
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
    """A delegate with a real EventEmitter and no network.

    Subclassing rather than mocking matters: the router re-emits metrics from
    the delegate's own emitter, and that test only proves anything if the
    emitter actually dispatches.
    """

    def __init__(self, model: str) -> None:
        super().__init__()
        self._model = model
        self.chat_calls: list[llm.ChatContext] = []
        self.prewarm_calls = 0

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools=None,
        conn_options=None,
        parallel_tool_calls=None,
        tool_choice=None,
        extra_kwargs=None,
    ) -> llm.LLMStream:
        self.chat_calls.append(chat_ctx)
        return MagicMock(spec=llm.LLMStream)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._model.split("-")[0]

    def prewarm(self, *, loop=None) -> None:
        self.prewarm_calls += 1

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


# -- single provider -------------------------------------------------------


def test_default_is_stepfun_step_3_7_flash(monkeypatch):
    built = FakeLLM("stepfun")
    captured: dict = {}

    def fake_build(model, effort):
        captured.update({"model": model, "effort": effort})
        return built

    monkeypatch.setattr(providers, "_build_stepfun", fake_build)

    assert providers.build_llm(reasoning_effort="low") is built
    assert captured == {"model": "step-3.7-flash", "effort": "low"}


def test_reasoning_effort_is_omitted_for_models_that_do_not_document_it(monkeypatch):
    captured: dict = {}

    def fake_build(model, effort):
        captured.update({"model": model, "effort": effort})
        return FakeLLM("stepfun")

    monkeypatch.setattr(providers, "_build_stepfun", fake_build)
    monkeypatch.setenv("STEPFUN_LLM_MODEL", "step-3.5-flash")

    providers.build_llm(reasoning_effort="high")
    assert captured["model"] == "step-3.5-flash"


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


def test_deepseek_defaults_to_deepseek_chat(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    from livekit.plugins import openai

    with_deepseek = MagicMock(return_value=FakeLLM("deepseek"))
    monkeypatch.setattr(openai.LLM, "with_deepseek", with_deepseek)

    providers.build_llm()
    assert with_deepseek.call_args.kwargs["model"] == "deepseek-chat"


def test_deepseek_without_a_key_fails_loudly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY"):
        providers.build_llm()


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        providers._choice("openai", reasoning_effort=None)


# -- mixing ----------------------------------------------------------------


def _mix(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LLM_PROVIDER", "mix")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    # Both builders are replaced: the mix tests are about selection, and the
    # real constructors are covered by the single-provider tests above. Keeps
    # the delegates offline and gives each a real EventEmitter.
    monkeypatch.setattr(
        providers, "_build_stepfun", lambda model, effort: FakeLLM("step-3.7-flash")
    )
    monkeypatch.setattr(providers, "_build_deepseek", lambda model, key: FakeLLM("deepseek-chat"))


def test_round_robin_alternates_per_turn(monkeypatch):
    _mix(monkeypatch)
    router = providers.build_llm(reasoning_effort="low")
    assert isinstance(router, providers.MixingLLM)

    seen = []
    for _ in range(4):
        router.chat(chat_ctx=llm.ChatContext.empty())
        seen.append((router.provider, router.model))

    assert seen == [
        ("StepFun", "step-3.7-flash"),
        ("DeepSeek", "deepseek-chat"),
        ("StepFun", "step-3.7-flash"),
        ("DeepSeek", "deepseek-chat"),
    ]


def test_chat_delegates_to_the_selected_model(monkeypatch):
    _mix(monkeypatch)
    router = providers.build_llm()
    stepfun_delegate, deepseek_delegate = router._delegates

    ctx = llm.ChatContext.empty()
    router.chat(chat_ctx=ctx)
    router.chat(chat_ctx=ctx)

    assert stepfun_delegate.chat_calls == [ctx]
    assert deepseek_delegate.chat_calls == [ctx]


def test_weighted_strategy_can_be_pinned_to_one_provider(monkeypatch):
    _mix(monkeypatch, LLM_MIX_STRATEGY="weighted", LLM_MIX_WEIGHTS="deepseek:1,stepfun:0")
    router = providers.build_llm()

    for _ in range(20):
        router.chat(chat_ctx=llm.ChatContext.empty())
        assert router.provider == "DeepSeek"


def test_zero_total_weight_is_rejected(monkeypatch):
    _mix(monkeypatch, LLM_MIX_STRATEGY="weighted", LLM_MIX_WEIGHTS="stepfun:0,deepseek:0")
    with pytest.raises(ValueError, match="positive weight"):
        providers.build_llm()


def test_weights_naming_an_unconfigured_provider_are_rejected(monkeypatch):
    _mix(monkeypatch, LLM_MIX_WEIGHTS="openai:1")
    with pytest.raises(ValueError, match="not in LLM_MIX_PROVIDERS"):
        providers.build_llm()


def test_unknown_strategy_is_rejected(monkeypatch):
    _mix(monkeypatch, LLM_MIX_STRATEGY="random")
    with pytest.raises(ValueError, match="LLM_MIX_STRATEGY"):
        providers.build_llm()


def test_delegates_are_built_eagerly_so_bad_config_fails_at_startup(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mix")
    monkeypatch.setenv("LLM_MIX_PROVIDERS", "stepfun,deepseek")
    # No DEEPSEEK_API_KEY: the failure must surface during build, not on turn 2
    # with a caller already on the line.
    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY"):
        providers.build_llm()


async def test_every_delegate_is_prewarmed(monkeypatch):
    _mix(monkeypatch)
    router = providers.build_llm()
    router.prewarm()
    assert [d.prewarm_calls for d in router._delegates] == [1, 1]


def test_prewarm_outside_a_running_loop_is_a_no_op(monkeypatch):
    """`LLM.prewarm` falls back to `get_event_loop()`, which raises on 3.14+.

    A router that paid for two delegates would fail twice; prewarming is a
    latency optimisation, so it must never take startup down.
    """
    _mix(monkeypatch)
    providers.build_llm().prewarm()  # must not raise


def test_metrics_from_a_delegate_reach_listeners_on_the_router(monkeypatch):
    _mix(monkeypatch)
    router = providers.build_llm()
    received = []
    router.on("metrics_collected", received.append)

    event = MagicMock()
    # However the delegate is reached, the router re-emits: the eval controller
    # only ever listens on `session.llm`, which is the router.
    router._delegates[1].emit("metrics_collected", event)
    assert received == [event]


def test_describe_llm_records_the_whole_roster(monkeypatch):
    _mix(monkeypatch)
    described = providers.describe_llm(providers.build_llm())

    assert described["llm_provider"] == "mix"
    assert described["llm_strategy"] == "round_robin"
    assert described["llm_models"] == [
        {"llm_provider": "StepFun", "llm_model": "step-3.7-flash"},
        {"llm_provider": "DeepSeek", "llm_model": "deepseek-chat"},
    ]


def test_describe_llm_reports_a_single_provider_plainly():
    assert providers.describe_llm(FakeLLM("deepseek-chat")) == {
        "llm_provider": "deepseek",
        "llm_model": "deepseek-chat",
    }


def test_duplicate_mix_providers_are_rejected(monkeypatch):
    _mix(monkeypatch, LLM_MIX_PROVIDERS="stepfun,stepfun")
    with pytest.raises(ValueError, match="twice"):
        providers.build_llm()
