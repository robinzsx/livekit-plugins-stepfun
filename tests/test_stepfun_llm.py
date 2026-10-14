"""LLM wrapper tests.

Split by what each part needs:

  * Base-URL and model resolution is pure logic and always runs — it lives in
    `_utils` precisely so it is testable without the optional OpenAI plugin.
  * The subclass itself needs `livekit-plugins-openai`, so those tests skip when
    the `llm` extra is not installed. Skipping is deliberate: the ASR/TTS-only
    install is a supported configuration and its CI job must stay green.
"""

from __future__ import annotations

import importlib.util

import pytest

from livekit.plugins import stepfun
from livekit.plugins.stepfun import _utils
from livekit.plugins.stepfun.models import (
    DEFAULT_LLM_MODEL,
    REASONING_EFFORT_LLM_MODELS,
    REASONING_LLM_MODELS,
)

_HAS_OPENAI = importlib.util.find_spec("livekit.plugins.openai") is not None
requires_openai = pytest.mark.skipif(
    not _HAS_OPENAI, reason="needs the [llm] extra (livekit-plugins-openai)"
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_API_KEY", "test-key")
    monkeypatch.delenv("STEPFUN_CLUSTER", raising=False)


# ---------------------------------------------------------------------------
# Endpoint resolution (no optional dependency)
# ---------------------------------------------------------------------------


def test_llm_base_url_gets_version_suffix() -> None:
    # The OpenAI-compatible client expects the API version in the base URL
    # rather than in each request path.
    assert _utils.resolve_llm_base_url(None, "international") == "https://api.stepfun.ai/v1"
    assert _utils.resolve_llm_base_url(None, "domestic") == "https://api.stepfun.com/v1"


def test_llm_base_url_not_doubled() -> None:
    # A doubled /v1/v1 returns 404 in a way that is annoying to diagnose, so an
    # explicit base_url that already carries the suffix is left alone.
    assert (
        _utils.resolve_llm_base_url("https://api.stepfun.ai/v1", "international")
        == "https://api.stepfun.ai/v1"
    )


def test_llm_base_url_override_wins() -> None:
    assert (
        _utils.resolve_llm_base_url("https://proxy.internal", "international")
        == "https://proxy.internal/v1"
    )


def test_llm_base_url_strips_trailing_slash() -> None:
    assert (
        _utils.resolve_llm_base_url("https://api.stepfun.ai/", "international")
        == "https://api.stepfun.ai/v1"
    )


# ---------------------------------------------------------------------------
# Model metadata (no optional dependency)
# ---------------------------------------------------------------------------


def test_default_is_not_the_reasoning_heavyweight() -> None:
    """step-5-preview is a 1M-context reasoning model for agentic work.

    In a voice pipeline its thinking time lands entirely before the first
    speakable token, so the user hears silence for its whole duration. It is a
    supported choice, not a sensible default.
    """
    assert DEFAULT_LLM_MODEL == "step-3.7-flash"
    assert DEFAULT_LLM_MODEL != "step-5-preview"


def test_reasoning_model_sets_documented() -> None:
    # Only step-3.7-flash is documented as taking the three effort levels; both
    # of these return a `reasoning` field.
    assert "step-3.7-flash" in REASONING_EFFORT_LLM_MODELS
    assert {"step-5-preview", "step-3.7-flash"} <= REASONING_LLM_MODELS


def test_llm_models_exported() -> None:
    from typing import get_args

    models = set(get_args(stepfun.LLMModels))
    assert {"step-5-preview", "step-3.7-flash", "step-3.5-flash"} <= models


# ---------------------------------------------------------------------------
# Lazy import contract
# ---------------------------------------------------------------------------


def test_base_package_imports_without_llm_extra() -> None:
    """ASR and TTS must not depend on the OpenAI plugin.

    stepfun.LLM is resolved through a module __getattr__ so that an
    ASR/TTS-only install keeps working; importing it eagerly would break the
    whole package.
    """
    assert stepfun.STT is not None
    assert stepfun.TTS is not None


def test_unknown_attribute_still_raises_attribute_error() -> None:
    # The lazy __getattr__ must not swallow genuine typos.
    with pytest.raises(AttributeError, match="no attribute 'Nonexistent'"):
        _ = stepfun.Nonexistent  # type: ignore[attr-defined]


@pytest.mark.skipif(_HAS_OPENAI, reason="only meaningful without the extra")
def test_llm_access_explains_the_missing_extra() -> None:
    with pytest.raises(ImportError, match=r"livekit-plugins-stepfun\[llm\]"):
        _ = stepfun.LLM


# ---------------------------------------------------------------------------
# The subclass itself
# ---------------------------------------------------------------------------


@requires_openai
def test_llm_defaults_to_international_cluster() -> None:
    llm = stepfun.LLM()
    assert llm.model == DEFAULT_LLM_MODEL
    assert llm.provider == "StepFun"
    assert llm.cluster == "international"


@requires_openai
def test_llm_honours_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEPFUN_CLUSTER", "domestic")
    assert stepfun.LLM().cluster == "domestic"


@requires_openai
def test_llm_accepts_step_5_preview() -> None:
    llm = stepfun.LLM(model="step-5-preview")
    assert llm.model == "step-5-preview"


@requires_openai
def test_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        stepfun.LLM()


@requires_openai
def test_reasoning_effort_warns_on_undocumented_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Silently passing a parameter the server may ignore is worse than saying so:
    # the symptom would be unexplained latency, not an error.
    with caplog.at_level("WARNING", logger="livekit.plugins.stepfun"):
        stepfun.LLM(model="step-3.5-flash", reasoning_effort="low")
    assert "reasoning_effort" in caplog.text


@requires_openai
def test_reasoning_effort_silent_on_documented_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="livekit.plugins.stepfun"):
        stepfun.LLM(model="step-3.7-flash", reasoning_effort="low")
    assert "reasoning_effort" not in caplog.text
